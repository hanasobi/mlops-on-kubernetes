#!/usr/bin/env python3
"""
Promote LoRA Adapter

Promotes a source adapter version to the live slot:
1. Downloads adapter from MLflow (source alias version)
2. Copies adapter files to the live slot on the vLLM pod
3. Unloads old live adapter, loads new one via runtime API
4. Updates MLflow aliases: source version → live, old live → previous
5. Cleans up intermediate gate aliases (eval-passed, perf-passed, etc.)

Supports rollback by using --source-alias=previous.

Usage:
    python promote_adapter.py \
        --model-name mistral-7b-lora \
        --deployment vllm \
        --namespace ml-models \
        --vllm-url http://vllm-service.ml-models:8000

    # Rollback to previous version:
    python promote_adapter.py \
        --model-name mistral-7b-lora \
        --source-alias previous

Environment Variables:
    MLFLOW_TRACKING_URI: MLflow server URL
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, "/scripts")

from utils.mlflow_helpers import MLflowHelper
from utils.vllm_client import VllmClient

SOURCE_SLOT = "aws-rag-qa-candidate"
TARGET_SLOT = "aws-rag-qa-live"


def parse_arguments():
    parser = argparse.ArgumentParser(description="Promote LoRA adapter from candidate to live")

    parser.add_argument("--model-name", required=True, help="MLflow registered model name")
    parser.add_argument("--source-alias", default="staged",
                        help="MLflow alias to promote (default: staged, use 'previous' for rollback)")
    parser.add_argument("--deployment", default="vllm", help="vLLM deployment name")
    parser.add_argument("--namespace", default="ml-models", help="Kubernetes namespace")
    parser.add_argument("--vllm-url", default="http://vllm-service.ml-models:8000", help="vLLM server URL")

    return parser.parse_args()


def run_command(cmd: list, timeout: int = 120) -> subprocess.CompletedProcess:
    """Run a shell command and return the result."""
    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if result.stdout:
        print(f"  {result.stdout.strip()}")
    if result.returncode != 0 and result.stderr:
        print(f"  stderr: {result.stderr.strip()}")
    return result


def find_vllm_pod(deployment: str, namespace: str) -> str:
    """Find the running vLLM pod name."""
    result = run_command([
        "kubectl", "get", "pods",
        "-n", namespace,
        "-l", f"app={deployment}",
        "--field-selector=status.phase=Running",
        "-o", "jsonpath={.items[0].metadata.name}",
    ])

    if result.returncode != 0 or not result.stdout.strip():
        raise Exception(f"Could not find running pod for deployment '{deployment}'")

    return result.stdout.strip()


def copy_adapter_to_pod(local_dir: str, pod_name: str, namespace: str, slot_name: str) -> None:
    """Copy adapter files to the vLLM pod via kubectl cp."""
    target_path = f"/mnt/adapters/{slot_name}/"

    run_command([
        "kubectl", "exec", pod_name,
        "-n", namespace, "-c", "vllm",
        "--", "mkdir", "-p", target_path,
    ])

    files = [f for f in os.listdir(local_dir) if os.path.isfile(os.path.join(local_dir, f))]

    for filename in files:
        local_file = os.path.join(local_dir, filename)
        remote_path = f"{namespace}/{pod_name}:{target_path}{filename}"

        print(f"  Copying {filename}...")
        result = run_command([
            "kubectl", "cp", local_file, remote_path,
            "-c", "vllm",
        ])

        if result.returncode != 0:
            raise Exception(f"Failed to copy {filename} to pod: {result.stderr}")


def main():
    print("=" * 80)
    print("Promote LoRA Adapter")
    print("=" * 80)

    args = parse_arguments()

    print(f"\nConfiguration:")
    print(f"  Model Name:   {args.model_name}")
    print(f"  Source Slot:   {SOURCE_SLOT}")
    print(f"  Target Slot:   {TARGET_SLOT}")

    mlflow_helper = MLflowHelper()
    vllm_client = VllmClient(args.vllm_url)

    # Step 1: Resolve the source version from MLflow
    print("\n" + "-" * 80)
    print(f"Resolving '{args.source_alias}' adapter version...")

    source_info = mlflow_helper.get_model_version_by_alias(args.model_name, args.source_alias)
    if not source_info:
        print(f"ERROR: No '{args.source_alias}' alias found in MLflow. Nothing to promote.")
        sys.exit(1)

    version = source_info["version"]
    print(f"Source version ({args.source_alias}): {version}")

    # Step 2: Download adapter from MLflow
    print("\n" + "-" * 80)
    print("Downloading adapter from MLflow...")

    temp_dir = tempfile.mkdtemp()
    try:
        metadata = mlflow_helper.download_adapter(
            args.model_name, f"version:{version}", temp_dir
        )
        print(f"Adapter downloaded (version {version})")

        # Step 3: Copy to live slot on vLLM pod
        print("\n" + "-" * 80)
        print("Copying adapter to live slot on vLLM pod...")

        pod_name = find_vllm_pod(args.deployment, args.namespace)
        copy_adapter_to_pod(temp_dir, pod_name, args.namespace, TARGET_SLOT)
        print(f"Adapter copied to {TARGET_SLOT}")

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)

    # Step 4: Unload old live adapter, load new one
    print("\n" + "-" * 80)
    print(f"Reloading live adapter...")

    try:
        vllm_client.unload_lora_adapter(TARGET_SLOT)
    except Exception as e:
        print(f"  No existing live adapter to unload: {e}")

    import time
    time.sleep(2)

    vllm_client.load_lora_adapter(TARGET_SLOT, f"/mnt/adapters/{TARGET_SLOT}")
    vllm_client.wait_until_model_available(TARGET_SLOT, timeout=60, poll_interval=5)
    print(f"Live adapter loaded and verified")

    # Step 5: Update MLflow aliases
    print("\n" + "-" * 80)
    print("Updating MLflow aliases...")

    updates = mlflow_helper.update_deployment_aliases(args.model_name, version)
    print(f"Aliases updated: {updates}")

    # Remove staged alias (version is now live)
    try:
        mlflow_helper.delete_alias(args.model_name, "staged")
        print("Removed 'staged' alias")
    except Exception:
        pass

    # Remove gate aliases (version is now live)
    for gate_alias in ["eval-passed", "perf-passed", "eval-failed", "perf-failed"]:
        try:
            mlflow_helper.delete_alias(args.model_name, gate_alias)
            print(f"Removed '{gate_alias}' alias")
        except Exception:
            pass

    # Summary
    print("\n" + "=" * 80)
    print("Promotion completed!")
    print("=" * 80)
    print(f"\n  Version {version}: {args.source_alias} -> live")
    print(f"  {SOURCE_SLOT} adapter promoted to {TARGET_SLOT}")

    sys.exit(0)


if __name__ == "__main__":
    main()
