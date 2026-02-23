#!/usr/bin/env python3
"""
Step 1: Load LoRA Adapter

Downloads a LoRA adapter from MLflow and loads it into the running
vLLM server via the runtime API — no pod restart required.

Flow:
  1. Download adapter from MLflow to local temp dir
  2. Find the vLLM pod
  3. kubectl cp adapter files to the vLLM pod
  4. Unload existing adapter (if loaded)
  5. Load new adapter via /v1/load_lora_adapter
  6. Verify adapter appears in /v1/models

Input Parameters:
    --model-name: MLflow registered model name (e.g. "mistral-7b-lora")
    --source: MLflow source (e.g. "alias:candidate" or "version:3")
    --slot-name: Adapter slot name (e.g. "aws-rag-qa-candidate")
    --deployment: vLLM deployment name (default: vllm)
    --namespace: Kubernetes namespace (default: ml-models)
    --vllm-url: vLLM server URL
    --output-dir: Directory to save output artifacts

Output Artifacts:
    load_result.json: Metadata (version, run_id, slot, etc.)
    load_result_status.txt: "success" or "failed"

Environment Variables:
    MLFLOW_TRACKING_URI: MLflow server URL

Note:
    Requires VLLM_ALLOW_RUNTIME_LORA_UPDATING=True on the vLLM server.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, "/scripts")

from utils.mlflow_helpers import MLflowHelper
from utils.vllm_client import VllmClient


def parse_arguments():
    parser = argparse.ArgumentParser(description="Load LoRA adapter from MLflow into running vLLM")

    parser.add_argument("--model-name", required=True, help="MLflow registered model name")
    parser.add_argument("--source", default="alias:candidate", help="MLflow source (alias:X or version:N)")
    parser.add_argument("--slot-name", required=True, help="Adapter slot name")
    parser.add_argument("--deployment", default="vllm", help="vLLM deployment name")
    parser.add_argument("--namespace", default="ml-models", help="Kubernetes namespace")
    parser.add_argument("--vllm-url", default="http://vllm-service.ml-models:8000", help="vLLM server URL")
    parser.add_argument("--output-dir", required=True, help="Directory to save output artifacts")

    return parser.parse_args()


def validate_environment():
    required_vars = ["MLFLOW_TRACKING_URI"]
    missing = [v for v in required_vars if v not in os.environ]
    if missing:
        print(f"ERROR: Missing required environment variables: {missing}")
        sys.exit(1)
    print(f"MLflow Tracking URI: {os.environ['MLFLOW_TRACKING_URI']}")


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

    pod_name = result.stdout.strip()
    print(f"Found vLLM pod: {pod_name}")
    return pod_name


def copy_adapter_to_pod(local_dir: str, pod_name: str, namespace: str, slot_name: str) -> None:
    """Copy adapter files to the vLLM pod via kubectl cp."""
    target_path = f"/mnt/adapters/{slot_name}/"

    print(f"Copying adapter files to {pod_name}:{target_path}")

    # Ensure target directory exists on the pod
    run_command([
        "kubectl", "exec", pod_name,
        "-n", namespace, "-c", "vllm",
        "--", "mkdir", "-p", target_path,
    ])

    # Copy each file
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

    # Verify files on pod
    run_command([
        "kubectl", "exec", pod_name,
        "-n", namespace, "-c", "vllm",
        "--", "ls", "-lh", target_path,
    ])

    print(f"All {len(files)} files copied to pod")


def main():
    print("=" * 80)
    print("Step 1: Load LoRA Adapter")
    print("=" * 80)

    args = parse_arguments()
    validate_environment()

    print(f"\nConfiguration:")
    print(f"  Model Name:  {args.model_name}")
    print(f"  Source:       {args.source}")
    print(f"  Slot Name:   {args.slot_name}")
    print(f"  Deployment:  {args.deployment}")
    print(f"  Namespace:   {args.namespace}")
    print(f"  vLLM URL:    {args.vllm_url}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Step 1: Download adapter from MLflow
    print("\n" + "-" * 80)
    print("Downloading adapter from MLflow...")

    mlflow_helper = MLflowHelper()
    temp_dir = tempfile.mkdtemp()

    try:
        metadata = mlflow_helper.download_adapter(
            args.model_name, args.source, temp_dir
        )

        print(f"\nAdapter downloaded successfully")
        print(f"  Version: {metadata['version']}")
        print(f"  Run ID:  {metadata['run_id']}")

        # Validate expected files
        expected_files = ["adapter_config.json", "adapter_model.safetensors"]
        for f in expected_files:
            if not os.path.exists(os.path.join(temp_dir, f)):
                print(f"ERROR: Expected file '{f}' not found in adapter artifacts")
                _write_results(args.output_dir, {}, "failed")
                sys.exit(1)

        # Step 2: Health check
        print("\n" + "-" * 80)
        print("Checking vLLM health...")

        vllm_client = VllmClient(args.vllm_url)

        try:
            vllm_client.wait_until_healthy(timeout=60, poll_interval=5)
        except Exception as e:
            print(f"\nERROR: vLLM is not healthy: {e}")
            _write_results(args.output_dir, metadata, "failed")
            sys.exit(1)

        # Step 3: Find vLLM pod and copy adapter files
        print("\n" + "-" * 80)
        print("Finding vLLM pod...")

        try:
            pod_name = find_vllm_pod(args.deployment, args.namespace)
        except Exception as e:
            print(f"\nERROR: {e}")
            _write_results(args.output_dir, metadata, "failed")
            sys.exit(1)

        print("\n" + "-" * 80)
        print("Copying adapter files to vLLM pod...")

        try:
            copy_adapter_to_pod(temp_dir, pod_name, args.namespace, args.slot_name)
        except Exception as e:
            print(f"\nERROR: Failed to copy adapter: {e}")
            _write_results(args.output_dir, metadata, "failed")
            sys.exit(1)

        # Step 4: Unload existing adapter (ignore error if not loaded)
        print("\n" + "-" * 80)
        print(f"Unloading adapter '{args.slot_name}' (if loaded)...")

        try:
            vllm_client.unload_lora_adapter(args.slot_name)
        except Exception as e:
            print(f"  Adapter not currently loaded (expected on first deploy): {e}")

        # Brief pause to let vLLM clean up
        time.sleep(2)

        # Step 5: Load new adapter
        print("\n" + "-" * 80)
        print(f"Loading adapter '{args.slot_name}'...")

        adapter_path = f"/mnt/adapters/{args.slot_name}"

        try:
            vllm_client.load_lora_adapter(args.slot_name, adapter_path)
        except Exception as e:
            print(f"\nERROR: Failed to load adapter: {e}")
            _write_results(args.output_dir, metadata, "failed")
            sys.exit(1)

        # Step 6: Verify adapter is available
        print("\n" + "-" * 80)
        print("Verifying adapter is available...")

        try:
            vllm_client.wait_until_model_available(args.slot_name, timeout=60, poll_interval=5)
        except Exception as e:
            print(f"\nERROR: Adapter not available after loading: {e}")
            _write_results(args.output_dir, metadata, "failed")
            sys.exit(1)

        # List all models
        try:
            models = vllm_client.list_models()
            print(f"Available models: {models}")
        except Exception as e:
            print(f"  Warning: Could not list models: {e}")

        # Save metadata
        metadata["slot_name"] = args.slot_name
        _write_results(args.output_dir, metadata, "success")

        # Summary
        print("\n" + "=" * 80)
        print("Step 1 completed successfully!")
        print("=" * 80)
        print(f"\nAdapter loaded:")
        print(f"  MLflow: {args.model_name} ({args.source}) version {metadata['version']}")
        print(f"  Slot:   {args.slot_name}")
        print(f"  Method: Runtime API (no restart)")

        sys.exit(0)

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def _write_results(output_dir: str, metadata: dict, status: str):
    """Write result metadata and status files."""
    results_path = os.path.join(output_dir, "load_result.json")
    with open(results_path, "w") as f:
        json.dump({"status": status, **metadata}, f, indent=2, default=str)

    status_path = os.path.join(output_dir, "load_result_status.txt")
    with open(status_path, "w") as f:
        f.write(status)


if __name__ == "__main__":
    main()
