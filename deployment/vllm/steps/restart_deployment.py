#!/usr/bin/env python3
"""
Step 2: Restart vLLM Deployment

Triggers a rolling restart of the vLLM deployment so the init-container
re-syncs adapters from S3. Waits for the rollout to complete.

Input Parameters:
    --deployment: Deployment name (default: vllm)
    --namespace: Kubernetes namespace (default: ml-models)
    --timeout: Rollout timeout in seconds (default: 600)
    --vllm-url: vLLM server URL for post-restart health check
    --output-dir: Directory to save output artifacts

Output Artifacts:
    restart_result.txt: "success" or "failed"

Note:
    Uses kubectl subprocess. The deployment image must include kubectl.
    `kubectl rollout restart` is ArgoCD-safe — it only sets a restartedAt annotation.
"""

import argparse
import os
import subprocess
import sys
import time

sys.path.insert(0, "/scripts")

from utils.vllm_client import VllmClient


def parse_arguments():
    parser = argparse.ArgumentParser(description="Restart vLLM deployment")

    parser.add_argument("--deployment", default="vllm", help="Deployment name")
    parser.add_argument("--namespace", default="ml-models", help="Kubernetes namespace")
    parser.add_argument("--timeout", type=int, default=600, help="Rollout timeout in seconds")
    parser.add_argument("--vllm-url", default="http://vllm-service.ml-models:8000", help="vLLM server URL")
    parser.add_argument("--output-dir", required=True, help="Directory to save output artifacts")

    return parser.parse_args()


def run_command(cmd: list, timeout: int = 60) -> subprocess.CompletedProcess:
    """Run a shell command and return the result."""
    print(f"  $ {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if result.stdout:
        print(f"  {result.stdout.strip()}")
    if result.returncode != 0 and result.stderr:
        print(f"  stderr: {result.stderr.strip()}")
    return result


def main():
    print("=" * 80)
    print("Step 2: Restart vLLM Deployment")
    print("=" * 80)

    args = parse_arguments()

    print(f"\nConfiguration:")
    print(f"  Deployment: {args.deployment}")
    print(f"  Namespace:  {args.namespace}")
    print(f"  Timeout:    {args.timeout}s")
    print(f"  vLLM URL:   {args.vllm_url}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Step 1: Trigger rolling restart
    print("\n" + "-" * 80)
    print("Triggering rolling restart...")

    result = run_command([
        "kubectl", "rollout", "restart",
        f"deployment/{args.deployment}",
        "-n", args.namespace,
    ])

    if result.returncode != 0:
        print(f"\nERROR: Failed to trigger rollout restart")
        _write_result(args.output_dir, "failed")
        sys.exit(1)

    print("Rollout restart triggered")

    # Step 2: Wait for rollout to complete
    print("\n" + "-" * 80)
    print(f"Waiting for rollout to complete (timeout: {args.timeout}s)...")

    result = run_command(
        [
            "kubectl", "rollout", "status",
            f"deployment/{args.deployment}",
            "-n", args.namespace,
            f"--timeout={args.timeout}s",
        ],
        timeout=args.timeout + 30,
    )

    if result.returncode != 0:
        print(f"\nERROR: Rollout did not complete within {args.timeout}s")
        _write_result(args.output_dir, "failed")
        sys.exit(1)

    print("Rollout completed")

    # Step 3: Wait for vLLM health
    print("\n" + "-" * 80)
    print("Waiting for vLLM health check...")

    vllm_client = VllmClient(args.vllm_url)

    try:
        vllm_client.wait_until_healthy(timeout=args.timeout, poll_interval=10)
    except Exception as e:
        print(f"\nERROR: vLLM did not become healthy: {e}")
        _write_result(args.output_dir, "failed")
        sys.exit(1)

    # Step 4: List available models
    print("\n" + "-" * 80)
    print("Available models after restart:")

    try:
        models = vllm_client.list_models()
        for model in models:
            print(f"  - {model}")
    except Exception as e:
        print(f"  Warning: Could not list models: {e}")

    _write_result(args.output_dir, "success")

    # Summary
    print("\n" + "=" * 80)
    print("Step 2 completed successfully!")
    print("=" * 80)

    sys.exit(0)


def _write_result(output_dir: str, result: str):
    path = os.path.join(output_dir, "restart_result.txt")
    with open(path, "w") as f:
        f.write(result)


if __name__ == "__main__":
    main()
