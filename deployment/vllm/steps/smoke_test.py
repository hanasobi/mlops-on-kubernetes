#!/usr/bin/env python3
"""
Step 3: Smoke Test

Sends test inference requests to the deployed LoRA adapter via the
vLLM OpenAI-compatible API. Validates that the adapter loads correctly
and can generate responses.

Uses /v1/completions (not /v1/chat/completions) because the base model
(Mistral-7B-v0.1-AWQ) is a base model without a chat template.

Input Parameters:
    --vllm-url: vLLM server URL
    --adapter-name: Adapter name to test (e.g. "aws-rag-qa-candidate")
    --num-requests: Number of test requests (default: 3)
    --max-latency: Maximum acceptable latency in seconds (default: 30)
    --output-dir: Directory to save output artifacts

Output Artifacts:
    smoke_test_result.json: Detailed test results
    smoke_test_status.txt: "passed" or "failed"
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, "/scripts")

from utils.vllm_client import VllmClient

# Simple test prompts for smoke testing (text completion format)
TEST_PROMPTS = [
    {
        "prompt": "Question: What is Amazon S3?\nAnswer:",
        "description": "Basic AWS knowledge question",
    },
    {
        "prompt": "Question: Explain what a VPC is in one sentence.\nAnswer:",
        "description": "Short answer question",
    },
    {
        "prompt": "Question: What are the benefits of using IAM roles?\nAnswer:",
        "description": "List-style question",
    },
]


def parse_arguments():
    parser = argparse.ArgumentParser(description="Smoke test for deployed LoRA adapter")

    parser.add_argument("--vllm-url", default="http://vllm-service.ml-models:8000", help="vLLM server URL")
    parser.add_argument("--adapter-name", required=True, help="Adapter name to test")
    parser.add_argument("--num-requests", type=int, default=3, help="Number of test requests")
    parser.add_argument("--max-latency", type=float, default=30.0, help="Max acceptable latency (seconds)")
    parser.add_argument("--output-dir", required=True, help="Directory to save output artifacts")

    return parser.parse_args()


def run_single_test(vllm_client: VllmClient, adapter_name: str, prompt: dict, max_latency: float) -> dict:
    """Run a single inference test and validate the response."""
    result = {
        "description": prompt["description"],
        "passed": False,
        "checks": [],
    }

    start_time = time.time()

    try:
        response = vllm_client.completion(
            model=adapter_name,
            prompt=prompt["prompt"],
            max_tokens=64,
            temperature=0.1,
        )
        latency = time.time() - start_time
        result["latency_s"] = round(latency, 2)

        # Check 1: Response has choices
        choices = response.get("choices", [])
        if not choices:
            result["checks"].append("FAIL: No choices in response")
            return result
        result["checks"].append("PASS: Response has choices")

        # Check 2: Response has text content
        text = choices[0].get("text", "")
        if not text.strip():
            result["checks"].append("FAIL: Empty response text")
            return result
        result["checks"].append(f"PASS: Response has text ({len(text)} chars)")
        result["response_preview"] = text[:200]

        # Check 3: Latency within threshold
        if latency <= max_latency:
            result["checks"].append(f"PASS: Latency {latency:.2f}s <= {max_latency}s")
        else:
            result["checks"].append(f"FAIL: Latency {latency:.2f}s > {max_latency}s")
            return result

        # Check 4: Finish reason
        finish_reason = choices[0].get("finish_reason", "unknown")
        result["checks"].append(f"INFO: Finish reason: {finish_reason}")

        result["passed"] = True

    except Exception as e:
        result["error"] = str(e)
        result["checks"].append(f"FAIL: Request failed: {e}")

    return result


def main():
    print("=" * 80)
    print("Step 3: Smoke Test")
    print("=" * 80)

    args = parse_arguments()

    print(f"\nConfiguration:")
    print(f"  vLLM URL:     {args.vllm_url}")
    print(f"  Adapter:      {args.adapter_name}")
    print(f"  Num Requests: {args.num_requests}")
    print(f"  Max Latency:  {args.max_latency}s")

    os.makedirs(args.output_dir, exist_ok=True)

    vllm_client = VllmClient(args.vllm_url)

    # Step 1: Verify adapter is available
    print("\n" + "-" * 80)
    print("Checking adapter availability...")

    try:
        models = vllm_client.list_models()
        print(f"Available models: {models}")

        if args.adapter_name not in models:
            print(f"\nERROR: Adapter '{args.adapter_name}' not found in available models")
            _write_results(args.output_dir, [], "failed")
            sys.exit(1)

        print(f"Adapter '{args.adapter_name}' is available")

    except Exception as e:
        print(f"\nERROR: Failed to list models: {e}")
        _write_results(args.output_dir, [], "failed")
        sys.exit(1)

    # Step 2: Run test requests
    print("\n" + "-" * 80)
    print(f"Running {args.num_requests} smoke test requests...")

    test_results = []
    prompts = TEST_PROMPTS[: args.num_requests]

    for i, prompt in enumerate(prompts):
        print(f"\nTest {i + 1}/{len(prompts)}: {prompt['description']}")

        result = run_single_test(vllm_client, args.adapter_name, prompt, args.max_latency)
        test_results.append(result)

        for check in result["checks"]:
            print(f"  {check}")

        if result["passed"]:
            print(f"  Result: PASSED ({result.get('latency_s', '?')}s)")
        else:
            print(f"  Result: FAILED")

    # Step 3: Summarize
    print("\n" + "-" * 80)
    passed = sum(1 for r in test_results if r["passed"])
    total = len(test_results)
    all_passed = passed == total

    print(f"Results: {passed}/{total} tests passed")

    if all_passed:
        latencies = [r["latency_s"] for r in test_results if "latency_s" in r]
        if latencies:
            print(f"Average latency: {sum(latencies) / len(latencies):.2f}s")

    status = "passed" if all_passed else "failed"
    _write_results(args.output_dir, test_results, status)

    # Final output
    print("\n" + "=" * 80)
    if all_passed:
        print("Step 3 completed successfully!")
        print("=" * 80)
        print(f"\nSmoke test PASSED for adapter '{args.adapter_name}'")
        sys.exit(0)
    else:
        print("Step 3 FAILED!")
        print("=" * 80)
        print(f"\nSmoke test FAILED for adapter '{args.adapter_name}'")
        sys.exit(1)


def _write_results(output_dir: str, test_results: list, status: str):
    """Write test results and status to output files."""
    # Detailed results
    results_path = os.path.join(output_dir, "smoke_test_result.json")
    with open(results_path, "w") as f:
        json.dump({"status": status, "tests": test_results}, f, indent=2)

    # Simple status for Argo output parameter
    status_path = os.path.join(output_dir, "smoke_test_status.txt")
    with open(status_path, "w") as f:
        f.write(status)


if __name__ == "__main__":
    main()
