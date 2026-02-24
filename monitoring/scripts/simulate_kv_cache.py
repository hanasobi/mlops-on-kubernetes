#!/usr/bin/env python3
"""
Simulate KV Cache Saturation

Sends requests with very long contexts to fill the vLLM KV cache.
With max_model_len=4096, a few long-context requests can saturate the cache.

Prerequisites:
    kubectl port-forward svc/vllm-service 8000:8000 -n ml-models

Usage:
    python simulate_kv_cache.py --vllm-url http://localhost:8000

    # More aggressive
    python simulate_kv_cache.py --vllm-url http://localhost:8000 \
        --num-requests 20 --context-tokens 3500
"""

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

# A long filler text repeated to fill context. Each repetition is ~50 tokens.
FILLER_BLOCK = (
    "Amazon Web Services provides a broad set of global cloud-based products "
    "including compute, storage, databases, analytics, networking, mobile, "
    "developer tools, management tools, IoT, security, and enterprise "
    "applications on demand, available in seconds, with pay-as-you-go pricing. "
)


def build_long_prompt(target_tokens: int) -> str:
    """Build a prompt that's approximately target_tokens long."""
    # ~1.3 tokens per word, ~50 tokens per FILLER_BLOCK repetition
    repetitions = max(1, target_tokens // 50)
    context = FILLER_BLOCK * repetitions
    return (
        f"Based on the following context, provide a detailed summary:\n\n"
        f"{context}\n\n"
        f"Summarize the key points above in detail."
    )


def fire_request(vllm_url: str, model: str, prompt: str) -> dict:
    """Fire a single long-context request."""
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": 256,
        "temperature": 0.0,
    }
    start = time.time()
    try:
        resp = requests.post(
            f"{vllm_url.rstrip('/')}/v1/completions",
            json=payload,
            timeout=180,
        )
        resp.raise_for_status()
        elapsed = time.time() - start
        usage = resp.json().get("usage", {})
        return {
            "success": True,
            "latency": round(elapsed, 2),
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": usage.get("completion_tokens", 0),
        }
    except Exception as e:
        elapsed = time.time() - start
        return {"success": False, "latency": round(elapsed, 2), "error": str(e)}


def main():
    parser = argparse.ArgumentParser(
        description="Simulate KV cache saturation on vLLM"
    )
    parser.add_argument(
        "--vllm-url",
        default="http://localhost:8000",
        help="vLLM server URL (default: http://localhost:8000)",
    )
    parser.add_argument(
        "--model",
        default="aws-rag-qa-live",
        help="Model/adapter name (default: aws-rag-qa-live)",
    )
    parser.add_argument(
        "--num-requests",
        type=int,
        default=10,
        help="Number of long-context requests to send (default: 10)",
    )
    parser.add_argument(
        "--context-tokens",
        type=int,
        default=3000,
        help="Approximate context length in tokens (default: 3000, max_model_len=4096)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Number of concurrent requests (default: 5)",
    )
    args = parser.parse_args()

    print(f"Simulating KV cache saturation on {args.vllm_url}")
    print(f"  Model:          {args.model}")
    print(f"  Requests:       {args.num_requests}")
    print(f"  Context tokens: ~{args.context_tokens}")
    print(f"  Concurrency:    {args.concurrency}")
    print()

    prompt = build_long_prompt(args.context_tokens)
    total_success = 0
    total_errors = 0

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [
            pool.submit(fire_request, args.vllm_url, args.model, prompt)
            for _ in range(args.num_requests)
        ]

        for i, future in enumerate(as_completed(futures), 1):
            result = future.result()
            if result["success"]:
                total_success += 1
                print(
                    f"  [{i}/{args.num_requests}] OK "
                    f"({result['latency']}s, "
                    f"{result['prompt_tokens']} prompt tokens, "
                    f"{result['completion_tokens']} completion tokens)"
                )
            else:
                total_errors += 1
                print(
                    f"  [{i}/{args.num_requests}] FAIL "
                    f"({result['latency']}s: {result.get('error', 'unknown')})"
                )

    print(f"\nResults:")
    print(f"  Success: {total_success}")
    print(f"  Errors:  {total_errors}")
    print(f"\nCheck Grafana for KV cache usage spike (> 90% triggers alert).")


if __name__ == "__main__":
    main()
