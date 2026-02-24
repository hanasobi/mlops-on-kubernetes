#!/usr/bin/env python3
"""
Simulate Latency Spike

Floods vLLM with concurrent requests to trigger P95 latency and TTFT alerts.
Reuses the same request pattern as the performance load test pipeline.

Prerequisites:
    kubectl port-forward svc/vllm-service 8000:8000 -n ml-models

Usage:
    python simulate_latency.py --vllm-url http://localhost:8000

    # Higher concurrency for stronger spike
    python simulate_latency.py --vllm-url http://localhost:8000 \
        --concurrency 20 --duration 180

    # With specific adapter
    python simulate_latency.py --vllm-url http://localhost:8000 \
        --model aws-rag-qa-live
"""

import argparse
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

DEFAULT_PROMPTS = [
    "Explain in great detail how Amazon EKS manages the Kubernetes control plane, including the architecture of etcd clusters, API server high availability, and how upgrades are performed across multiple availability zones. Cover the networking aspects including VPC CNI plugin, pod networking, and service discovery mechanisms.",
    "Write a comprehensive guide on implementing a multi-region disaster recovery strategy using AWS services. Include detailed steps for RDS cross-region read replicas, S3 cross-region replication, Route 53 failover routing, and CloudFormation StackSets. Provide example configurations and explain RPO/RTO tradeoffs.",
    "Describe the complete lifecycle of an HTTP request through an AWS Application Load Balancer, including TLS termination, target group health checks, connection draining, sticky sessions, and how WAF rules are evaluated. Include details about access logging and CloudWatch metrics integration.",
    "Explain how AWS Lambda cold starts work at a deep technical level, including the microVM architecture with Firecracker, execution environment reuse, provisioned concurrency implementation, and memory allocation strategies. Discuss how different runtimes affect cold start performance.",
    "Provide a detailed analysis of Amazon DynamoDB's internal architecture, including the partition management system, consistent hashing, Paxos-based replication, adaptive capacity, and how global secondary indexes maintain eventual consistency. Cover the request router and storage node interactions.",
]


def fire_request(
    vllm_url: str, model: str, prompt: str, max_tokens: int
) -> dict:
    """Fire a single completion request."""
    payload = {
        "model": model,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.7,
    }
    start = time.time()
    try:
        resp = requests.post(
            f"{vllm_url.rstrip('/')}/v1/completions",
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        elapsed = time.time() - start
        return {"success": True, "latency": round(elapsed, 2)}
    except Exception as e:
        elapsed = time.time() - start
        return {"success": False, "latency": round(elapsed, 2), "error": str(e)}


def main():
    parser = argparse.ArgumentParser(description="Simulate latency spike on vLLM")
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
        "--concurrency",
        type=int,
        default=10,
        help="Number of concurrent requests (default: 10)",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=120,
        help="Duration in seconds (default: 120)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Max tokens per request (default: 512)",
    )
    args = parser.parse_args()

    print(f"Simulating latency spike on {args.vllm_url}")
    print(f"  Model:       {args.model}")
    print(f"  Concurrency: {args.concurrency}")
    print(f"  Duration:    {args.duration}s")
    print(f"  Max tokens:  {args.max_tokens}")
    print()

    end_time = time.time() + args.duration
    total_requests = 0
    total_success = 0
    total_errors = 0
    prompt_idx = 0

    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {}

        while time.time() < end_time:
            # Keep the pool full
            while len(futures) < args.concurrency and time.time() < end_time:
                prompt = DEFAULT_PROMPTS[prompt_idx % len(DEFAULT_PROMPTS)]
                prompt_idx += 1
                future = pool.submit(
                    fire_request, args.vllm_url, args.model, prompt, args.max_tokens
                )
                futures[future] = time.time()

            # Collect completed futures
            done = [f for f in futures if f.done()]
            for f in done:
                result = f.result()
                total_requests += 1
                if result["success"]:
                    total_success += 1
                else:
                    total_errors += 1
                    print(f"  Error: {result.get('error', 'unknown')}")
                del futures[f]

            time.sleep(0.1)

        # Wait for remaining
        for f in as_completed(futures):
            result = f.result()
            total_requests += 1
            if result["success"]:
                total_success += 1
            else:
                total_errors += 1

    elapsed = args.duration
    print(f"\nResults ({elapsed}s):")
    print(f"  Total requests: {total_requests}")
    print(f"  Success:        {total_success}")
    print(f"  Errors:         {total_errors}")
    print(f"  Throughput:     {total_requests / elapsed:.2f} req/s")
    print(f"\nCheck Grafana for P95 latency and TTFT spikes.")


if __name__ == "__main__":
    main()
