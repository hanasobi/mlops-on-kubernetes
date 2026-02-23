#!/usr/bin/env python3
"""
Performance Gate Step: Run Load Test

Generates concurrent load against a LoRA adapter via the vLLM
OpenAI-compatible API. Performance metrics (latency, TTFT, throughput)
are collected from Prometheus (server-side), not client-side timing.
This ensures we measure adapter performance without test-system noise.

The script:
1. Verifies vLLM health and adapter availability
2. Fires concurrent requests using eval data prompts
3. Waits for Prometheus to scrape the latest metrics
4. Queries Prometheus for P50/P95/P99 latency, TTFT, throughput, etc.
5. Writes results as JSON artifact for the decision step

Input Parameters:
    --vllm-url: vLLM server URL
    --adapter-name: Adapter name (e.g. "aws-rag-qa-candidate")
    --eval-data-path: Path to eval.jsonl
    --max-tokens: Maximum tokens to generate (default: 256)
    --temperature: Sampling temperature (default: 0.0)
    --concurrency: Number of parallel requests (default: 10)
    --max-samples: Max samples to use, 0 for all (default: 0)
    --prometheus-url: Prometheus server URL
    --output-dir: Directory to save output artifacts

Output Artifacts:
    load_test_results.json: Prometheus metrics summary + client stats
    load_test_status.txt: "completed" or "failed"
"""

import argparse
import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

sys.path.insert(0, "/scripts")
sys.path.insert(0, "/perf-scripts")

from utils.vllm_client import VllmClient


# PromQL queries for vLLM metrics (WINDOW replaced at query time)
PROMQL_QUERIES = {
    "p50_latency_s": 'histogram_quantile(0.50, rate(vllm:e2e_request_latency_seconds_bucket{{job="vllm-service"}}[{window}]))',
    "p95_latency_s": 'histogram_quantile(0.95, rate(vllm:e2e_request_latency_seconds_bucket{{job="vllm-service"}}[{window}]))',
    "p99_latency_s": 'histogram_quantile(0.99, rate(vllm:e2e_request_latency_seconds_bucket{{job="vllm-service"}}[{window}]))',
    "p50_ttft_s": 'histogram_quantile(0.50, rate(vllm:time_to_first_token_seconds_bucket{{job="vllm-service"}}[{window}]))',
    "p95_ttft_s": 'histogram_quantile(0.95, rate(vllm:time_to_first_token_seconds_bucket{{job="vllm-service"}}[{window}]))',
    "p99_ttft_s": 'histogram_quantile(0.99, rate(vllm:time_to_first_token_seconds_bucket{{job="vllm-service"}}[{window}]))',
    "throughput_rps": 'sum(rate(vllm:request_success_total{{job="vllm-service"}}[{window}]))',
    "kv_cache_usage": 'vllm:kv_cache_usage_perc{{job="vllm-service"}}',
    "gpu_utilization": 'DCGM_FI_DEV_GPU_UTIL{{gpu="0"}}',
}


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run load test on candidate adapter")

    parser.add_argument("--vllm-url", default="http://vllm-service.ml-models:8000",
                        help="vLLM server URL")
    parser.add_argument("--adapter-name", required=True,
                        help="Adapter name to test")
    parser.add_argument("--eval-data-path", required=True,
                        help="Path to eval.jsonl")
    parser.add_argument("--max-tokens", type=int, default=256,
                        help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature")
    parser.add_argument("--concurrency", type=int, default=10,
                        help="Number of parallel requests")
    parser.add_argument("--max-samples", type=int, default=0,
                        help="Max samples to use (0 = all)")
    parser.add_argument("--prometheus-url", default="http://monitoring-kube-prometheus-prometheus.ai-platform:9090",
                        help="Prometheus server URL")
    parser.add_argument("--output-dir", required=True,
                        help="Directory to save output artifacts")

    return parser.parse_args()


def load_prompts(eval_data_path: str, max_samples: int) -> list:
    """Load prompts from eval.jsonl, extracting the prompt_inference field."""
    prompts = []
    with open(eval_data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                sample = json.loads(line)
                prompt = sample.get("prompt_inference", "")
                if prompt:
                    prompts.append(prompt)

    if max_samples > 0 and len(prompts) > max_samples:
        prompts = prompts[:max_samples]

    return prompts


def fire_request(vllm_url: str, adapter_name: str, prompt: str,
                 max_tokens: int, temperature: float) -> dict:
    """Fire a single request. Returns success/error only -- metrics come from Prometheus."""
    payload = {
        "model": adapter_name,
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    try:
        response = requests.post(
            f"{vllm_url.rstrip('/')}/v1/completions",
            json=payload,
            timeout=120,
        )
        response.raise_for_status()
        return {"success": True, "error": None}
    except Exception as e:
        return {"success": False, "error": str(e)}


def query_prometheus(prometheus_url: str, query: str) -> float:
    """Query Prometheus HTTP API. Returns float value or None."""
    try:
        response = requests.get(
            f"{prometheus_url.rstrip('/')}/api/v1/query",
            params={"query": query},
            timeout=10,
        )
        response.raise_for_status()
        data = response.json()
        results = data.get("data", {}).get("result", [])
        if results:
            value = float(results[0]["value"][1])
            # NaN from histogram_quantile when no data
            if math.isnan(value) or math.isinf(value):
                return None
            return round(value, 4)
    except Exception as e:
        print(f"  Warning: Prometheus query failed: {e}")
        print(f"  Query: {query}")
    return None


def collect_prometheus_metrics(prometheus_url: str, window: str) -> dict:
    """Query all performance metrics from Prometheus."""
    metrics = {}
    for name, query_template in PROMQL_QUERIES.items():
        query = query_template.format(window=window)
        metrics[name] = query_prometheus(prometheus_url, query)
    return metrics


def compute_window(duration_s: float) -> str:
    """Compute Prometheus query window from test duration.

    Rounds up to the nearest minute, minimum 1m.
    """
    minutes = max(1, math.ceil(duration_s / 60))
    return f"{minutes}m"


def main():
    print("=" * 80)
    print("Performance Gate Step: Run Load Test")
    print("=" * 80)

    args = parse_arguments()

    print(f"\nConfiguration:")
    print(f"  vLLM URL:        {args.vllm_url}")
    print(f"  Adapter:         {args.adapter_name}")
    print(f"  Eval Data:       {args.eval_data_path}")
    print(f"  Max Tokens:      {args.max_tokens}")
    print(f"  Temperature:     {args.temperature}")
    print(f"  Concurrency:     {args.concurrency}")
    print(f"  Max Samples:     {args.max_samples if args.max_samples > 0 else 'all'}")
    print(f"  Prometheus URL:  {args.prometheus_url}")

    os.makedirs(args.output_dir, exist_ok=True)

    client = VllmClient(args.vllm_url)

    # Health check
    print("\n" + "-" * 80)
    print("Checking vLLM health...")
    try:
        client.wait_until_healthy(timeout=120, poll_interval=5)
    except Exception as e:
        print(f"ERROR: vLLM is not healthy: {e}")
        _write_results(args.output_dir, {}, "failed")
        sys.exit(1)

    # Verify adapter is available
    print("Checking adapter availability...")
    try:
        models = client.list_models()
        print(f"Available models: {models}")
        if args.adapter_name not in models:
            print(f"ERROR: Adapter '{args.adapter_name}' not found")
            _write_results(args.output_dir, {}, "failed")
            sys.exit(1)
        print(f"Adapter '{args.adapter_name}' is available")
    except Exception as e:
        print(f"ERROR: Failed to list models: {e}")
        _write_results(args.output_dir, {}, "failed")
        sys.exit(1)

    # Load prompts from eval data
    print("\n" + "-" * 80)
    print("Loading test prompts from eval data...")
    prompts = load_prompts(args.eval_data_path, args.max_samples)
    print(f"Loaded {len(prompts)} prompts")

    if not prompts:
        print("ERROR: No prompts loaded")
        _write_results(args.output_dir, {}, "failed")
        sys.exit(1)

    # Run load test
    print("\n" + "-" * 80)
    print(f"Running load test: {len(prompts)} requests at concurrency {args.concurrency}...")

    successful = 0
    failed = 0
    completed = 0

    start_time = time.time()

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {}
        for i, prompt in enumerate(prompts):
            future = executor.submit(
                fire_request, args.vllm_url, args.adapter_name,
                prompt, args.max_tokens, args.temperature,
            )
            futures[future] = i

        for future in as_completed(futures):
            result = future.result()
            completed += 1

            if result["success"]:
                successful += 1
            else:
                failed += 1

            if completed % 50 == 0 or completed == len(prompts):
                print(f"  Progress: {completed}/{len(prompts)} "
                      f"(ok={successful}, errors={failed})")

    end_time = time.time()
    wall_clock_duration = round(end_time - start_time, 2)
    client_success_rate = round(successful / len(prompts), 4) if prompts else 0.0

    print(f"\nLoad generation complete:")
    print(f"  Duration:     {wall_clock_duration}s")
    print(f"  Successful:   {successful}/{len(prompts)}")
    print(f"  Success Rate: {client_success_rate:.2%}")

    # Wait for Prometheus scrape to catch up
    scrape_wait = 20
    print(f"\n" + "-" * 80)
    print(f"Waiting {scrape_wait}s for Prometheus scrape cycle...")
    time.sleep(scrape_wait)

    # Query Prometheus for server-side metrics
    window = compute_window(wall_clock_duration)
    print(f"Querying Prometheus (window={window})...")

    metrics = collect_prometheus_metrics(args.prometheus_url, window)

    print(f"\n  Server-side metrics (from Prometheus):")
    print(f"    P50 Latency:      {metrics.get('p50_latency_s')}s")
    print(f"    P95 Latency:      {metrics.get('p95_latency_s')}s")
    print(f"    P99 Latency:      {metrics.get('p99_latency_s')}s")
    print(f"    P50 TTFT:         {metrics.get('p50_ttft_s')}s")
    print(f"    P95 TTFT:         {metrics.get('p95_ttft_s')}s")
    print(f"    P99 TTFT:         {metrics.get('p99_ttft_s')}s")
    print(f"    Throughput:       {metrics.get('throughput_rps')} req/s")
    print(f"    KV-Cache Usage:   {metrics.get('kv_cache_usage')}")
    print(f"    GPU Utilization:  {metrics.get('gpu_utilization')}")

    # Build output
    summary = {
        "adapter": args.adapter_name,
        "concurrency": args.concurrency,
        "total_requests": len(prompts),
        "successful_requests": successful,
        "failed_requests": failed,
        "client_success_rate": client_success_rate,
        "wall_clock_duration_s": wall_clock_duration,
        "prometheus_window": window,
        **metrics,
    }

    output = {"summary": summary}
    _write_results(args.output_dir, output, "completed")

    print("\n" + "=" * 80)
    print(f"Load test completed for adapter '{args.adapter_name}'")
    print("=" * 80)
    sys.exit(0)


def _write_results(output_dir: str, output: any, status: str):
    """Write results and status to output files."""
    results_path = os.path.join(output_dir, "load_test_results.json")
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)

    status_path = os.path.join(output_dir, "load_test_status.txt")
    with open(status_path, "w") as f:
        f.write(status)


if __name__ == "__main__":
    main()
