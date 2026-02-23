#!/usr/bin/env python3
"""
Evaluation Step: Run Inference

Sends evaluation samples to a single LoRA adapter via the vLLM
OpenAI-compatible API and collects model answers. Called twice in
parallel (once per adapter: candidate and live).

Uses /v1/completions (base model endpoint) with pre-formatted prompts
from eval.jsonl. Includes deterministic refusal detection for negative
samples.

Input Parameters:
    --vllm-url: vLLM server URL
    --adapter-name: Adapter name (e.g. "aws-rag-qa-candidate")
    --eval-data-path: Path to eval.jsonl
    --max-tokens: Maximum tokens to generate (default: 512)
    --temperature: Sampling temperature (default: 0.0)
    --max-concurrent: Maximum concurrent requests (default: 15)
    --output-dir: Directory to save output artifacts

Output Artifacts:
    inference_results.json: Per-sample results with model answers
    inference_status.txt: "completed" or "failed"
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, "/scripts")
sys.path.insert(0, "/eval-scripts")

from utils.vllm_client import VllmClient
from eval_utils.refusal import is_refusal


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run inference on eval dataset")

    parser.add_argument("--vllm-url", default="http://vllm-service.ml-models:8000", help="vLLM server URL")
    parser.add_argument("--adapter-name", required=True, help="Adapter name to evaluate")
    parser.add_argument("--eval-data-path", required=True, help="Path to eval.jsonl")
    parser.add_argument("--max-tokens", type=int, default=512, help="Max tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--max-concurrent", type=int, default=15, help="Max concurrent requests")
    parser.add_argument("--output-dir", required=True, help="Directory to save output artifacts")

    return parser.parse_args()


def load_eval_data(eval_data_path: str) -> list:
    """Load evaluation samples from JSONL file."""
    samples = []
    with open(eval_data_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def run_single_inference(client: VllmClient, adapter_name: str, sample: dict,
                         index: int, max_tokens: int, temperature: float) -> dict:
    """Run inference for a single sample with retry logic."""
    prompt = sample.get("prompt_inference", "")
    question_type = sample.get("metadata", {}).get("question_type", "factual")

    result = {
        "index": index,
        "question": sample.get("question", ""),
        "context": sample.get("context", ""),
        "reference_answer": sample.get("reference_answer", ""),
        "question_type": question_type,
        "adapter": adapter_name,
        "model_answer": "",
        "is_refusal": False,
        "latency_s": 0.0,
        "error": None,
    }

    max_retries = 3
    for attempt in range(max_retries):
        try:
            start_time = time.time()
            response = client.completion(
                model=adapter_name,
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=temperature,
            )
            latency = time.time() - start_time

            choices = response.get("choices", [])
            if choices:
                model_answer = choices[0].get("text", "").strip()
                result["model_answer"] = model_answer
                result["is_refusal"] = is_refusal(model_answer)
                result["latency_s"] = round(latency, 3)
                return result

            result["error"] = "No choices in response"
            return result

        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                time.sleep(wait)
            else:
                result["error"] = str(e)

    return result


def main():
    print("=" * 80)
    print("Evaluation Step: Run Inference")
    print("=" * 80)

    args = parse_arguments()

    print(f"\nConfiguration:")
    print(f"  vLLM URL:       {args.vllm_url}")
    print(f"  Adapter:        {args.adapter_name}")
    print(f"  Eval Data:      {args.eval_data_path}")
    print(f"  Max Tokens:     {args.max_tokens}")
    print(f"  Temperature:    {args.temperature}")
    print(f"  Max Concurrent: {args.max_concurrent}")

    os.makedirs(args.output_dir, exist_ok=True)

    client = VllmClient(args.vllm_url)

    # Health check
    print("\n" + "-" * 80)
    print("Checking vLLM health...")
    try:
        client.wait_until_healthy(timeout=120, poll_interval=5)
    except Exception as e:
        print(f"ERROR: vLLM is not healthy: {e}")
        _write_results(args.output_dir, [], "failed")
        sys.exit(1)

    # Verify adapter is available
    print("Checking adapter availability...")
    try:
        models = client.list_models()
        print(f"Available models: {models}")
        if args.adapter_name not in models:
            print(f"ERROR: Adapter '{args.adapter_name}' not found")
            _write_results(args.output_dir, [], "failed")
            sys.exit(1)
        print(f"Adapter '{args.adapter_name}' is available")
    except Exception as e:
        print(f"ERROR: Failed to list models: {e}")
        _write_results(args.output_dir, [], "failed")
        sys.exit(1)

    # Load eval data
    print("\n" + "-" * 80)
    print("Loading evaluation data...")
    samples = load_eval_data(args.eval_data_path)
    print(f"Loaded {len(samples)} evaluation samples")

    negative_count = sum(1 for s in samples if s.get("metadata", {}).get("question_type") == "negative")
    positive_count = len(samples) - negative_count
    print(f"  Positive samples: {positive_count}")
    print(f"  Negative samples: {negative_count}")

    # Run inference
    print("\n" + "-" * 80)
    print(f"Running inference with {args.max_concurrent} concurrent workers...")

    results = [None] * len(samples)
    completed = 0
    errors = 0

    with ThreadPoolExecutor(max_workers=args.max_concurrent) as executor:
        futures = {}
        for i, sample in enumerate(samples):
            future = executor.submit(
                run_single_inference, client, args.adapter_name, sample,
                i, args.max_tokens, args.temperature,
            )
            futures[future] = i

        for future in as_completed(futures):
            idx = futures[future]
            result = future.result()
            results[idx] = result
            completed += 1

            if result.get("error"):
                errors += 1

            if completed % 100 == 0 or completed == len(samples):
                print(f"  Progress: {completed}/{len(samples)} ({errors} errors)")

    # Compute summary metrics
    print("\n" + "-" * 80)
    print("Computing summary metrics...")

    valid_results = [r for r in results if r is not None]
    latencies = [r["latency_s"] for r in valid_results if r["latency_s"] > 0]

    negative_results = [r for r in valid_results if r["question_type"] == "negative"]
    negative_refusals = sum(1 for r in negative_results if r["is_refusal"])
    negative_refusal_rate = negative_refusals / len(negative_results) if negative_results else 0.0

    summary = {
        "adapter": args.adapter_name,
        "total_samples": len(samples),
        "completed": len(valid_results),
        "errors": errors,
        "negative_samples": len(negative_results),
        "negative_refusals": negative_refusals,
        "negative_refusal_rate": round(negative_refusal_rate, 4),
        "avg_latency_s": round(sum(latencies) / len(latencies), 3) if latencies else 0,
        "p95_latency_s": round(sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0, 3),
    }

    print(f"  Total samples:         {summary['total_samples']}")
    print(f"  Completed:             {summary['completed']}")
    print(f"  Errors:                {summary['errors']}")
    print(f"  Negative refusal rate: {summary['negative_refusal_rate']:.2%}")
    print(f"  Avg latency:           {summary['avg_latency_s']}s")
    print(f"  P95 latency:           {summary['p95_latency_s']}s")

    # Write results
    output = {
        "summary": summary,
        "results": valid_results,
    }

    _write_results(args.output_dir, output, "completed")

    print("\n" + "=" * 80)
    print(f"Inference completed for adapter '{args.adapter_name}'")
    print("=" * 80)
    sys.exit(0)


def _write_results(output_dir: str, output: any, status: str):
    """Write results and status to output files."""
    results_path = os.path.join(output_dir, "inference_results.json")
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)

    status_path = os.path.join(output_dir, "inference_status.txt")
    with open(status_path, "w") as f:
        f.write(status)


if __name__ == "__main__":
    main()
