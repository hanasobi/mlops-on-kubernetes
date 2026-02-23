#!/usr/bin/env python3
"""
Evaluation Step: Run Judge

Takes inference results from candidate (and optionally live) adapter,
sends each answer to the Judge LLM (Llama 3.1 70B Instruct) for
A/B/C grading.

Uses /v1/chat/completions (chat model endpoint) on the Judge vLLM instance.

Input Parameters:
    --judge-url: Judge vLLM server URL
    --judge-model: Judge model name (empty = use default model)
    --candidate-results: Path to candidate inference_results.json
    --live-results: Path to live inference_results.json (optional)
    --max-concurrent: Maximum concurrent requests (default: 15)
    --max-tokens: Maximum tokens for judge response (default: 128)
    --output-dir: Directory to save output artifacts

Output Artifacts:
    judge_results.json: Per-sample grades for both adapters
    judge_status.txt: "completed" or "failed"
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
from eval_utils.judge_prompt import build_judge_prompt, parse_judge_response


def parse_arguments():
    parser = argparse.ArgumentParser(description="Judge inference results with LLM")

    parser.add_argument("--judge-url", required=True, help="Judge vLLM server URL")
    parser.add_argument("--judge-model", default="", help="Judge model name (empty = default)")
    parser.add_argument("--candidate-results", required=True, help="Path to candidate inference_results.json")
    parser.add_argument("--live-results", default="", help="Path to live inference_results.json (optional)")
    parser.add_argument("--max-concurrent", type=int, default=15, help="Max concurrent requests")
    parser.add_argument("--max-tokens", type=int, default=128, help="Max tokens for judge response")
    parser.add_argument("--output-dir", required=True, help="Directory to save output artifacts")

    return parser.parse_args()


def judge_single_answer(client: VllmClient, model: str, result: dict,
                        max_tokens: int) -> dict:
    """Send a single answer to the judge for grading."""
    messages = build_judge_prompt(
        context=result.get("context", ""),
        question=result.get("question", ""),
        answer=result.get("model_answer", ""),
        question_type=result.get("question_type", "factual"),
    )

    grade = {
        "index": result.get("index", -1),
        "question_type": result.get("question_type", "factual"),
        "rating": "ERROR",
        "hallucination": None,
        "reasoning": "",
        "raw_response": "",
    }

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = client.chat_completion(
                model=model,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.0,
            )

            choices = response.get("choices", [])
            if choices:
                raw_text = choices[0].get("message", {}).get("content", "")
                grade["raw_response"] = raw_text

                parsed = parse_judge_response(raw_text)
                grade["rating"] = parsed["rating"]
                grade["hallucination"] = parsed["hallucination"]
                grade["reasoning"] = parsed["reasoning"]
                return grade

            grade["reasoning"] = "No choices in judge response"
            return grade

        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** attempt
                time.sleep(wait)
            else:
                grade["reasoning"] = f"Judge request failed: {e}"

    # Parse failures treated as C (conservative)
    if grade["rating"] == "ERROR":
        grade["rating"] = "C"
        grade["reasoning"] = f"Treated as C after failure: {grade['reasoning']}"

    return grade


def grade_results(client: VllmClient, model: str, results: list,
                  max_concurrent: int, max_tokens: int, label: str) -> list:
    """Grade a list of inference results using the judge."""
    print(f"\nGrading {len(results)} {label} results...")

    grades = [None] * len(results)
    completed = 0

    with ThreadPoolExecutor(max_workers=max_concurrent) as executor:
        futures = {}
        for i, result in enumerate(results):
            future = executor.submit(
                judge_single_answer, client, model, result, max_tokens,
            )
            futures[future] = i

        for future in as_completed(futures):
            idx = futures[future]
            grades[idx] = future.result()
            completed += 1

            if completed <= 3:
                g = grades[idx]
                print(f"  [{label}] Sample {g['index']}: rating={g['rating']}, raw={g['raw_response'][:120]}")

            if completed % 100 == 0 or completed == len(results):
                print(f"  [{label}] Progress: {completed}/{len(results)}")

    return [g for g in grades if g is not None]


def compute_grade_summary(grades: list) -> dict:
    """Compute summary statistics from grades."""
    total = len(grades)
    if total == 0:
        return {"total": 0, "a_count": 0, "b_count": 0, "c_count": 0,
                "a_rate": 0.0, "b_rate": 0.0, "c_rate": 0.0, "error_count": 0}

    a_count = sum(1 for g in grades if g["rating"] == "A")
    b_count = sum(1 for g in grades if g["rating"] == "B")
    c_count = sum(1 for g in grades if g["rating"] == "C")
    error_count = sum(1 for g in grades if g["rating"] == "ERROR")

    return {
        "total": total,
        "a_count": a_count,
        "b_count": b_count,
        "c_count": c_count,
        "error_count": error_count,
        "a_rate": round(a_count / total, 4),
        "b_rate": round(b_count / total, 4),
        "c_rate": round(c_count / total, 4),
    }


def main():
    print("=" * 80)
    print("Evaluation Step: Run Judge")
    print("=" * 80)

    args = parse_arguments()

    print(f"\nConfiguration:")
    print(f"  Judge URL:      {args.judge_url}")
    print(f"  Judge Model:    {args.judge_model or '(default)'}")
    print(f"  Candidate:      {args.candidate_results}")
    print(f"  Live:           {args.live_results or '(none)'}")
    print(f"  Max Concurrent: {args.max_concurrent}")
    print(f"  Max Tokens:     {args.max_tokens}")

    os.makedirs(args.output_dir, exist_ok=True)

    client = VllmClient(args.judge_url)

    # Health check
    print("\n" + "-" * 80)
    print("Checking Judge vLLM health...")
    try:
        client.wait_until_healthy(timeout=600, poll_interval=10)
    except Exception as e:
        print(f"ERROR: Judge vLLM is not healthy: {e}")
        _write_results(args.output_dir, {}, "failed")
        sys.exit(1)

    # Determine judge model name
    judge_model = args.judge_model
    if not judge_model:
        models = client.list_models()
        if models:
            judge_model = models[0]
            print(f"Using default judge model: {judge_model}")
        else:
            print("ERROR: No models available on judge vLLM")
            _write_results(args.output_dir, {}, "failed")
            sys.exit(1)

    # Load candidate results
    print("\n" + "-" * 80)
    print("Loading inference results...")

    with open(args.candidate_results, "r") as f:
        candidate_data = json.load(f)
    candidate_results = candidate_data.get("results", [])
    print(f"  Candidate results: {len(candidate_results)}")

    # Load live results (optional)
    live_results = []
    if args.live_results and os.path.exists(args.live_results):
        with open(args.live_results, "r") as f:
            live_data = json.load(f)
        live_results = live_data.get("results", [])
        print(f"  Live results:      {len(live_results)}")
    else:
        print("  Live results:      (none)")

    # Grade candidate
    print("\n" + "-" * 80)
    candidate_grades = grade_results(
        client, judge_model, candidate_results,
        args.max_concurrent, args.max_tokens, "candidate",
    )
    candidate_summary = compute_grade_summary(candidate_grades)

    print(f"\n  Candidate Summary:")
    print(f"    A-Rate: {candidate_summary['a_rate']:.2%} ({candidate_summary['a_count']}/{candidate_summary['total']})")
    print(f"    B-Rate: {candidate_summary['b_rate']:.2%} ({candidate_summary['b_count']}/{candidate_summary['total']})")
    print(f"    C-Rate: {candidate_summary['c_rate']:.2%} ({candidate_summary['c_count']}/{candidate_summary['total']})")

    # Grade live (if available)
    live_grades = []
    live_summary = None
    if live_results:
        live_grades = grade_results(
            client, judge_model, live_results,
            args.max_concurrent, args.max_tokens, "live",
        )
        live_summary = compute_grade_summary(live_grades)

        print(f"\n  Live Summary:")
        print(f"    A-Rate: {live_summary['a_rate']:.2%} ({live_summary['a_count']}/{live_summary['total']})")
        print(f"    B-Rate: {live_summary['b_rate']:.2%} ({live_summary['b_count']}/{live_summary['total']})")
        print(f"    C-Rate: {live_summary['c_rate']:.2%} ({live_summary['c_count']}/{live_summary['total']})")

    # Write output
    output = {
        "judge_model": judge_model,
        "candidate_grades": candidate_grades,
        "candidate_summary": candidate_summary,
        "live_grades": live_grades,
        "live_summary": live_summary,
    }

    _write_results(args.output_dir, output, "completed")

    print("\n" + "=" * 80)
    print("Judge evaluation completed")
    print("=" * 80)
    sys.exit(0)


def _write_results(output_dir: str, output: any, status: str):
    """Write results and status to output files."""
    results_path = os.path.join(output_dir, "judge_results.json")
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2)

    status_path = os.path.join(output_dir, "judge_status.txt")
    with open(status_path, "w") as f:
        f.write(status)


if __name__ == "__main__":
    main()
