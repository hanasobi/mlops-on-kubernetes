#!/usr/bin/env python3
"""
Performance Gate Step: Decision

Evaluates load test results against absolute floors and relative baselines
from the last MLflow run. Logs all results to MLflow for auditability.

Four checks must ALL pass:
1. P95 E2E Latency:  < max-p95-latency absolute, <= baseline * regression factor
2. P95 TTFT:         < max-p95-ttft absolute, <= baseline * regression factor
3. Throughput:       > min-throughput absolute, >= baseline * retention factor
4. Success Rate:     > min-success-rate absolute, >= baseline

On pass: Sets 'perf-passed' alias in MLflow
On fail: Sets 'perf-failed' alias in MLflow

Input Parameters:
    --load-test-results: Path to load_test_results.json
    --model-name: MLflow registered model name
    --candidate-source: MLflow source for candidate (default: "alias:eval-passed")
    --max-p95-latency: Absolute P95 latency ceiling (default: 8.0s)
    --max-p95-ttft: Absolute P95 TTFT ceiling (default: 2.0s)
    --min-throughput: Absolute min throughput (default: 2.0 req/s)
    --min-success-rate: Absolute min success rate (default: 0.99)
    --latency-regression-factor: Max latency regression vs baseline (default: 1.15)
    --throughput-regression-factor: Min throughput retention vs baseline (default: 0.85)
    --mlflow-experiment: MLflow experiment name
    --output-dir: Directory to save output artifacts

Output Artifacts:
    perf_decision.json: Decision details with all metrics
    perf_status.txt: "passed" or "failed"
"""

import argparse
import json
import os
import sys

sys.path.insert(0, "/scripts")
sys.path.insert(0, "/perf-scripts")

import mlflow
import pandas as pd
from utils.mlflow_helpers import MLflowHelper


def parse_arguments():
    parser = argparse.ArgumentParser(description="Performance gate decision")

    parser.add_argument("--load-test-results", required=True,
                        help="Path to load_test_results.json")
    parser.add_argument("--model-name", default="mistral-7b-lora",
                        help="MLflow registered model name")
    parser.add_argument("--candidate-source", default="alias:eval-passed",
                        help="MLflow source for candidate")
    parser.add_argument("--max-p95-latency", type=float, default=8.0,
                        help="Absolute P95 latency ceiling (seconds)")
    parser.add_argument("--max-p95-ttft", type=float, default=2.0,
                        help="Absolute P95 TTFT ceiling (seconds)")
    parser.add_argument("--min-throughput", type=float, default=2.0,
                        help="Absolute min throughput (req/s)")
    parser.add_argument("--min-success-rate", type=float, default=0.99,
                        help="Absolute min success rate")
    parser.add_argument("--latency-regression-factor", type=float, default=1.15,
                        help="Max latency regression multiplier vs baseline")
    parser.add_argument("--throughput-regression-factor", type=float, default=0.85,
                        help="Min throughput retention vs baseline")
    parser.add_argument("--mlflow-experiment", default="vllm-lora-performance",
                        help="MLflow experiment name")
    parser.add_argument("--output-dir", required=True,
                        help="Directory to save output artifacts")
    parser.add_argument("--eval-data-path", default=None,
                        help="Path to eval.jsonl for dataset tracking")
    parser.add_argument("--data-source", default=None,
                        help="S3 URI of eval data for lineage tracking")

    return parser.parse_args()


def get_baseline_metrics(experiment_name: str) -> dict:
    """Query MLflow for the most recent passed performance run.

    Returns dict with p95_latency_s, p95_ttft_s, throughput_rps,
    client_success_rate, run_id -- or None if no baseline exists.
    """
    try:
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            return None

        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            filter_string="tags.perf_decision = 'passed'",
            order_by=["start_time DESC"],
            max_results=1,
        )

        if runs.empty:
            return None

        run = runs.iloc[0]
        return {
            "p95_latency_s": run.get("metrics.p95_latency_s"),
            "p95_ttft_s": run.get("metrics.p95_ttft_s"),
            "throughput_rps": run.get("metrics.throughput_rps"),
            "client_success_rate": run.get("metrics.client_success_rate"),
            "run_id": run.get("run_id"),
        }
    except Exception as e:
        print(f"  Warning: Could not fetch baseline: {e}")
        return None


def main():
    print("=" * 80)
    print("Performance Gate Step: Decision")
    print("=" * 80)

    args = parse_arguments()

    print(f"\nConfiguration:")
    print(f"  Model Name:         {args.model_name}")
    print(f"  Candidate Source:   {args.candidate_source}")
    print(f"  Max P95 Latency:    {args.max_p95_latency}s")
    print(f"  Max P95 TTFT:       {args.max_p95_ttft}s")
    print(f"  Min Throughput:     {args.min_throughput} req/s")
    print(f"  Min Success Rate:   {args.min_success_rate:.0%}")
    print(f"  Latency Regression: {args.latency_regression_factor}x")
    print(f"  Throughput Retain:  {args.throughput_regression_factor}x")
    print(f"  MLflow Experiment:  {args.mlflow_experiment}")

    os.makedirs(args.output_dir, exist_ok=True)

    # Load load test results
    print("\n" + "-" * 80)
    print("Loading load test results...")

    with open(args.load_test_results, "r") as f:
        load_test_data = json.load(f)

    summary = load_test_data.get("summary", {})

    candidate_p95_latency = summary.get("p95_latency_s")
    candidate_p95_ttft = summary.get("p95_ttft_s")
    candidate_throughput = summary.get("throughput_rps")
    candidate_success_rate = summary.get("client_success_rate", 0.0)

    print(f"  Candidate Metrics:")
    print(f"    P95 Latency:      {candidate_p95_latency}s")
    print(f"    P95 TTFT:         {candidate_p95_ttft}s")
    print(f"    Throughput:       {candidate_throughput} req/s")
    print(f"    Success Rate:     {candidate_success_rate:.2%}")

    # Fetch baseline from MLflow
    print("\n" + "-" * 80)
    print("Fetching baseline from MLflow...")

    baseline = get_baseline_metrics(args.mlflow_experiment)
    has_baseline = baseline is not None
    mode = "comparative" if has_baseline else "absolute"

    if has_baseline:
        print(f"  Baseline found (run: {baseline['run_id']}):")
        print(f"    P95 Latency:      {baseline['p95_latency_s']}s")
        print(f"    P95 TTFT:         {baseline['p95_ttft_s']}s")
        print(f"    Throughput:       {baseline['throughput_rps']} req/s")
        print(f"    Success Rate:     {baseline['client_success_rate']}")
    else:
        print("  No baseline found (first run) -- using absolute thresholds only")

    print(f"  Mode: {mode}")

    # Run checks
    print("\n" + "-" * 80)
    print("Running performance checks...")

    checks = []

    # Check 1: P95 E2E Latency
    if candidate_p95_latency is not None:
        abs_ok = candidate_p95_latency < args.max_p95_latency
        if has_baseline and baseline["p95_latency_s"] is not None:
            threshold_rel = round(baseline["p95_latency_s"] * args.latency_regression_factor, 4)
            rel_ok = candidate_p95_latency <= threshold_rel
            check_passed = abs_ok and rel_ok
        else:
            threshold_rel = None
            check_passed = abs_ok
    else:
        check_passed = False
        threshold_rel = None

    checks.append({
        "name": "P95 E2E Latency",
        "passed": check_passed,
        "candidate": candidate_p95_latency,
        "threshold_absolute": args.max_p95_latency,
        "baseline": baseline["p95_latency_s"] if has_baseline else None,
        "threshold_relative": threshold_rel,
    })

    # Check 2: P95 TTFT
    if candidate_p95_ttft is not None:
        abs_ok = candidate_p95_ttft < args.max_p95_ttft
        if has_baseline and baseline["p95_ttft_s"] is not None:
            threshold_rel = round(baseline["p95_ttft_s"] * args.latency_regression_factor, 4)
            rel_ok = candidate_p95_ttft <= threshold_rel
            check_passed = abs_ok and rel_ok
        else:
            threshold_rel = None
            check_passed = abs_ok
    else:
        check_passed = False
        threshold_rel = None

    checks.append({
        "name": "P95 TTFT",
        "passed": check_passed,
        "candidate": candidate_p95_ttft,
        "threshold_absolute": args.max_p95_ttft,
        "baseline": baseline["p95_ttft_s"] if has_baseline else None,
        "threshold_relative": threshold_rel,
    })

    # Check 3: Throughput
    if candidate_throughput is not None:
        abs_ok = candidate_throughput > args.min_throughput
        if has_baseline and baseline["throughput_rps"] is not None:
            threshold_rel = round(baseline["throughput_rps"] * args.throughput_regression_factor, 4)
            rel_ok = candidate_throughput >= threshold_rel
            check_passed = abs_ok and rel_ok
        else:
            threshold_rel = None
            check_passed = abs_ok
    else:
        check_passed = False
        threshold_rel = None

    checks.append({
        "name": "Throughput",
        "passed": check_passed,
        "candidate": candidate_throughput,
        "threshold_absolute": args.min_throughput,
        "baseline": baseline["throughput_rps"] if has_baseline else None,
        "threshold_relative": threshold_rel,
    })

    # Check 4: Success Rate
    abs_ok = candidate_success_rate >= args.min_success_rate
    if has_baseline and baseline["client_success_rate"] is not None:
        rel_ok = candidate_success_rate >= baseline["client_success_rate"]
        check_passed = abs_ok and rel_ok
    else:
        check_passed = abs_ok

    checks.append({
        "name": "Success Rate",
        "passed": check_passed,
        "candidate": candidate_success_rate,
        "threshold_absolute": args.min_success_rate,
        "baseline": baseline["client_success_rate"] if has_baseline else None,
    })

    # Print check results
    all_passed = all(c["passed"] for c in checks)
    for check in checks:
        status = "PASS" if check["passed"] else "FAIL"
        candidate_val = check["candidate"]
        line = f"  [{status}] {check['name']}: candidate={candidate_val}"
        line += f", floor={check.get('threshold_absolute')}"
        if check.get("baseline") is not None:
            line += f", baseline={check['baseline']}"
        if check.get("threshold_relative") is not None:
            line += f", max={check['threshold_relative']}"
        print(line)

    decision = "passed" if all_passed else "failed"
    print(f"\n  Decision: {decision.upper()}")

    # Resolve candidate version from MLflow
    print("\n" + "-" * 80)
    print("Logging to MLflow...")

    mlflow_helper = MLflowHelper()

    candidate_version = None
    try:
        _, source_type = mlflow_helper.parse_source(args.model_name, args.candidate_source)
        if source_type == "alias":
            alias = args.candidate_source.split(":", 1)[1]
            version_info = mlflow_helper.get_model_version_by_alias(args.model_name, alias)
            if version_info:
                candidate_version = version_info["version"]
                print(f"  Candidate version: {candidate_version}")
    except Exception as e:
        print(f"  Warning: Could not get candidate version: {e}")

    # Create MLflow experiment run
    try:
        experiment = mlflow.get_experiment_by_name(args.mlflow_experiment)
        if experiment is None:
            experiment_id = mlflow.create_experiment(args.mlflow_experiment)
        else:
            experiment_id = experiment.experiment_id

        with mlflow.start_run(experiment_id=experiment_id,
                              run_name=f"perf-{args.model_name}"):
            # Params
            mlflow.log_param("model_name", args.model_name)
            mlflow.log_param("candidate_adapter", summary.get("adapter", ""))
            if candidate_version:
                mlflow.log_param("candidate_version", candidate_version)
            mlflow.log_param("concurrency", summary.get("concurrency", 0))
            mlflow.log_param("total_requests", summary.get("total_requests", 0))
            mlflow.log_param("max_p95_latency", args.max_p95_latency)
            mlflow.log_param("max_p95_ttft", args.max_p95_ttft)
            mlflow.log_param("min_throughput", args.min_throughput)
            mlflow.log_param("min_success_rate", args.min_success_rate)
            mlflow.log_param("latency_regression_factor", args.latency_regression_factor)
            mlflow.log_param("throughput_regression_factor", args.throughput_regression_factor)
            mlflow.log_param("mode", mode)
            if has_baseline:
                mlflow.log_param("baseline_run_id", baseline["run_id"])

            # Candidate metrics
            for key in ["p50_latency_s", "p95_latency_s", "p99_latency_s",
                         "p50_ttft_s", "p95_ttft_s", "p99_ttft_s",
                         "throughput_rps", "kv_cache_usage", "gpu_utilization"]:
                value = summary.get(key)
                if value is not None:
                    mlflow.log_metric(key, value)

            mlflow.log_metric("client_success_rate", candidate_success_rate)
            mlflow.log_metric("wall_clock_duration_s",
                              summary.get("wall_clock_duration_s", 0))

            # Baseline metrics
            if has_baseline:
                for key in ["p95_latency_s", "p95_ttft_s", "throughput_rps",
                             "client_success_rate"]:
                    value = baseline.get(key)
                    if value is not None:
                        mlflow.log_metric(f"baseline_{key}", value)

            # Decision tag
            mlflow.set_tag("perf_decision", decision)

            # Artifacts
            mlflow.log_artifact(args.load_test_results)

            # Dataset lineage
            if args.eval_data_path and os.path.exists(args.eval_data_path):
                try:
                    eval_df = pd.read_json(args.eval_data_path, lines=True)
                    eval_ds = mlflow.data.from_pandas(
                        eval_df,
                        source=args.data_source or args.eval_data_path,
                        name=f"eval-v{args.data_source.split('/')[-2] if args.data_source else 'unknown'}"
                    )
                    mlflow.log_input(eval_ds, context="performance-test")
                    print(f"  Dataset lineage logged (eval: {len(eval_df)} samples)")
                except Exception as e:
                    print(f"  Warning: Dataset logging failed: {e}")

        print("  MLflow run logged successfully")

    except Exception as e:
        print(f"  Warning: MLflow logging failed: {e}")
        print("  Continuing with alias update...")

    # Set MLflow alias
    print("\n" + "-" * 80)
    if candidate_version:
        alias = "perf-passed" if all_passed else "perf-failed"
        opposite_alias = "perf-failed" if all_passed else "perf-passed"
        try:
            mlflow_helper.delete_alias(args.model_name, opposite_alias)
            print(f"Removed stale alias '{opposite_alias}' (if existed)")
        except Exception:
            pass
        try:
            mlflow_helper.set_alias(args.model_name, alias, candidate_version)
            print(f"Set alias '{alias}' on version {candidate_version}")
        except Exception as e:
            print(f"Warning: Failed to set alias: {e}")
    else:
        print("Warning: No candidate version found, skipping alias update")

    # Write output
    output = {
        "decision": decision,
        "mode": mode,
        "candidate_metrics": {
            "p95_latency_s": candidate_p95_latency,
            "p95_ttft_s": candidate_p95_ttft,
            "throughput_rps": candidate_throughput,
            "success_rate": candidate_success_rate,
        },
        "baseline_metrics": {
            "p95_latency_s": baseline["p95_latency_s"],
            "p95_ttft_s": baseline["p95_ttft_s"],
            "throughput_rps": baseline["throughput_rps"],
            "success_rate": baseline["client_success_rate"],
            "run_id": baseline["run_id"],
        } if has_baseline else None,
        "checks": checks,
        "candidate_version": candidate_version,
    }

    _write_results(args.output_dir, output, decision)

    print("\n" + "=" * 80)
    if all_passed:
        print("Performance gate PASSED - adapter approved")
    else:
        print("Performance gate FAILED - adapter rejected")
    print("=" * 80)

    sys.exit(0 if all_passed else 1)


def _json_default(obj):
    """Handle numpy types that aren't JSON serializable."""
    if hasattr(obj, "item"):
        return obj.item()
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


def _write_results(output_dir: str, output: any, status: str):
    """Write results and status to output files."""
    results_path = os.path.join(output_dir, "perf_decision.json")
    with open(results_path, "w") as f:
        json.dump(output, f, indent=2, default=_json_default)

    status_path = os.path.join(output_dir, "perf_status.txt")
    with open(status_path, "w") as f:
        f.write(status)


if __name__ == "__main__":
    main()
