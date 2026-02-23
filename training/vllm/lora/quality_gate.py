"""
Quality Gate for LoRA Adapters

Checks training metrics against thresholds and registers the adapter
in the MLflow Model Registry with the 'candidate' alias if passed.

Usage:
    python quality_gate.py --run-id <mlflow-run-id>
"""

import argparse
import sys
import mlflow
from mlflow.tracking import MlflowClient

from config import DEFAULT_TRAINING_CONFIG


def check_quality_gate(run_id: str, config) -> dict:
    """
    Check if training metrics meet quality thresholds.

    Returns dict with 'passed' bool and details.
    """
    client = MlflowClient()
    run = client.get_run(run_id)
    metrics = run.data.metrics

    eval_loss = metrics.get("final_eval_loss")
    perplexity = metrics.get("final_perplexity")

    if eval_loss is None or perplexity is None:
        return {
            "passed": False,
            "reason": f"Missing metrics: eval_loss={eval_loss}, perplexity={perplexity}",
        }

    checks = []
    passed = True

    # Check eval loss
    if eval_loss <= config.max_eval_loss:
        checks.append(f"eval_loss: {eval_loss:.4f} <= {config.max_eval_loss} (PASS)")
    else:
        checks.append(f"eval_loss: {eval_loss:.4f} > {config.max_eval_loss} (FAIL)")
        passed = False

    # Check perplexity
    if perplexity <= config.max_perplexity:
        checks.append(f"perplexity: {perplexity:.2f} <= {config.max_perplexity} (PASS)")
    else:
        checks.append(f"perplexity: {perplexity:.2f} > {config.max_perplexity} (FAIL)")
        passed = False

    return {"passed": passed, "checks": checks, "eval_loss": eval_loss, "perplexity": perplexity}


def register_adapter(run_id: str, model_name: str):
    """
    Register the adapter from this run in the MLflow Model Registry
    and assign the 'candidate' alias.
    """
    client = MlflowClient()

    # Register model from the run's adapter artifacts
    model_uri = f"runs:/{run_id}/adapter"
    result = mlflow.register_model(model_uri, model_name)
    version = result.version

    print(f"Registered model: {model_name} version {version}")

    # Set 'candidate' alias
    client.set_registered_model_alias(model_name, "candidate", version)
    print(f"Set alias 'candidate' -> version {version}")

    return version


def main():
    parser = argparse.ArgumentParser(description="Quality Gate for LoRA adapters")
    parser.add_argument(
        "--run-id",
        type=str,
        required=True,
        help="MLflow run ID from training step",
    )
    parser.add_argument("--max-eval-loss", type=float, default=None, help="Override max eval loss threshold")
    parser.add_argument("--max-perplexity", type=float, default=None, help="Override max perplexity threshold")
    args = parser.parse_args()

    config = DEFAULT_TRAINING_CONFIG
    if args.max_eval_loss is not None:
        config.max_eval_loss = args.max_eval_loss
    if args.max_perplexity is not None:
        config.max_perplexity = args.max_perplexity

    print("=" * 80)
    print("Quality Gate: LoRA Adapter")
    print("=" * 80)
    print(f"Run ID:         {args.run_id}")
    print(f"Max eval loss:  {config.max_eval_loss}")
    print(f"Max perplexity: {config.max_perplexity}")
    print("=" * 80)

    # Check metrics
    result = check_quality_gate(args.run_id, config)

    print("\nChecks:")
    for check in result.get("checks", []):
        print(f"  {check}")

    if not result["passed"]:
        reason = result.get("reason", "Threshold(s) not met")
        print(f"\nRESULT=failed")
        print(f"Quality gate FAILED: {reason}")
        # Write result for Argo output parameter
        with open("/tmp/quality_gate_result", "w") as f:
            f.write("failed")
        sys.exit(0)  # Don't fail the step, let Argo decide based on output

    print(f"\nQuality gate PASSED")

    # Register adapter in MLflow Model Registry
    print(f"\nRegistering adapter in MLflow Model Registry...")
    version = register_adapter(args.run_id, config.mlflow_model_name)

    print(f"\nRESULT=passed")

    # Write result for Argo output parameter
    with open("/tmp/quality_gate_result", "w") as f:
        f.write("passed")


if __name__ == "__main__":
    main()
