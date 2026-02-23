#!/usr/bin/env python3
"""
Step 4: Set Staged Alias

Sets the 'staged' alias on the model version in MLflow after the
smoke test has passed. This signals that the adapter is deployed
to the candidate slot in vLLM and ready for evaluation.

Alias lifecycle:
    candidate  ->  candidate + staged  ->  live  ->  previous

Input Parameters:
    --model-name: MLflow registered model name (e.g. "mistral-7b-lora")
    --source: MLflow source used in staging (e.g. "alias:candidate" or "version:3")
    --output-dir: Directory to save output artifacts

Output Artifacts:
    staged_alias_result.txt: "success" or "failed"

Environment Variables:
    MLFLOW_TRACKING_URI: MLflow server URL
"""

import argparse
import json
import os
import sys

sys.path.insert(0, "/scripts")

from utils.mlflow_helpers import MLflowHelper


def parse_arguments():
    parser = argparse.ArgumentParser(description="Set staged alias in MLflow")

    parser.add_argument("--model-name", required=True, help="MLflow registered model name")
    parser.add_argument("--source", default="alias:candidate", help="MLflow source (alias:X or version:N)")
    parser.add_argument("--output-dir", required=True, help="Directory to save output artifacts")

    return parser.parse_args()


def validate_environment():
    required_vars = ["MLFLOW_TRACKING_URI"]
    missing = [v for v in required_vars if v not in os.environ]
    if missing:
        print(f"ERROR: Missing required environment variables: {missing}")
        sys.exit(1)


def main():
    print("=" * 80)
    print("Step 4: Set Staged Alias")
    print("=" * 80)

    args = parse_arguments()
    validate_environment()

    print(f"\nConfiguration:")
    print(f"  Model Name: {args.model_name}")
    print(f"  Source:      {args.source}")

    os.makedirs(args.output_dir, exist_ok=True)

    mlflow_helper = MLflowHelper()

    # Resolve the model version from the source
    print("\n" + "-" * 80)
    print("Resolving model version...")

    try:
        _, source_type = mlflow_helper.parse_source(args.model_name, args.source)

        if source_type == "alias":
            alias = args.source.split(":", 1)[1]
            version_info = mlflow_helper.get_model_version_by_alias(args.model_name, alias)
            if not version_info:
                raise Exception(f"No model version found for alias '{alias}'")
            version = version_info["version"]
        else:
            version = args.source.split(":", 1)[1]

        print(f"Model version: {version}")

    except Exception as e:
        print(f"\nERROR: Could not resolve model version: {e}")
        _write_result(args.output_dir, "failed")
        sys.exit(1)

    # Set the staged alias
    print("\n" + "-" * 80)
    print(f"Setting 'staged' alias on version {version}...")

    try:
        mlflow_helper.set_alias(args.model_name, "staged", version)
        print(f"Alias 'staged' set to version {version}")
    except Exception as e:
        print(f"\nERROR: Failed to set staged alias: {e}")
        _write_result(args.output_dir, "failed")
        sys.exit(1)

    _write_result(args.output_dir, "success")

    # Summary
    print("\n" + "=" * 80)
    print("Step 4 completed successfully!")
    print("=" * 80)
    print(f"\n{args.model_name} version {version} now has aliases: candidate + staged")

    sys.exit(0)


def _write_result(output_dir: str, result: str):
    path = os.path.join(output_dir, "staged_alias_result.txt")
    with open(path, "w") as f:
        f.write(result)


if __name__ == "__main__":
    main()
