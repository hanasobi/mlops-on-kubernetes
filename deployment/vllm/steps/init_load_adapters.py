#!/usr/bin/env python3
"""
Init Container: Load adapters from MLflow on cold start.

Queries MLflow for 'live' and 'staged' aliases and downloads
the corresponding adapter files to /mnt/adapters/. This runs
as an init-container before vLLM starts, so the adapters are
available at startup via --lora-modules.

Alias-to-slot mapping:
    live   → /mnt/adapters/aws-rag-qa-live/
    staged → /mnt/adapters/aws-rag-qa-candidate/

If an alias does not exist, the corresponding slot is skipped.
This handles first-time deployments gracefully.

Environment Variables:
    MLFLOW_TRACKING_URI: MLflow server URL
    MODEL_NAME: MLflow registered model name (default: mistral-7b-lora)
"""

import os
import sys

sys.path.insert(0, "/scripts")

from utils.mlflow_helpers import MLflowHelper

# Maps MLflow alias → adapter slot directory name
ALIAS_SLOT_MAPPING = {
    "live": "aws-rag-qa-live",
    "staged": "aws-rag-qa-candidate",
}

ADAPTERS_DIR = "/mnt/adapters"


def main():
    print("=" * 80)
    print("Init Container: Load Adapters from MLflow")
    print("=" * 80)

    model_name = os.environ.get("MODEL_NAME", "mistral-7b-lora")
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")

    if not tracking_uri:
        print("WARNING: MLFLOW_TRACKING_URI not set. Skipping adapter loading.")
        print("vLLM will start without LoRA adapters.")
        sys.exit(0)

    print(f"\nConfiguration:")
    print(f"  MLflow URI:  {tracking_uri}")
    print(f"  Model Name:  {model_name}")
    print(f"  Adapters Dir: {ADAPTERS_DIR}")

    mlflow_helper = MLflowHelper(tracking_uri)
    loaded = 0

    for alias, slot_name in ALIAS_SLOT_MAPPING.items():
        print(f"\n" + "-" * 80)
        print(f"Checking alias '{alias}' → slot '{slot_name}'...")

        version_info = mlflow_helper.get_model_version_by_alias(model_name, alias)

        if not version_info:
            print(f"  No '{alias}' alias found. Skipping.")
            continue

        version = version_info["version"]
        print(f"  Found version {version}")

        output_dir = os.path.join(ADAPTERS_DIR, slot_name)

        try:
            metadata = mlflow_helper.download_adapter(
                model_name, f"version:{version}", output_dir
            )
            print(f"  Adapter downloaded to {output_dir}")
            print(f"  Files: {metadata.get('adapter_files', [])}")
            loaded += 1

        except Exception as e:
            print(f"  WARNING: Failed to download adapter for '{alias}': {e}")
            print(f"  Continuing without this adapter.")

    # Summary
    print(f"\n" + "=" * 80)
    print(f"Init complete: {loaded}/{len(ALIAS_SLOT_MAPPING)} adapters loaded")

    if loaded == 0:
        print("vLLM will start without LoRA adapters.")
    print("=" * 80)

    sys.exit(0)


if __name__ == "__main__":
    main()
