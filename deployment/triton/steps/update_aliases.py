#!/usr/bin/env python3
"""
Step 6: Update MLflow Deployment Aliases

Synchronizes the MLflow Model Registry aliases with the current
deployment state. This step implements our alias strategy:

- The current "live" alias is moved to "previous" (rollback target)
- The newly deployed version gets the "live" alias (production marker)

This gives us clear tracking of which model is currently running in
production and enables fast rollbacks to the previous version.

IMPORTANT: This step is critical for the rollback workflow. If the
alias updates fail, we cannot safely rollback later. Therefore any
error leads to a workflow failure (sys.exit(1)).

Input Parameters:
    --model-name: Name of the model in MLflow Registry
    --mlflow-version: The MLflow version that was just deployed
    --mlflow-tracking-uri: MLflow server URL
    --dry-run: If set, only shows what would happen without changing anything
    --output-dir: Where output artifacts should be saved

Output Artifacts:
    alias_updates.json: Documentation of which aliases were changed and how

Environment Variables:
    MLFLOW_TRACKING_URI: Optional - overridden by --mlflow-tracking-uri
"""

import argparse
import json
import os
import sys

from utils.mlflow_helpers import MLflowHelper


def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Update MLflow deployment aliases after successful deployment'
    )

    parser.add_argument(
        '--model-name',
        required=True,
        help='Name of the registered model in MLflow'
    )

    parser.add_argument(
        '--mlflow-version',
        required=True,
        help='MLflow version number that was just deployed'
    )

    parser.add_argument(
        '--mlflow-tracking-uri',
        required=True,
        help='MLflow tracking server URL'
    )

    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be done without actually changing aliases'
    )

    parser.add_argument(
        '--output-dir',
        required=True,
        help='Directory to save output artifacts'
    )

    return parser.parse_args()


def document_current_state(mlflow_helper, model_name):
    """
    Documents the current alias state before changes.

    This is important for audit trails and for debugging. If something
    goes wrong later, we want to be able to trace what the state was
    before we changed the aliases. This documentation ends up in the
    workflow logs and in the output artifact.

    Args:
        mlflow_helper: Initialized MLflowHelper
        model_name: Name of the model

    Returns:
        Dictionary with the current alias state
    """
    print("Documenting current alias state...")

    current_state = {}

    # Check the relevant aliases
    for alias in ['live', 'previous', 'deploy', 'champion']:
        version_info = mlflow_helper.get_model_version_by_alias(model_name, alias)

        if version_info:
            current_state[alias] = version_info['version']
            print(f"  {alias}: version {version_info['version']}")
        else:
            current_state[alias] = None
            print(f"  {alias}: not set")

    return current_state


def main():
    """Main logic of the update_aliases step."""
    print("=" * 80)
    print("Step 6: Update MLflow Deployment Aliases")
    print("=" * 80)

    # Step 1: Parse arguments
    args = parse_arguments()

    print(f"\nConfiguration:")
    print(f"  Model Name: {args.model_name}")
    print(f"  MLflow Version: {args.mlflow_version}")
    print(f"  MLflow URI: {args.mlflow_tracking_uri}")
    print(f"  Dry Run: {args.dry_run}")
    print(f"  Output Directory: {args.output_dir}")

    if args.dry_run:
        print("\n\u26a0\ufe0f  DRY RUN MODE - No changes will be made")

    # Step 2: Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Step 3: Initialize MLflow helper
    print("\n" + "-" * 80)
    print("Initializing MLflow Helper...")

    try:
        mlflow_helper = MLflowHelper(tracking_uri=args.mlflow_tracking_uri)
    except Exception as e:
        print(f"\nERROR: Failed to initialize MLflow helper")
        print(f"Details: {e}")
        print("\nPossible causes:")
        print("  - MLflow server is not reachable")
        print("  - Invalid tracking URI")
        print("  - Network connectivity issues")
        sys.exit(1)

    # Step 4: Document current state
    print("\n" + "-" * 80)

    try:
        state_before = document_current_state(mlflow_helper, args.model_name)
    except Exception as e:
        print(f"\nERROR: Failed to get current alias state")
        print(f"Details: {e}")
        print("\nPossible causes:")
        print("  - Model does not exist in MLflow Registry")
        print("  - MLflow server connection issues")
        print("  - Insufficient permissions to read model metadata")
        sys.exit(1)

    # Step 5: Update aliases
    print("\n" + "-" * 80)
    print("Updating deployment aliases...")

    try:
        # The update_deployment_aliases method has the complete logic
        # It handles: live -> previous, new -> live
        alias_updates = mlflow_helper.update_deployment_aliases(
            model_name=args.model_name,
            new_version=args.mlflow_version,
            dry_run=args.dry_run
        )

        if args.dry_run:
            print("\nDry run completed - no changes were made")
        else:
            print("\nAliases updated successfully")

    except Exception as e:
        print(f"\nERROR: Failed to update deployment aliases")
        print(f"Details: {e}")
        print("\nThis is a critical error because:")
        print("  - The deployment to Triton was successful")
        print("  - But MLflow Registry is now out of sync")
        print("  - Rollback workflow depends on correct aliases")
        print("\nYou need to manually investigate and fix the aliases")
        print("or re-run this step once the underlying issue is resolved.")

        # Consistently critical - alias updates are essential
        # for the rollback workflow and must succeed
        sys.exit(1)

    # Step 6: Document new state
    print("\n" + "-" * 80)
    print("Documenting new alias state...")

    try:
        state_after = document_current_state(mlflow_helper, args.model_name)
    except Exception as e:
        print(f"\nERROR: Failed to document final state")
        print(f"Details: {e}")
        print("\nAlias updates were performed but we cannot verify the result")
        sys.exit(1)

    # Step 7: Save changes as artifact
    update_summary = {
        'model_name': args.model_name,
        'deployed_version': args.mlflow_version,
        'dry_run': args.dry_run,
        'state_before': state_before,
        'state_after': state_after,
        'changes': alias_updates
    }

    output_path = os.path.join(args.output_dir, "alias_updates.json")

    try:
        print(f"\nWriting alias update summary to {output_path}")
        with open(output_path, 'w') as f:
            json.dump(update_summary, f, indent=2)

        print("Update summary written successfully")

    except Exception as e:
        print(f"\nERROR: Failed to write update summary artifact")
        print(f"Details: {e}")
        print("\nThis prevents proper audit trail documentation")
        sys.exit(1)

    # Step 8: Print summary
    print("\n" + "=" * 80)
    print("Step 6 completed successfully!")
    print("=" * 80)

    if not args.dry_run:
        print(f"\nAlias Updates:")

        if 'live' in alias_updates:
            print(f"  live: \u2192 version {alias_updates['live']}")

        if 'previous' in alias_updates:
            print(f"  previous: \u2192 version {alias_updates['previous']}")
        else:
            print(f"  previous: not updated (first deployment)")

        print(f"\nProduction State:")
        print(f"  Current live version: {alias_updates.get('live', 'unknown')}")
        print(f"  Rollback target: {alias_updates.get('previous', 'none')}")

        print(f"\nMLflow Registry is now synchronized with Triton deployment")
    else:
        print(f"\nDry run completed - no changes were made")
        print(f"To actually update aliases, run without --dry-run flag")

    # Success Exit Code
    sys.exit(0)


if __name__ == '__main__':
    main()
