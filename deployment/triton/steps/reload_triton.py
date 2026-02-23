#!/usr/bin/env python3
"""
Step 5: Reload Triton Model

Triggers Triton Inference Server to load the new model from the S3 repository.
This step is the critical moment where the model is actually activated and
made available for inference.

The process is:
1. Repository index refresh - Triton sees new version in S3
2. Model load - Triton starts the load process
3. Wait until ready - We wait until the model is fully loaded
4. Verification - We check that the correct version was loaded

Input Parameters:
    --model-name: Name of the model in Triton
    --triton-url: Base URL of the Triton server
    --expected-version: Optional - which version we expect to load
    --load-timeout: Maximum wait time until model is ready (seconds)
    --output-dir: Where output artifacts should be saved

Output Artifacts:
    loaded_version.txt: Which version was actually loaded

Environment Variables:
    None required - Triton URL is passed as a parameter
"""

import argparse
import os
import sys
import time

from utils.triton_client import TritonClient


def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Reload model in Triton Inference Server'
    )

    parser.add_argument(
        '--model-name',
        required=True,
        help='Name of the model in Triton (must match S3 repository)'
    )

    parser.add_argument(
        '--triton-url',
        default='http://triton-service.ai-platform:8000',
        help='Base URL of Triton server (default: Kubernetes service)'
    )

    parser.add_argument(
        '--expected-version',
        type=int,
        default=None,
        help='Expected version to be loaded (for verification)'
    )

    parser.add_argument(
        '--load-timeout',
        type=int,
        default=60,
        help='Maximum time to wait for model to become ready (seconds)'
    )

    parser.add_argument(
        '--output-dir',
        required=True,
        help='Directory to save output artifacts'
    )

    return parser.parse_args()


def check_triton_health(triton_client):
    """
    Checks whether the Triton server is reachable at all.

    Before we try to load a model, we should make sure that Triton
    is running and responding to requests. This is an early sanity
    check that saves us time if Triton is unavailable for any reason.

    Args:
        triton_client: Initialized TritonClient

    Raises:
        Exception if Triton is not healthy
    """
    print("Checking Triton server health...")

    try:
        if triton_client.is_server_ready():
            print("  \u2713 Triton server is healthy and ready")
        else:
            raise Exception("Triton server is not ready")

    except Exception as e:
        print(f"\nERROR: Cannot connect to Triton server")
        print(f"Details: {e}")
        print("\nPossible causes:")
        print("  - Triton server is not running")
        print("  - Network connectivity issues")
        print("  - Incorrect Triton URL")
        print("  - Triton is starting up (try again in a moment)")
        raise


def get_currently_loaded_versions(triton_client, model_name):
    """
    Determines which versions are currently in Triton memory.

    This is important for two things: First, we want to know which
    versions are already loaded before we start, so we can later see
    what changed. Second, we need this for debugging if something
    goes wrong - we want to be able to show the operator what the
    state was before and after the load.

    Args:
        triton_client: Initialized TritonClient
        model_name: Name of the model

    Returns:
        List of integer version numbers
    """
    print(f"\nChecking currently loaded versions of '{model_name}'...")

    try:
        versions = triton_client.get_loaded_versions(model_name)

        if versions:
            print(f"  Currently loaded: {versions}")
        else:
            print(f"  No versions currently loaded (first deployment)")

        return versions

    except Exception as e:
        print(f"  Warning: Could not get loaded versions: {e}")
        print(f"  This may be normal if model has never been loaded before")
        return []


def refresh_repository(triton_client):
    """
    Triggers Triton to rescan the S3 repository.

    This is a critical step that is often overlooked. Triton caches
    its view of the repository for performance reasons. If we just
    uploaded a new version to S3, Triton does not know about it yet.
    The repository index refresh explicitly tells Triton: "Look again
    at what is in the S3 bucket, something may have changed".

    Without this refresh, Triton would not see the new version during
    the subsequent load and would instead load the old one, which
    would lead to confusion.

    Args:
        triton_client: Initialized TritonClient

    Raises:
        Exception if refresh fails
    """
    print("\nRefreshing Triton repository index...")
    print("This ensures Triton sees the newly uploaded version in S3")

    try:
        triton_client.refresh_repository_index()
        print("  \u2713 Repository index refreshed successfully")

        # Short pause to give Triton time to finish the scan
        # This is not strictly necessary but helps with timing issues
        time.sleep(2)

    except Exception as e:
        print(f"\nERROR: Failed to refresh repository index")
        print(f"Details: {e}")
        print("\nThis is a critical error - cannot proceed without refresh")
        raise


def load_model(triton_client, model_name):
    """
    Triggers Triton to load the model.

    We call load WITHOUT a version parameter, which tells Triton: "Load the
    latest available version from the repository". This is exactly what we
    want - we just uploaded a new version and after the repository refresh
    it is the latest, so it will be loaded.

    The version_policy in the config.pbtxt (num_versions: 2) automatically
    ensures that old versions are unloaded if necessary.

    Args:
        triton_client: Initialized TritonClient
        model_name: Name of the model

    Raises:
        Exception if load fails
    """
    print(f"\nLoading model '{model_name}'...")
    print("Loading latest version from repository (no version specified)")
    print("Triton will automatically manage version_policy (keep latest 2)")

    try:
        triton_client.load_model(model_name)
        print("  \u2713 Load request sent successfully")
        print("\nNote: Load is asynchronous - model is not ready yet")
        print("We will wait for ready status in the next step")

    except Exception as e:
        print(f"\nERROR: Failed to send load request")
        print(f"Details: {e}")
        print("\nPossible causes:")
        print("  - Model name does not match S3 repository structure")
        print("  - config.pbtxt is missing or invalid")
        print("  - ONNX file is corrupted")
        print("  - Triton encountered an internal error")
        raise


def wait_for_model_ready(triton_client, model_name, timeout):
    """
    Waits until the model is fully loaded and ready.

    The load API call returns immediately, but that does not mean the
    model is ready yet. Triton still needs to fetch the ONNX file from
    S3 (if not cached), parse it, initialize the weights, and start
    the inference engine. This can take several seconds.

    We poll the ready API at regular intervals until either the model
    is ready or the timeout is reached. This is a classic polling
    pattern for asynchronous operations.

    Args:
        triton_client: Initialized TritonClient
        model_name: Name of the model
        timeout: Maximum wait time in seconds

    Raises:
        Exception if model does not become ready within the timeout
    """
    print(f"\nWaiting for model to become ready (timeout: {timeout}s)...")

    try:
        # The TritonClient already has the wait_until_ready logic
        # We use it here and catch errors to provide helpful messages
        triton_client.wait_until_ready(model_name, timeout=timeout)

        print(f"  \u2713 Model is ready and accepting inference requests")

    except Exception as e:
        print(f"\nERROR: Model did not become ready within {timeout} seconds")
        print(f"Details: {e}")
        print("\nPossible causes:")
        print("  - Model is too large and takes longer to load (increase timeout)")
        print("  - ONNX file has compatibility issues with Triton version")
        print("  - Triton encountered an error during model initialization")
        print("  - Resource constraints (CPU/Memory) slowing down load")
        print("\nCheck Triton server logs for detailed error messages")
        raise


def verify_loaded_version(triton_client, model_name, expected_version):
    """
    Verifies that the expected version was actually loaded.

    After the load we want to make sure the correct version is in
    memory. This is an important sanity check - imagine the repository
    refresh did not work for some reason and Triton loaded an old
    version instead of the new one. We want to detect that immediately
    and not only later when unexpectedly old predictions come back.

    Args:
        triton_client: Initialized TritonClient
        model_name: Name of the model
        expected_version: The version we expect (or None)

    Returns:
        The actually loaded versions

    Raises:
        Exception if expected version is not loaded
    """
    print("\nVerifying loaded versions...")

    try:
        loaded_versions = triton_client.get_loaded_versions(model_name)

        if not loaded_versions:
            raise Exception(f"No versions loaded after load operation!")

        print(f"  Currently loaded versions: {loaded_versions}")

        # The highest loaded version is the active one (receives traffic)
        active_version = max(loaded_versions)
        print(f"  Active version (receives traffic): {active_version}")

        # If an expected version was specified, we check it
        if expected_version is not None:
            if active_version != expected_version:
                raise Exception(
                    f"Version mismatch! Expected {expected_version}, "
                    f"but active version is {active_version}. "
                    f"This suggests the repository refresh or load failed."
                )

            print(f"  \u2713 Verification passed: Active version matches expected ({expected_version})")
        else:
            print(f"  No expected version specified - cannot verify")
            print(f"  Assuming active version {active_version} is correct")

        return loaded_versions

    except Exception as e:
        print(f"\nERROR: Version verification failed")
        print(f"Details: {e}")
        raise


def main():
    """Main logic of the reload_triton step."""
    print("=" * 80)
    print("Step 5: Reload Triton Model")
    print("=" * 80)

    # Step 1: Parse arguments
    args = parse_arguments()

    print(f"\nConfiguration:")
    print(f"  Model Name: {args.model_name}")
    print(f"  Triton URL: {args.triton_url}")
    print(f"  Expected Version: {args.expected_version if args.expected_version else 'Not specified'}")
    print(f"  Load Timeout: {args.load_timeout}s")
    print(f"  Output Directory: {args.output_dir}")

    # Step 2: Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Step 3: Initialize Triton client
    print("\n" + "-" * 80)
    print("Initializing Triton Client...")

    try:
        triton_client = TritonClient(args.triton_url)
        print(f"  \u2713 Connected to Triton at {args.triton_url}")
    except Exception as e:
        print(f"\nERROR: Failed to initialize Triton client")
        print(f"Details: {e}")
        sys.exit(1)

    # Step 4: Triton health check
    print("\n" + "-" * 80)

    try:
        check_triton_health(triton_client)
    except Exception as e:
        sys.exit(1)

    # Step 5: Determine current state (before the load)
    print("\n" + "-" * 80)

    versions_before = get_currently_loaded_versions(triton_client, args.model_name)

    # Step 6: Refresh repository index
    print("\n" + "-" * 80)

    try:
        refresh_repository(triton_client)
    except Exception as e:
        sys.exit(1)

    # Step 7: Load model
    print("\n" + "-" * 80)

    try:
        load_model(triton_client, args.model_name)
    except Exception as e:
        sys.exit(1)

    # Step 8: Wait until ready
    print("\n" + "-" * 80)

    try:
        wait_for_model_ready(triton_client, args.model_name, args.load_timeout)
    except Exception as e:
        print("\nAttempting to get current state for debugging...")
        try:
            versions_after_fail = get_currently_loaded_versions(triton_client, args.model_name)
            print(f"Versions after failed load: {versions_after_fail}")
        except:
            print("Could not get version info")
        sys.exit(1)

    # Step 9: Version verification
    print("\n" + "-" * 80)

    try:
        loaded_versions = verify_loaded_version(
            triton_client,
            args.model_name,
            args.expected_version
        )
    except Exception as e:
        sys.exit(1)

    # Step 10: Write loaded version to file
    active_version = max(loaded_versions)
    version_output = os.path.join(args.output_dir, "loaded_version.txt")

    try:
        print(f"\nWriting loaded version to {version_output}")
        with open(version_output, 'w') as f:
            f.write(str(active_version))

        print("Loaded version written successfully")

    except Exception as e:
        print(f"\nWarning: Failed to write loaded version file: {e}")
        print("Continuing despite this error")

    # Step 11: Print summary
    print("\n" + "=" * 80)
    print("Step 5 completed successfully!")
    print("=" * 80)
    print(f"\nDeployment Summary:")
    print(f"  Model Name: {args.model_name}")
    print(f"  Active Version: {active_version}")
    print(f"  All Loaded Versions: {loaded_versions}")

    if versions_before:
        print(f"\nVersion Changes:")
        print(f"  Before: {versions_before}")
        print(f"  After:  {loaded_versions}")

        new_versions = set(loaded_versions) - set(versions_before)
        removed_versions = set(versions_before) - set(loaded_versions)

        if new_versions:
            print(f"  Added: {sorted(new_versions)}")
        if removed_versions:
            print(f"  Removed: {sorted(removed_versions)} (version_policy cleanup)")
    else:
        print(f"\nThis was the first load of this model")

    print(f"\nThe model is now serving inference requests at:")
    print(f"  {args.triton_url}/v2/models/{args.model_name}/infer")

    # Success Exit Code
    sys.exit(0)


if __name__ == '__main__':
    main()
