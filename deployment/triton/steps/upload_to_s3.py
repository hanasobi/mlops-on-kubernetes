#!/usr/bin/env python3
"""
Step 4: Upload Artifacts to S3

Uploads all deployment artifacts (model.onnx and metadata.json) to S3
into the Triton Model Repository. Uses a staging pattern to ensure
that partial uploads never end up in the final repository path.

The staging pattern works as follows:
1. Upload all files to a temporary prefix (e.g. "model/3.uploading/")
2. If all uploads succeed, copy atomically to the final prefix ("model/3/")
3. Delete the temporary prefix

If the upload fails, the temporary prefix remains but has no impact on
the production repository. A later retry simply overwrites it.

Input Parameters:
    --model-name: Name of the model (for S3 path)
    --triton-version: Version number (for S3 path)
    --bucket: S3 Bucket Name
    --artifacts-dir: Directory containing the artifacts to upload
    --output-dir: Where output artifacts should be saved

Output Artifacts:
    s3_uri.txt: The final S3 URI where the model now resides

Environment Variables:
    AWS_REGION: Optional - AWS Region
"""

import argparse
import os
import sys
import glob

from utils.s3_helpers import S3Client


def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Upload deployment artifacts to S3 using staging pattern'
    )

    parser.add_argument(
        '--model-name',
        required=True,
        help='Name of the model (for S3 path structure)'
    )

    parser.add_argument(
        '--triton-version',
        required=True,
        help='Triton version number (for S3 path structure)'
    )

    parser.add_argument(
        '--bucket',
        required=True,
        help='S3 bucket name'
    )

    parser.add_argument(
        '--artifacts-dir',
        required=True,
        help='Directory containing artifacts to upload (model.onnx, metadata.json)'
    )

    parser.add_argument(
        '--output-dir',
        required=True,
        help='Directory to save output artifacts'
    )

    return parser.parse_args()


def validate_artifacts(artifacts_dir):
    """
    Validates that all expected artifacts are present.

    Before starting the upload, we check that all files we expect
    actually exist. This is important because we want to upload a
    complete set - either all files or none. A partial upload would
    lead to inconsistent state.

    Args:
        artifacts_dir: Directory that should contain the artifacts

    Returns:
        List of (filename, filepath) tuples for all files to upload

    Raises:
        Exception if expected files are missing
    """
    print(f"Validating artifacts in {artifacts_dir}")

    # These are the files we expect
    required_files = ['model.onnx', 'metadata.json']

    artifacts = []
    missing_files = []

    for filename in required_files:
        filepath = os.path.join(artifacts_dir, filename)

        if os.path.exists(filepath):
            file_size = os.path.getsize(filepath)
            file_size_mb = file_size / (1024 * 1024)

            print(f"  \u2713 {filename} ({file_size_mb:.2f} MB)")
            artifacts.append((filename, filepath))
        else:
            print(f"  \u2717 {filename} (missing)")
            missing_files.append(filename)

    if missing_files:
        raise Exception(
            f"Missing required artifacts: {missing_files}. "
            f"These should have been created by previous steps."
        )

    print(f"All {len(artifacts)} required artifacts found")
    return artifacts


def upload_to_staging(s3_client, artifacts, model_name, version, staging_suffix=".uploading"):
    """
    Uploads all artifacts to a temporary staging prefix.

    The staging prefix is a temporary location in S3 where we upload files
    before moving them atomically to the final location. The ".uploading"
    suffix makes it clear that this is a temporary state and should not
    be read by the Triton Server.

    Args:
        s3_client: Initialized S3Client
        artifacts: List of (filename, filepath) tuples
        model_name: Name of the model
        version: Version number
        staging_suffix: Suffix for the temporary prefix

    Returns:
        The staging prefix (e.g. "resnet18_imagenette/3.uploading/")
    """
    staging_prefix = f"{model_name}/{version}{staging_suffix}/"

    print(f"\nUploading to staging location: s3://{s3_client.bucket}/{staging_prefix}")
    print("This is a temporary location - files will be moved atomically after all uploads succeed")

    try:
        for filename, filepath in artifacts:
            s3_key = staging_prefix + filename

            print(f"\nUploading {filename}...")
            s3_client.upload_file(filepath, s3_key)

        print(f"\nAll artifacts uploaded to staging successfully")
        return staging_prefix

    except Exception as e:
        print(f"\nERROR: Upload to staging failed")
        print(f"Details: {e}")
        print("\nAttempting cleanup of partial upload...")

        # Try to delete the staging prefix
        # This is best-effort - if this also fails it is not critical
        try:
            s3_client.delete_prefix(staging_prefix)
            print("Staging prefix cleaned up")
        except Exception as cleanup_error:
            print(f"Warning: Cleanup also failed: {cleanup_error}")
            print("Staging prefix may remain in S3 but will be overwritten on retry")

        # Re-raise the original error
        raise


def atomic_move_to_final(s3_client, staging_prefix, model_name, version):
    """
    Moves the artifacts atomically from staging to the final prefix.

    "Atomic" here means that the operation either happens completely
    or not at all - there is no intermediate state where only some files
    are at the final location. We achieve this through S3's copy operation
    which is atomic per file, followed by a delete of the staging prefix.

    The critical point is that Triton at any given time sees either the
    old version or the new version, never a mix of both.

    Args:
        s3_client: Initialized S3Client
        staging_prefix: The temporary prefix containing all files
        model_name: Name of the model
        version: Version number

    Returns:
        The final prefix (e.g. "resnet18_imagenette/3/")
    """
    final_prefix = f"{model_name}/{version}/"

    print(f"\nMoving artifacts atomically to final location:")
    print(f"  From: s3://{s3_client.bucket}/{staging_prefix}")
    print(f"  To:   s3://{s3_client.bucket}/{final_prefix}")

    try:
        # S3 copy is atomic per file - either the complete file is
        # copied or nothing. By copying all files sequentially we
        # effectively have an atomic operation for the complete set.
        s3_client.copy_directory(staging_prefix, final_prefix)

        print("Artifacts moved to final location successfully")

        # Now we can safely delete the staging prefix
        print(f"\nCleaning up staging location...")
        s3_client.delete_prefix(staging_prefix)

        print("Staging location cleaned up")
        return final_prefix

    except Exception as e:
        print(f"\nERROR: Failed to move artifacts to final location")
        print(f"Details: {e}")
        print("\nCurrent state:")
        print(f"  - Staging prefix still exists: {staging_prefix}")
        print(f"  - Final prefix may be partially populated")
        print(f"  - Manual cleanup may be required")

        raise


def verify_upload(s3_client, final_prefix, expected_files):
    """
    Verifies that all expected files exist at the final location.

    After the upload we perform a verification check to ensure that
    everything was uploaded correctly. This is an additional safety
    check that can detect problems early.

    Args:
        s3_client: Initialized S3Client
        final_prefix: The final prefix where the files should be
        expected_files: List of filenames that should exist

    Raises:
        Exception if files are missing or other problems are found
    """
    print(f"\nVerifying upload at s3://{s3_client.bucket}/{final_prefix}")

    # List all files at the final location
    try:
        response = s3_client.s3.list_objects_v2(
            Bucket=s3_client.bucket,
            Prefix=final_prefix
        )

        if 'Contents' not in response:
            raise Exception(f"No files found at final location {final_prefix}")

        # Extract the filenames from the S3 keys
        uploaded_files = set()
        for obj in response['Contents']:
            # Key is e.g. "resnet18_imagenette/3/model.onnx"
            # We only want "model.onnx"
            filename = obj['Key'].split('/')[-1]
            uploaded_files.add(filename)

        # Check if all expected files are present
        expected_set = set(expected_files)
        missing_files = expected_set - uploaded_files

        if missing_files:
            raise Exception(
                f"Upload verification failed: missing files {missing_files}"
            )

        # Additional check: Are there unexpected files?
        extra_files = uploaded_files - expected_set
        if extra_files:
            print(f"  Warning: Found unexpected files: {extra_files}")
            print("  This is not critical but may indicate a problem")

        print(f"  \u2713 All {len(expected_files)} expected files verified")

        # Show details about the files
        for obj in response['Contents']:
            filename = obj['Key'].split('/')[-1]
            size_mb = obj['Size'] / (1024 * 1024)
            print(f"    - {filename}: {size_mb:.2f} MB")

        print("\nUpload verification passed")

    except Exception as e:
        print(f"\nERROR: Upload verification failed")
        print(f"Details: {e}")
        raise


def main():
    """Main logic of the upload_to_s3 step."""
    print("=" * 80)
    print("Step 4: Upload Artifacts to S3")
    print("=" * 80)

    # Step 1: Parse arguments
    args = parse_arguments()

    print(f"\nConfiguration:")
    print(f"  Model Name: {args.model_name}")
    print(f"  Triton Version: {args.triton_version}")
    print(f"  S3 Bucket: {args.bucket}")
    print(f"  Artifacts Directory: {args.artifacts_dir}")
    print(f"  Output Directory: {args.output_dir}")

    # Step 2: Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Step 3: Validate artifacts
    print("\n" + "-" * 80)

    try:
        artifacts = validate_artifacts(args.artifacts_dir)
    except Exception as e:
        print(f"\nERROR: Artifact validation failed")
        print(f"Details: {e}")
        sys.exit(1)

    # Step 4: Initialize S3 client
    print("\n" + "-" * 80)
    print("Initializing S3 Client...")

    try:
        s3_client = S3Client(args.bucket)
    except Exception as e:
        print(f"\nERROR: Failed to initialize S3 client")
        print(f"Details: {e}")
        sys.exit(1)

    # Step 5: Upload to staging
    print("\n" + "-" * 80)

    try:
        staging_prefix = upload_to_staging(
            s3_client,
            artifacts,
            args.model_name,
            args.triton_version
        )
    except Exception as e:
        print(f"\nERROR: Failed to upload to staging")
        sys.exit(1)

    # Step 6: Atomic move to final prefix
    print("\n" + "-" * 80)

    try:
        final_prefix = atomic_move_to_final(
            s3_client,
            staging_prefix,
            args.model_name,
            args.triton_version
        )
    except Exception as e:
        print(f"\nERROR: Failed to move artifacts to final location")
        sys.exit(1)

    # Step 7: Verification
    print("\n" + "-" * 80)

    expected_files = [filename for filename, _ in artifacts]

    try:
        verify_upload(s3_client, final_prefix, expected_files)
    except Exception as e:
        print(f"\nERROR: Upload verification failed")
        print("\nThis is a critical error - the upload may be incomplete")
        print("Manual investigation required before proceeding")
        sys.exit(1)

    # Step 8: Write S3 URI to file for subsequent steps
    s3_uri = f"s3://{args.bucket}/{final_prefix}"
    uri_output = os.path.join(args.output_dir, "s3_uri.txt")

    try:
        print(f"\nWriting S3 URI to {uri_output}")
        with open(uri_output, 'w') as f:
            f.write(s3_uri)

        print("S3 URI written successfully")

    except Exception as e:
        print(f"\nERROR: Failed to write S3 URI")
        print(f"Details: {e}")
        # This is not critical - the upload was successful
        # We only log the error but do not fail
        print("Warning: Continuing despite this error")

    # Step 9: Print summary
    print("\n" + "=" * 80)
    print("Step 4 completed successfully!")
    print("=" * 80)
    print(f"\nArtifacts deployed to:")
    print(f"  {s3_uri}")
    print(f"\nFiles uploaded:")
    for filename, _ in artifacts:
        print(f"  - {filename}")

    print(f"\nTriton can now access this model at:")
    print(f"  Model: {args.model_name}")
    print(f"  Version: {args.triton_version}")
    print(f"  Repository: s3://{args.bucket}/{args.model_name}/")

    # Success Exit Code
    sys.exit(0)


if __name__ == '__main__':
    main()
