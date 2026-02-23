#!/usr/bin/env python3
"""
Step 2: Determine Triton Version Number

Determines which version number we will use in Triton for this deployment.
Scans the S3 model repository to find existing versions and calculates
the next available version.

Input Parameters:
    --model-name: Name of the model (must match S3 repository structure)
    --bucket: S3 bucket name (e.g. "my-triton-models")
    --force-version: Optional - force a specific version number
    --output-dir: Where the output artifacts should be saved

Output Artifacts:
    triton_version.txt: A single line with the version number (e.g. "3")

Environment Variables:
    AWS_REGION: Optional - AWS Region (if not default)
"""

import argparse
import os
import sys

from utils.s3_helpers import S3Client


def parse_arguments():
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Determine the next Triton version number for deployment'
    )
    
    parser.add_argument(
        '--model-name',
        required=True,
        help='Name of the model (must match S3 repository structure)'
    )
    
    parser.add_argument(
        '--bucket',
        required=True,
        help='S3 bucket name (e.g., my-triton-models)'
    )
    
    parser.add_argument(
        '--force-version',
        type=int,
        default=None,
        help='Force a specific version number (use with caution!)'
    )
    
    parser.add_argument(
        '--output-dir',
        required=True,
        help='Directory to save output artifacts'
    )
    
    return parser.parse_args()


def validate_force_version(force_version, existing_versions):
    """
    Validates that force-version is sensible.

    We only allow force-version in certain situations. The main rule
    is that we don't want to overwrite a version that is already deployed
    and currently running. That would lead to inconsistent state.

    Allowed scenarios:
    1. Force-version is higher than all existing ones -> OK (skip forward)
    2. Force-version does not exist in S3 -> OK (fill gap)
    3. Force-version already exists -> WARNING but allowed (re-deploy)

    Not allowed:
    - Force-version is 0 or negative -> Invalid

    Args:
        force_version: The desired version number
        existing_versions: List of already existing versions

    Returns:
        True if valid, False if invalid
    """
    # Basic validation: version must be positive
    if force_version <= 0:
        print(f"ERROR: force-version must be positive (got {force_version})")
        return False
    
    # If the version already exists, we issue a warning
    if force_version in existing_versions:
        print(f"\n⚠️  WARNING: Version {force_version} already exists in S3!")
        print("This deployment will OVERWRITE the existing version.")
        print("This is allowed but should only be done intentionally.")
        print("Common use cases:")
        print("  - Fixing a broken deployment")
        print("  - Re-deploying after manual cleanup")
        print("\nIf this is unintentional, abort the workflow now!")
        # We still return True - it's allowed, but warned
        return True
    
    # All other cases are OK
    return True


def calculate_next_version(existing_versions):
    """
    Calculates the next available version number.

    The logic is simple: Take the highest existing version and add one.
    The edge case is when no versions exist yet, in which case we start
    at one.

    Args:
        existing_versions: List of integer version numbers

    Returns:
        The next version number as integer

    Example:
        >>> calculate_next_version([1, 2, 3])
        4
        >>> calculate_next_version([1, 3])  # Gap at 2
        4  # We don't fill gaps, we always move forward
        >>> calculate_next_version([])
        1  # First version ever
    """
    if not existing_versions:
        # First deployment ever for this model
        print("No existing versions found - this is the first deployment")
        return 1
    
    # Sort to find the highest (should already be sorted, but better safe than sorry)
    max_version = max(existing_versions)
    next_version = max_version + 1
    
    print(f"Existing versions: {existing_versions}")
    print(f"Highest version: {max_version}")
    print(f"Next version: {next_version}")
    
    return next_version


def main():
    """Main logic of the determine_version step."""
    print("=" * 80)
    print("Step 2: Determine Triton Version Number")
    print("=" * 80)
    
    # Step 1: Parse arguments
    args = parse_arguments()
    
    print(f"\nConfiguration:")
    print(f"  Model Name: {args.model_name}")
    print(f"  S3 Bucket: {args.bucket}")
    print(f"  Force Version: {args.force_version if args.force_version else 'Auto (max + 1)'}")
    print(f"  Output Directory: {args.output_dir}")
    
    # Step 2: Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Step 3: Initialize S3 client
    print("\nInitializing S3 Client...")
    try:
        s3_client = S3Client(args.bucket)
    except Exception as e:
        print(f"\nERROR: Failed to initialize S3 client")
        print(f"Details: {e}")
        print("\nPossible causes:")
        print("  - Invalid bucket name")
        print("  - Missing AWS credentials")
        print("  - Insufficient S3 permissions")
        sys.exit(1)
    
    # Step 4: Scan existing versions from S3
    print(f"\nScanning S3 repository for existing versions of '{args.model_name}'...")
    
    try:
        existing_versions = s3_client.list_model_versions(args.model_name)
        
        if existing_versions:
            print(f"Found {len(existing_versions)} existing version(s): {existing_versions}")
        else:
            print("No existing versions found in S3 repository")
            
    except Exception as e:
        print(f"\nERROR: Failed to scan S3 repository")
        print(f"Details: {e}")
        print("\nPossible causes:")
        print("  - S3 bucket does not exist")
        print("  - Network connectivity issues")
        print("  - Insufficient S3 read permissions")
        sys.exit(1)
    
    # Step 5: Determine version number
    if args.force_version:
        print(f"\nUsing forced version: {args.force_version}")
        
        # Validate that force-version is sensible
        if not validate_force_version(args.force_version, existing_versions):
            print("\nERROR: Invalid force-version specified")
            sys.exit(1)
        
        triton_version = args.force_version
        
    else:
        print("\nCalculating next version automatically...")
        triton_version = calculate_next_version(existing_versions)
    
    # Step 6: Write version number to file
    output_path = os.path.join(args.output_dir, "triton_version.txt")
    
    try:
        print(f"\nWriting version number to {output_path}")
        with open(output_path, 'w') as f:
            f.write(str(triton_version))
        
        print(f"Version number written successfully")
        
    except Exception as e:
        print(f"\nERROR: Failed to write version number")
        print(f"Details: {e}")
        sys.exit(1)
    
    # Step 7: Print summary
    print("\n" + "=" * 80)
    print("Step 2 completed successfully!")
    print("=" * 80)
    print(f"\nDetermined Triton Version: {triton_version}")
    print(f"Output Artifact: {output_path}")
    
    # Additional context information that helps with debugging
    if existing_versions:
        print(f"\nDeployment Context:")
        print(f"  Previous versions in S3: {existing_versions}")
        print(f"  This will be version #{len(existing_versions) + 1}")
    else:
        print(f"\nDeployment Context:")
        print(f"  This is the FIRST deployment of this model to Triton")
    
    # Success Exit Code
    sys.exit(0)


if __name__ == '__main__':
    main()
    