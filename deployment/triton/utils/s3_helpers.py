"""
S3 Helper Functions for Model Deployment.

Provides high-level functions for S3 operations that we need in multiple
deployment steps. Uses boto3 internally but abstracts away the complexity.
"""

import boto3
import os
import json
import tempfile
from typing import List, Optional
from botocore.exceptions import ClientError


class S3Client:
    """Wrapper around boto3 S3 client with deployment-specific methods."""

    def __init__(self, bucket_name: str):
        """
        Initializes the S3 client.

        Args:
            bucket_name: Name of the S3 bucket (e.g. "my-triton-models")
        """
        self.bucket = bucket_name
        self.s3 = boto3.client('s3')

    def list_model_versions(self, model_name: str) -> List[int]:
        """
        Lists all existing Triton version numbers for a model.

        Scans the S3 repository and extracts the version numbers from
        the folder names. Triton versions are always integer folders.

        Args:
            model_name: Name of the model (e.g. "resnet18_imagenette")

        Returns:
            List of integer version numbers, sorted.
            Empty list if the model does not exist yet.

        Example:
            >>> client = S3Client("my-bucket")
            >>> versions = client.list_model_versions("resnet18_imagenette")
            >>> print(versions)  # [1, 2, 3]
        """
        prefix = f"{model_name}/"

        try:
            # List all "folders" under the model
            response = self.s3.list_objects_v2(
                Bucket=self.bucket,
                Prefix=prefix,
                Delimiter='/'
            )

            # CommonPrefixes contains the "folder" structures
            if 'CommonPrefixes' not in response:
                # Model does not exist in S3 yet
                return []

            versions = []
            for prefix_obj in response['CommonPrefixes']:
                # Prefix is e.g. "resnet18_imagenette/1/"
                # We extract the "1"
                folder_name = prefix_obj['Prefix'].rstrip('/').split('/')[-1]

                # Try to parse as integer
                # Ignore non-integer folders (e.g. "config" or ".uploading")
                try:
                    version = int(folder_name)
                    versions.append(version)
                except ValueError:
                    # Not a version number, skip
                    continue

            return sorted(versions)

        except ClientError as e:
            print(f"Error listing versions: {e}")
            raise

    def version_exists(self, model_name: str, version: int) -> bool:
        """
        Checks whether a specific version exists in S3.

        Args:
            model_name: Name of the model
            version: Version number to check

        Returns:
            True if the version exists, False otherwise
        """
        # Check if the version folder exists
        prefix = f"{model_name}/{version}/"

        try:
            response = self.s3.list_objects_v2(
                Bucket=self.bucket,
                Prefix=prefix,
                MaxKeys=1  # We only need to know if ANYTHING is there
            )

            return response.get('KeyCount', 0) > 0

        except ClientError as e:
            print(f"Error checking version existence: {e}")
            return False

    def upload_file(self, local_path: str, s3_key: str) -> None:
        """
        Uploads a local file to S3.

        Args:
            local_path: Path to the local file
            s3_key: Destination path in S3 (e.g. "resnet18_imagenette/3/model.onnx")

        Raises:
            FileNotFoundError: If local_path does not exist
            ClientError: On S3 upload error
        """
        if not os.path.exists(local_path):
            raise FileNotFoundError(f"Local file not found: {local_path}")

        try:
            print(f"Uploading {local_path} to s3://{self.bucket}/{s3_key}")
            self.s3.upload_file(local_path, self.bucket, s3_key)
            print(f"Upload successful")

        except ClientError as e:
            print(f"Upload failed: {e}")
            raise

    def download_file(self, s3_key: str, local_path: str) -> None:
        """
        Downloads a file from S3.

        Args:
            s3_key: Source path in S3
            local_path: Destination path locally

        Raises:
            ClientError: On S3 download error
        """
        try:
            print(f"Downloading s3://{self.bucket}/{s3_key} to {local_path}")
            self.s3.download_file(self.bucket, s3_key, local_path)
            print(f"Download successful")

        except ClientError as e:
            print(f"Download failed: {e}")
            raise

    def copy_directory(self, source_prefix: str, dest_prefix: str) -> None:
        """
        Copies all files from one S3 prefix to another.

        This is our "atomic rename" operation for the staging pattern.
        S3 has no native rename, but copy is atomic per file.

        Args:
            source_prefix: Source prefix (e.g. "model/3.uploading/")
            dest_prefix: Destination prefix (e.g. "model/3/")
        """
        # List all files under source
        try:
            response = self.s3.list_objects_v2(
                Bucket=self.bucket,
                Prefix=source_prefix
            )

            if 'Contents' not in response:
                print(f"No files found under {source_prefix}")
                return

            # Copy each file
            for obj in response['Contents']:
                source_key = obj['Key']
                # Replace source_prefix with dest_prefix
                dest_key = source_key.replace(source_prefix, dest_prefix, 1)

                copy_source = {'Bucket': self.bucket, 'Key': source_key}
                print(f"Copying {source_key} to {dest_key}")

                self.s3.copy_object(
                    CopySource=copy_source,
                    Bucket=self.bucket,
                    Key=dest_key
                )

            print(f"Directory copy complete: {source_prefix} -> {dest_prefix}")

        except ClientError as e:
            print(f"Copy failed: {e}")
            raise

    def delete_prefix(self, prefix: str) -> None:
        """
        Deletes all files under a prefix (for cleanup).

        Args:
            prefix: Prefix to delete (e.g. "model/3.uploading/")
        """
        try:
            # List all files
            response = self.s3.list_objects_v2(
                Bucket=self.bucket,
                Prefix=prefix
            )

            if 'Contents' not in response:
                print(f"No files to delete under {prefix}")
                return

            # Delete all files
            objects_to_delete = [{'Key': obj['Key']} for obj in response['Contents']]

            self.s3.delete_objects(
                Bucket=self.bucket,
                Delete={'Objects': objects_to_delete}
            )

            print(f"Deleted {len(objects_to_delete)} files under {prefix}")

        except ClientError as e:
            print(f"Delete failed: {e}")
            raise

    def download_json(self, s3_key: str) -> dict:
        """
        Downloads and parses a JSON file from S3.

        [Docstring unchanged...]
        """
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w+', suffix='.json', delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Download
            self.download_file(s3_key, tmp_path)

            # Parse JSON
            with open(tmp_path, 'r') as f:
                data = json.load(f)

            return data

        finally:
            # Cleanup of the temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)

    def upload_json(self, data: dict, s3_key: str) -> None:
      """
      Serializes a dictionary as JSON and uploads it to S3.

      Companion method to download_json - takes a dictionary,
      writes it as formatted JSON to a temporary file,
      and uploads it to S3.

      Args:
          data: Dictionary to serialize
          s3_key: Destination path in S3

      Raises:
          ClientError: On S3 upload error

      Example:
          >>> metadata = {
          ...     "mlflow": {"version": "5"},
          ...     "triton": {"version": "3"}
          ... }
          >>> client.upload_json(metadata, "resnet18_imagenette/3/metadata.json")
      """
      # Create temporary file
      with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
          tmp_path = tmp.name
          json.dump(data, tmp, indent=2)

      try:
          # Upload
          self.upload_file(tmp_path, s3_key)

      finally:
          # Cleanup of the temporary file
          if os.path.exists(tmp_path):
              os.unlink(tmp_path)
