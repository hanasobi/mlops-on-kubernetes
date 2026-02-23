"""
MLflow Model Registry Helper Functions.

FIXED v3:
- Returns file paths separately instead of setting attributes on ONNX model
- Handles ONNX external data format (.onnx + .onnx.data files)
"""

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException
from typing import Dict, Any, Optional, Tuple
import os
import tempfile
import shutil


class MLflowHelper:
    """Helper class for MLflow Model Registry operations."""

    def __init__(self, tracking_uri: str = None):
        """Initializes the MLflow helper."""
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)

        self.client = MlflowClient()
        self.tracking_uri = mlflow.get_tracking_uri()
        print(f"MLflow Helper initialized with tracking URI: {self.tracking_uri}")

    def parse_source(self, model_name: str, source: str) -> Tuple[str, str]:
        """Parses the source parameter into an MLflow model URI."""
        if source.startswith("alias:"):
            alias = source.split(":", 1)[1]
            model_uri = f"models:/{model_name}@{alias}"
            return model_uri, "alias"

        elif source.startswith("version:"):
            version = source.split(":", 1)[1]
            model_uri = f"models:/{model_name}/{version}"
            return model_uri, "version"

        else:
            raise ValueError(
                f"Invalid source format: {source}. "
                f"Expected 'alias:NAME' or 'version:NUMBER'"
            )

    def load_model(self, model_name: str, source: str = "alias:deploy") -> Tuple[Any, Dict, Dict]:
        """
        Loads an ONNX model from MLflow and retrieves its metadata.

        Args:
            model_name: Name of the registered model
            source: Where to load the model from (default: "alias:deploy")

        Returns:
            Tuple of (model, metadata, file_paths)
            - model: The loaded ONNX model object
            - metadata: Dictionary with MLflow metadata
            - file_paths: Dictionary with 'onnx' and optional 'onnx_data' keys
        """
        model_uri, source_type = self.parse_source(model_name, source)

        print(f"Loading model from {model_uri}")

        try:
            # Get model version info
            if source_type == "alias":
                alias = source.split(":", 1)[1]
                model_version = self.client.get_model_version_by_alias(
                    model_name,
                    alias
                )
            else:
                version = source.split(":", 1)[1]
                model_version = self.client.get_model_version(
                    model_name,
                    version
                )

            print(f"Found model version {model_version.version} from run {model_version.run_id}")

            # Get run info
            run = self.client.get_run(model_version.run_id)

            # Download artifacts directly
            print(f"Downloading artifacts from run {model_version.run_id}")
            temp_dir = tempfile.mkdtemp()

            try:
                # Download all artifacts
                artifact_path = self.client.download_artifacts(
                    model_version.run_id,
                    "",  # Empty string = all artifacts
                    temp_dir
                )

                # Search for ONNX file and optional .data file
                onnx_file = None
                onnx_data_file = None

                for root, dirs, files in os.walk(artifact_path):
                    for file in files:
                        full_path = os.path.join(root, file)

                        if file.endswith('.onnx') and not file.endswith('.onnx.data'):
                            onnx_file = full_path
                            print(f"Found ONNX file: {file}")

                        elif file.endswith('.onnx.data'):
                            onnx_data_file = full_path
                            print(f"Found ONNX external data file: {file}")

                if not onnx_file:
                    raise Exception("No ONNX file found in artifacts")

                # Info about external data
                if onnx_data_file:
                    data_size_mb = os.path.getsize(onnx_data_file) / (1024 * 1024)
                    print(f"Model uses external data format (data file: {data_size_mb:.2f} MB)")

                # Load ONNX model
                import onnx
                model = onnx.load(onnx_file)
                print(f"ONNX model loaded successfully")

                # Return file paths for the caller
                file_paths = {
                    'onnx': onnx_file,
                    'onnx_data': onnx_data_file,  # None if no external data
                    'temp_dir': temp_dir  # Must be cleaned up by the caller
                }

            except Exception as e:
                # Cleanup on error
                shutil.rmtree(temp_dir, ignore_errors=True)
                raise

            # Assemble metadata dictionary
            metadata = {
                'model_name': model_name,
                'version': model_version.version,
                'run_id': model_version.run_id,
                'source': source,
                'source_type': source_type,
                'status': model_version.status,
                'creation_timestamp': model_version.creation_timestamp,
                'last_updated_timestamp': model_version.last_updated_timestamp,
                'run_name': run.info.run_name,
                'metrics': run.data.metrics,
                'params': run.data.params,
                'tags': dict(model_version.tags)
            }

            print(f"Successfully loaded model version {metadata['version']}")
            return model, metadata, file_paths

        except MlflowException as e:
            print(f"Failed to load model: {e}")
            raise

    def get_model_version_by_alias(self, model_name: str, alias: str) -> Optional[Dict]:
        """Gets model version info for a specific alias."""
        try:
            model_version = self.client.get_model_version_by_alias(model_name, alias)

            return {
                'version': model_version.version,
                'run_id': model_version.run_id,
                'status': model_version.status,
                'creation_timestamp': model_version.creation_timestamp
            }

        except MlflowException:
            return None

    def update_deployment_aliases(self, model_name: str, new_version: str, dry_run: bool = False) -> Dict[str, str]:
        """Updates the deployment aliases after a successful deploy."""
        current_live = self.get_model_version_by_alias(model_name, "live")

        result = {}

        if current_live:
            old_version = current_live['version']
            print(f"Current live version: {old_version}")

            if not dry_run:
                print(f"Setting 'previous' alias to version {old_version}")
                self.client.set_registered_model_alias(model_name, "previous", old_version)
                result['previous'] = old_version
            else:
                print(f"[DRY RUN] Would set 'previous' to version {old_version}")
        else:
            print("No current 'live' version found (first deployment)")

        if not dry_run:
            print(f"Setting 'live' alias to version {new_version}")
            self.client.set_registered_model_alias(model_name, "live", new_version)
            result['live'] = new_version
        else:
            print(f"[DRY RUN] Would set 'live' to version {new_version}")

        return result

    def set_alias(self, model_name: str, alias: str, version: str) -> None:
        """Sets a specific alias to a specific version."""
        print(f"Setting alias '{alias}' to version {version} for {model_name}")

        try:
            self.client.set_registered_model_alias(model_name, alias, version)
            print(f"Alias set successfully")
        except MlflowException as e:
            print(f"Failed to set alias: {e}")
            raise

    def delete_alias(self, model_name: str, alias: str) -> None:
        """Deletes an alias completely."""
        print(f"Deleting alias '{alias}' from {model_name}")

        try:
            self.client.delete_registered_model_alias(model_name, alias)
            print(f"Alias deleted successfully")
        except MlflowException as e:
            print(f"Failed to delete alias: {e}")
            raise

    def get_all_versions(self, model_name: str) -> list:
        """Lists all registered versions of a model."""
        try:
            versions = self.client.search_model_versions(f"name='{model_name}'")
            versions.sort(key=lambda v: int(v.version), reverse=True)
            return [v.version for v in versions]
        except MlflowException as e:
            print(f"Failed to list versions: {e}")
            raise
