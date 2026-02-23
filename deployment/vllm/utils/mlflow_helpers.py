"""
MLflow Helper Functions for vLLM LoRA Adapter Deployment.

Handles downloading adapter artifacts from MLflow Model Registry
and managing deployment aliases (candidate, live, previous).
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
        """
        Parses the source parameter into an MLflow model URI.

        Args:
            model_name: Registered model name
            source: "alias:candidate" or "version:5"

        Returns:
            Tuple of (model_uri, source_type)
        """
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

    def download_adapter(self, model_name: str, source: str, output_dir: str) -> Dict[str, Any]:
        """
        Downloads a LoRA adapter from MLflow Model Registry.

        Expects the adapter to be logged as an artifact with files:
        - adapter_config.json
        - adapter_model.safetensors

        Args:
            model_name: Registered model name (e.g. "mistral-7b-lora")
            source: Where to load from (e.g. "alias:candidate" or "version:3")
            output_dir: Directory to save adapter files

        Returns:
            Dictionary with metadata (version, run_id, etc.)
        """
        model_uri, source_type = self.parse_source(model_name, source)
        print(f"Downloading adapter from {model_uri}")

        # Get model version info
        if source_type == "alias":
            alias = source.split(":", 1)[1]
            model_version = self.client.get_model_version_by_alias(model_name, alias)
        else:
            version = source.split(":", 1)[1]
            model_version = self.client.get_model_version(model_name, version)

        print(f"Found model version {model_version.version} from run {model_version.run_id}")

        # Download adapter artifacts
        temp_dir = tempfile.mkdtemp()
        try:
            artifact_path = self.client.download_artifacts(
                model_version.run_id,
                "adapter",  # LoRA adapters are logged under "adapter" path
                temp_dir,
            )

            # Copy adapter files to output directory
            os.makedirs(output_dir, exist_ok=True)
            adapter_files = []

            for filename in os.listdir(artifact_path):
                src = os.path.join(artifact_path, filename)
                dst = os.path.join(output_dir, filename)
                if os.path.isfile(src):
                    shutil.copy2(src, dst)
                    size_mb = os.path.getsize(dst) / (1024 * 1024)
                    print(f"  {filename}: {size_mb:.2f} MB")
                    adapter_files.append(filename)

            print(f"Downloaded {len(adapter_files)} adapter files")

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

        # Get run metadata
        run = self.client.get_run(model_version.run_id)

        metadata = {
            "model_name": model_name,
            "version": model_version.version,
            "run_id": model_version.run_id,
            "source": source,
            "adapter_files": adapter_files,
            "metrics": run.data.metrics,
            "tags": dict(model_version.tags),
        }

        return metadata

    def get_model_version_by_alias(self, model_name: str, alias: str) -> Optional[Dict]:
        """Gets model version info for a specific alias."""
        try:
            model_version = self.client.get_model_version_by_alias(model_name, alias)
            return {
                "version": model_version.version,
                "run_id": model_version.run_id,
                "status": model_version.status,
            }
        except MlflowException:
            return None

    def update_deployment_aliases(
        self, model_name: str, new_version: str
    ) -> Dict[str, str]:
        """
        Updates deployment aliases after a successful promotion.

        Moves: current live → previous, new_version → live

        Args:
            model_name: Registered model name
            new_version: Version to promote to live

        Returns:
            Dictionary with alias updates
        """
        result = {}

        # Move current live to previous
        current_live = self.get_model_version_by_alias(model_name, "live")
        if current_live:
            old_version = current_live["version"]
            print(f"Setting 'previous' alias to version {old_version}")
            self.client.set_registered_model_alias(model_name, "previous", old_version)
            result["previous"] = old_version
        else:
            print("No current 'live' version found (first promotion)")

        # Set new version as live
        print(f"Setting 'live' alias to version {new_version}")
        self.client.set_registered_model_alias(model_name, "live", new_version)
        result["live"] = new_version

        return result

    def set_alias(self, model_name: str, alias: str, version: str) -> None:
        """Sets a specific alias to a specific version."""
        print(f"Setting alias '{alias}' to version {version} for {model_name}")
        self.client.set_registered_model_alias(model_name, alias, version)

    def delete_alias(self, model_name: str, alias: str) -> None:
        """Deletes an alias."""
        print(f"Deleting alias '{alias}' from {model_name}")
        self.client.delete_registered_model_alias(model_name, alias)
