"""
Triton Inference Server Management API Client.

Encapsulates all management API calls that we need for model deployment.
The Triton Management API runs on port 8000 and provides endpoints for
model loading, unloading, health checks, and metadata queries.

Documentation: https://github.com/triton-inference-server/server/blob/main/docs/protocol/extension_model_repository.md
"""

import requests
import time
from typing import List, Optional, Dict, Any


class TritonClient:
    """Client for Triton Inference Server Management API."""

    def __init__(self, base_url: str = "http://triton-service.ai-platform:8000"):
        """
        Initializes the Triton client.

        Args:
            base_url: Base URL of the Triton server (HTTP endpoint)
                     Default is the service name in Kubernetes
        """
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()  # For connection pooling

    def _make_request(self, method: str, endpoint: str, **kwargs) -> requests.Response:
        """
        Makes an HTTP request with error handling.

        Helper method to centralize request logic and make
        error handling consistent.

        Args:
            method: HTTP method (GET, POST, etc.)
            endpoint: API endpoint (e.g. "/v2/models/resnet18/ready")
            **kwargs: Additional arguments for requests (params, json, etc.)

        Returns:
            requests.Response object

        Raises:
            requests.exceptions.RequestException: On connection errors
        """
        url = f"{self.base_url}{endpoint}"

        try:
            response = self.session.request(method, url, **kwargs)
            return response
        except requests.exceptions.ConnectionError as e:
            raise Exception(f"Cannot connect to Triton at {self.base_url}: {e}")
        except requests.exceptions.Timeout as e:
            raise Exception(f"Request to Triton timed out: {e}")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Request to Triton failed: {e}")

    def is_server_ready(self) -> bool:
        """
        Checks whether the Triton server is ready.

        Returns:
            True if the server is running and ready, False otherwise
        """
        try:
            response = self._make_request("GET", "/v2/health/ready")
            return response.status_code == 200
        except Exception as e:
            print(f"Server ready check failed: {e}")
            return False

    def refresh_repository_index(self) -> None:
        """
        Triggers Triton to rescan the model repository.

        This is important after S3 uploads - Triton needs to know that
        new model versions are available. In EXPLICIT mode, Triton does
        not scan automatically; we must trigger it explicitly.

        Raises:
            Exception: If repository refresh fails
        """
        print("Refreshing Triton repository index...")

        response = self._make_request("POST", "/v2/repository/index")

        if response.status_code == 200:
            print("Repository index refreshed successfully")
        else:
            error_msg = f"Repository refresh failed: {response.status_code} - {response.text}"
            raise Exception(error_msg)

    def get_loaded_versions(self, model_name: str) -> List[int]:
        """
        Determines which versions of a model are currently loaded.

        Uses the model metadata API to find out which versions are
        in memory. This is critical for our rollback logic.

        Args:
            model_name: Name of the model

        Returns:
            List of integer version numbers, sorted.
            Empty list if the model is not loaded.

        Example:
            >>> client = TritonClient()
            >>> versions = client.get_loaded_versions("resnet18_imagenette")
            >>> print(versions)  # [2, 3]
        """
        response = self._make_request("GET", f"/v2/models/{model_name}")

        if response.status_code == 200:
            metadata = response.json()
            # versions is an array of strings: ["2", "3"]
            versions = [int(v) for v in metadata.get('versions', [])]
            return sorted(versions)
        elif response.status_code == 400:
            # Model does not exist or is not loaded
            return []
        else:
            raise Exception(
                f"Failed to get model metadata: {response.status_code} - {response.text}"
            )

    def load_model(self, model_name: str, version: Optional[int] = None) -> None:
        """
        Loads a model into Triton memory.

        In EXPLICIT mode we must load models explicitly. If no version
        is specified, Triton loads the latest available version from the
        repository. If a version is specified, exactly that version is
        loaded.

        Args:
            model_name: Name of the model
            version: Optional - specific version to load

        Raises:
            Exception: If load fails

        Example:
            >>> client.load_model("resnet18_imagenette")  # Loads latest
            >>> client.load_model("resnet18_imagenette", version=3)  # Loads version 3
        """
        endpoint = f"/v2/repository/models/{model_name}/load"
        params = {}

        if version is not None:
            params['version'] = str(version)
            print(f"Loading {model_name} version {version}...")
        else:
            print(f"Loading latest version of {model_name}...")

        response = self._make_request("POST", endpoint, params=params)

        if response.status_code == 200:
            version_str = f"version {version}" if version else "latest version"
            print(f"Load request successful for {model_name} {version_str}")
        else:
            error_msg = f"Load failed: {response.status_code} - {response.text}"
            raise Exception(error_msg)

    def unload_model(self, model_name: str, version: int) -> None:
        """
        Unloads a specific version of a model from memory.

        Important: Unlike load, for unload we must always specify a
        specific version. Triton cannot guess which version we want
        to unload.

        Args:
            model_name: Name of the model
            version: Version number to unload

        Raises:
            Exception: If unload fails
        """
        endpoint = f"/v2/repository/models/{model_name}/unload"
        params = {'version': str(version)}

        print(f"Unloading {model_name} version {version}...")

        response = self._make_request("POST", endpoint, params=params)

        if response.status_code == 200:
            print(f"Unload successful for {model_name} version {version}")
        else:
            # Warning but not fatal - version might already be unloaded
            print(f"Warning: Unload request returned {response.status_code} - {response.text}")

    def is_model_ready(self, model_name: str, version: Optional[int] = None) -> bool:
        """
        Checks whether a model (or a specific version) is ready.

        A model is ready when it is fully loaded and can accept inference
        requests. We use this after a load to wait until the model is
        actually available.

        Args:
            model_name: Name of the model
            version: Optional - specific version to check

        Returns:
            True if the model is ready, False otherwise
        """
        endpoint = f"/v2/models/{model_name}"
        if version is not None:
            endpoint += f"/versions/{version}"
        endpoint += "/ready"

        try:
            response = self._make_request("GET", endpoint)
            return response.status_code == 200
        except Exception as e:
            print(f"Ready check failed: {e}")
            return False

    def wait_until_ready(
        self,
        model_name: str,
        version: Optional[int] = None,
        timeout: int = 60,
        poll_interval: int = 2
    ) -> None:
        """
        Waits until a model is ready, with timeout.

        Polls the ready API at regular intervals until the model is
        ready or the timeout is reached. This is essential after a
        load - we do not want to proceed until the model is truly
        available.

        Args:
            model_name: Name of the model
            version: Optional - specific version to wait for
            timeout: Maximum wait time in seconds (default: 60)
            poll_interval: How often to check in seconds (default: 2)

        Raises:
            Exception: If model does not become ready within the timeout

        Example:
            >>> client.load_model("resnet18_imagenette")
            >>> client.wait_until_ready("resnet18_imagenette", timeout=30)
        """
        version_str = f"version {version}" if version else "latest version"
        print(f"Waiting for {model_name} {version_str} to become ready (timeout: {timeout}s)...")

        elapsed = 0
        while elapsed < timeout:
            if self.is_model_ready(model_name, version):
                print(f"{model_name} is ready after {elapsed}s")
                return

            time.sleep(poll_interval)
            elapsed += poll_interval

            # Progress indicator for long waits
            if elapsed % 10 == 0:
                print(f"Still waiting... ({elapsed}s elapsed)")

        # Timeout reached
        raise Exception(
            f"{model_name} did not become ready within {timeout} seconds"
        )

    def infer(
        self,
        model_name: str,
        inputs: List[Dict[str, Any]],
        outputs: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Makes an inference request against Triton.

        Uses the KServe v2 Inference Protocol API. This is useful for
        the verification step where we want to test if inference works.

        Args:
            model_name: Name of the model
            inputs: List of input tensors in KServe format:
                    [{"name": "input", "shape": [1,3,224,224],
                      "datatype": "FP32", "data": [...]}]
            outputs: Optional - specific outputs to request

        Returns:
            Response dictionary with outputs

        Raises:
            Exception: If inference fails

        Example:
            >>> inputs = [{
            ...     "name": "input",
            ...     "shape": [1, 3, 224, 224],
            ...     "datatype": "FP32",
            ...     "data": [0.1, 0.2, ...]  # 150528 values
            ... }]
            >>> result = client.infer("resnet18_imagenette", inputs)
            >>> predictions = result["outputs"][0]["data"]
        """
        endpoint = f"/v2/models/{model_name}/infer"

        payload = {"inputs": inputs}
        if outputs:
            payload["outputs"] = outputs

        response = self._make_request("POST", endpoint, json=payload)

        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(
                f"Inference failed: {response.status_code} - {response.text}"
            )

    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """
        Retrieves the model configuration from Triton.

        Returns the config.pbtxt contents as JSON. Useful to validate
        that the config is correct or to read input/output specs
        programmatically.

        Args:
            model_name: Name of the model

        Returns:
            Dictionary with model config

        Raises:
            Exception: If config cannot be retrieved
        """
        response = self._make_request("GET", f"/v2/models/{model_name}/config")

        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(
                f"Failed to get model config: {response.status_code} - {response.text}"
            )
