"""
vLLM OpenAI-Compatible API Client.

Simple HTTP client for interacting with a vLLM server that exposes
the OpenAI-compatible API. Used by deployment steps for health checks,
model listing, and smoke test inference.
"""

import requests
import time
from typing import Dict, Any, List, Optional


class VllmClient:
    """Client for vLLM OpenAI-compatible API."""

    def __init__(self, base_url: str = "http://vllm-service.ml-models:8000"):
        """
        Initializes the vLLM client.

        Args:
            base_url: Base URL of the vLLM server (e.g. http://vllm-service.ml-models:8000)
        """
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()

    def _request(self, method: str, endpoint: str, **kwargs) -> requests.Response:
        """Makes an HTTP request with error handling."""
        url = f"{self.base_url}{endpoint}"
        try:
            response = self.session.request(method, url, timeout=120, **kwargs)
            return response
        except requests.exceptions.ConnectionError as e:
            raise Exception(f"Cannot connect to vLLM at {self.base_url}: {e}")
        except requests.exceptions.Timeout as e:
            raise Exception(f"Request to vLLM timed out: {e}")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Request to vLLM failed: {e}")

    def health_check(self) -> bool:
        """
        Checks whether the vLLM server is healthy.

        Returns:
            True if the server is healthy, False otherwise
        """
        try:
            response = self._request("GET", "/health")
            return response.status_code == 200
        except Exception:
            return False

    def list_models(self) -> List[str]:
        """
        Lists all available models (base model + loaded LoRA adapters).

        Returns:
            List of model IDs
        """
        response = self._request("GET", "/v1/models")
        response.raise_for_status()

        data = response.json()
        return [model["id"] for model in data.get("data", [])]

    def completion(
        self,
        model: str,
        prompt: str,
        max_tokens: int = 64,
        temperature: float = 0.1,
    ) -> Dict[str, Any]:
        """
        Sends a text completion request.

        Uses /v1/completions (not /v1/chat/completions) which works
        with base models that don't have a chat template.

        Args:
            model: Model ID (base model name or LoRA adapter name)
            prompt: Text prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Full API response as dict
        """
        payload = {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        response = self._request("POST", "/v1/completions", json=payload)
        response.raise_for_status()

        return response.json()

    def chat_completion(
        self,
        model: str,
        messages: list,
        max_tokens: int = 128,
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Sends a chat completion request.

        Uses /v1/chat/completions for instruct/chat models (e.g. Llama 3.1 Instruct).

        Args:
            model: Model ID
            messages: List of message dicts with 'role' and 'content'
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature

        Returns:
            Full API response as dict
        """
        payload = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }

        response = self._request("POST", "/v1/chat/completions", json=payload)
        response.raise_for_status()

        return response.json()

    def load_lora_adapter(self, lora_name: str, lora_path: str) -> None:
        """
        Loads a LoRA adapter at runtime via the vLLM API.

        Requires VLLM_ALLOW_RUNTIME_LORA_UPDATING=True on the server.

        Args:
            lora_name: Adapter name (e.g. "aws-rag-qa-candidate")
            lora_path: Path to adapter files on the vLLM server (e.g. "/mnt/adapters/aws-rag-qa-candidate")
        """
        payload = {
            "lora_name": lora_name,
            "lora_path": lora_path,
        }

        print(f"Loading LoRA adapter '{lora_name}' from {lora_path}...")
        response = self._request("POST", "/v1/load_lora_adapter", json=payload)
        response.raise_for_status()
        print(f"LoRA adapter '{lora_name}' loaded successfully")

    def unload_lora_adapter(self, lora_name: str) -> None:
        """
        Unloads a LoRA adapter at runtime via the vLLM API.

        Requires VLLM_ALLOW_RUNTIME_LORA_UPDATING=True on the server.

        Args:
            lora_name: Adapter name to unload (e.g. "aws-rag-qa-candidate")
        """
        payload = {
            "lora_name": lora_name,
        }

        print(f"Unloading LoRA adapter '{lora_name}'...")
        response = self._request("POST", "/v1/unload_lora_adapter", json=payload)
        response.raise_for_status()
        print(f"LoRA adapter '{lora_name}' unloaded successfully")

    def wait_until_healthy(self, timeout: int = 300, poll_interval: int = 10) -> None:
        """
        Waits until the vLLM server is healthy.

        Args:
            timeout: Maximum wait time in seconds
            poll_interval: How often to check in seconds

        Raises:
            Exception if server does not become healthy within timeout
        """
        print(f"Waiting for vLLM to become healthy (timeout: {timeout}s)...")

        elapsed = 0
        while elapsed < timeout:
            if self.health_check():
                print(f"vLLM is healthy after {elapsed}s")
                return

            time.sleep(poll_interval)
            elapsed += poll_interval

            if elapsed % 30 == 0:
                print(f"  Still waiting... ({elapsed}s elapsed)")

        raise Exception(f"vLLM did not become healthy within {timeout} seconds")

    def wait_until_model_available(
        self, model_name: str, timeout: int = 300, poll_interval: int = 10
    ) -> None:
        """
        Waits until a specific model (adapter) is listed by vLLM.

        Args:
            model_name: The adapter name to wait for (e.g. "aws-rag-qa-candidate")
            timeout: Maximum wait time in seconds
            poll_interval: How often to check in seconds

        Raises:
            Exception if model does not appear within timeout
        """
        print(f"Waiting for model '{model_name}' to be available (timeout: {timeout}s)...")

        elapsed = 0
        while elapsed < timeout:
            try:
                models = self.list_models()
                if model_name in models:
                    print(f"Model '{model_name}' is available after {elapsed}s")
                    return
            except Exception:
                pass

            time.sleep(poll_interval)
            elapsed += poll_interval

            if elapsed % 30 == 0:
                print(f"  Still waiting... ({elapsed}s elapsed)")

        raise Exception(
            f"Model '{model_name}' did not become available within {timeout} seconds"
        )
