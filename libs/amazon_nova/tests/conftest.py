"""Shared test configuration and fixtures."""

import os
from typing import List

import httpx
import pytest


def fetch_available_models() -> List[str]:
    """Fetch available Nova models from the API.

    Returns:
        List of model IDs available at the Nova endpoint.
    """
    api_key = os.getenv("NOVA_API_KEY")
    base_url = os.getenv("NOVA_BASE_URL", "https://api.nova.amazon.com/v1")

    if not api_key:
        pytest.skip("NOVA_API_KEY not set")

    try:
        # Create client with no compression to avoid zstd issues
        with httpx.Client(headers={"Accept-Encoding": "identity"}) as client:
            response = client.get(
                f"{base_url}/models",
                headers={"Authorization": f"Bearer {api_key}"},
                timeout=10.0,
            )
        response.raise_for_status()
        data = response.json()

        if "data" in data:
            models = [model["id"] for model in data["data"]]
            # Filter out models with spaces or non-URL-safe characters
            url_safe_models = [m for m in models if " " not in m]
            return url_safe_models
        else:
            pytest.fail(f"Unexpected response format: {data}")

    except Exception as e:
        pytest.fail(f"Failed to fetch models: {e}")


@pytest.fixture(scope="session")
def available_models() -> List[str]:
    """Fixture providing list of available Nova models."""
    return fetch_available_models()


@pytest.fixture(scope="session")
def single_test_model() -> str:
    """Fixture providing a single model for quick tests.

    Returns the first available model from the API.
    """
    models = fetch_available_models()
    if not models:
        pytest.fail("No models available")
    return models[0]
