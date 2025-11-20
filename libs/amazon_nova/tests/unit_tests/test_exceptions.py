"""Unit tests for Nova exception handling."""

import pytest

from langchain_nova._exceptions import (
    NovaConfigurationError,
    NovaError,
    NovaModelError,
    NovaModelNotFoundError,
    NovaThrottlingError,
    NovaToolCallError,
    NovaValidationError,
    map_http_error_to_nova_exception,
)


def test_base_nova_error():
    """Test base NovaError exception."""
    error = NovaError("test message", status_code=500, response={"error": "test"})

    assert str(error) == "test message"
    assert error.status_code == 500
    assert error.response == {"error": "test"}


def test_validation_error():
    """Test NovaValidationError (HTTP 400)."""
    error = NovaValidationError("Invalid temperature", response={"error": "bad param"})

    assert str(error) == "Invalid temperature"
    assert error.status_code == 400
    assert error.response == {"error": "bad param"}


def test_model_not_found_error():
    """Test NovaModelNotFoundError (HTTP 404)."""
    # With custom message
    error = NovaModelNotFoundError("invalid-model", message="Custom message")
    assert str(error) == "Custom message"
    assert error.status_code == 404
    assert error.model_name == "invalid-model"

    # With default message
    error = NovaModelNotFoundError("nova-xyz")
    assert "nova-xyz" in str(error)
    assert "not found" in str(error).lower()


def test_throttling_error():
    """Test NovaThrottlingError (HTTP 429)."""
    error = NovaThrottlingError(
        "Rate limit exceeded", retry_after=60, response={"error": "throttled"}
    )

    assert str(error) == "Rate limit exceeded"
    assert error.status_code == 429
    assert error.retry_after == 60
    assert error.response == {"error": "throttled"}


def test_model_error():
    """Test NovaModelError (HTTP 500)."""
    error = NovaModelError("Internal server error", response={"error": "server"})

    assert str(error) == "Internal server error"
    assert error.status_code == 500
    assert error.response == {"error": "server"}


def test_tool_call_error():
    """Test NovaToolCallError."""
    error = NovaToolCallError("Tool not supported", model_name="nova-lite-v1")

    assert str(error) == "Tool not supported"
    assert error.model_name == "nova-lite-v1"


def test_configuration_error():
    """Test NovaConfigurationError."""
    error = NovaConfigurationError("API key not set")

    assert str(error) == "API key not set"


class MockHTTPError(Exception):
    """Mock HTTP error for testing."""

    def __init__(self, status_code, message):
        super().__init__(message)
        self.status_code = status_code
        self.message = message
        self.response = {"error": message}


def test_map_http_error_400():
    """Test mapping HTTP 400 to NovaValidationError."""
    error = MockHTTPError(400, "Bad request")

    nova_error = map_http_error_to_nova_exception(error, model_name="nova-pro-v1")

    assert isinstance(nova_error, NovaValidationError)
    assert nova_error.status_code == 400
    assert "Bad request" in str(nova_error)


def test_map_http_error_404():
    """Test mapping HTTP 404 to NovaModelNotFoundError."""
    error = MockHTTPError(404, "Model not found")

    nova_error = map_http_error_to_nova_exception(error, model_name="invalid-model")

    assert isinstance(nova_error, NovaModelNotFoundError)
    assert nova_error.status_code == 404
    assert nova_error.model_name == "invalid-model"


def test_map_http_error_429():
    """Test mapping HTTP 429 to NovaThrottlingError."""
    error = MockHTTPError(429, "Too many requests")

    nova_error = map_http_error_to_nova_exception(error)

    assert isinstance(nova_error, NovaThrottlingError)
    assert nova_error.status_code == 429


def test_map_http_error_500():
    """Test mapping HTTP 500 to NovaModelError."""
    error = MockHTTPError(500, "Internal server error")

    nova_error = map_http_error_to_nova_exception(error)

    assert isinstance(nova_error, NovaModelError)
    assert nova_error.status_code == 500


def test_map_http_error_unknown():
    """Test mapping unknown error to base NovaError."""
    error = Exception("Unknown error")

    nova_error = map_http_error_to_nova_exception(error)

    assert isinstance(nova_error, NovaError)
    assert "Unknown error" in str(nova_error)


def test_map_http_error_no_status_code():
    """Test mapping error without status code."""
    error = Exception("Generic error")

    nova_error = map_http_error_to_nova_exception(error, model_name="nova-pro-v1")

    assert isinstance(nova_error, NovaError)
    assert nova_error.status_code is None
    assert "Generic error" in str(nova_error)


def test_exception_inheritance():
    """Test that all Nova exceptions inherit from NovaError."""
    assert issubclass(NovaValidationError, NovaError)
    assert issubclass(NovaModelNotFoundError, NovaError)
    assert issubclass(NovaThrottlingError, NovaError)
    assert issubclass(NovaModelError, NovaError)
    assert issubclass(NovaToolCallError, NovaError)
    assert issubclass(NovaConfigurationError, NovaError)


def test_exception_can_be_caught_as_base():
    """Test that specific exceptions can be caught as base NovaError."""
    try:
        raise NovaValidationError("test")
    except NovaError as e:
        assert isinstance(e, NovaValidationError)
        assert isinstance(e, NovaError)


def test_exception_preserves_original_error():
    """Test that mapped exceptions preserve original error info."""
    original = MockHTTPError(400, "Original error message")

    try:
        raise map_http_error_to_nova_exception(original, model_name="nova-pro-v1")
    except NovaError as e:
        assert "Original error message" in str(e)
        assert e.response == {"error": "Original error message"}
