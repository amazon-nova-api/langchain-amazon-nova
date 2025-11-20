"""Exceptions for Amazon Nova integration.

Based on Nova API specification error codes:
- 400: ValidationException (invalid parameters)
- 404: ModelNotFoundException (model not found/available)
- 429: ThrottlingException (rate limit exceeded)
- 500: ModelException (internal model error)
"""

from typing import Any, Dict, Optional


class NovaError(Exception):
    """Base exception for all Nova-related errors."""

    def __init__(
        self,
        message: str,
        status_code: Optional[int] = None,
        response: Optional[Dict[str, Any]] = None,
    ):
        """Initialize Nova error.

        Args:
            message: Error message
            status_code: HTTP status code if available
            response: Full response data if available
        """
        super().__init__(message)
        self.status_code = status_code
        self.response = response


class NovaValidationError(NovaError):
    """Exception raised for invalid request parameters (HTTP 400).

    This includes:
    - Invalid model parameters (temperature, top_p, etc.)
    - Malformed requests
    - Invalid tool definitions
    - Invalid message formats
    """

    def __init__(
        self,
        message: str,
        response: Optional[Dict[str, Any]] = None,
    ):
        """Initialize validation error.

        Args:
            message: Error message
            response: Full response data if available
        """
        super().__init__(message, status_code=400, response=response)


class NovaModelNotFoundError(NovaError):
    """Exception raised when model is not found or not available (HTTP 404).

    This occurs when:
    - Model name is invalid
    - Model is not available in the region
    - Model access has not been granted
    """

    def __init__(
        self,
        model_name: str,
        message: Optional[str] = None,
        response: Optional[Dict[str, Any]] = None,
    ):
        """Initialize model not found error.

        Args:
            model_name: Name of the model that was not found
            message: Optional custom error message
            response: Full response data if available
        """
        if message is None:
            message = f"Model '{model_name}' not found or not available"
        super().__init__(message, status_code=404, response=response)
        self.model_name = model_name


class NovaThrottlingError(NovaError):
    """Exception raised when rate limit is exceeded (HTTP 429).

    This occurs when:
    - Too many requests in a time window
    - Token rate limit exceeded
    - Request quota exceeded
    """

    def __init__(
        self,
        message: str,
        retry_after: Optional[int] = None,
        response: Optional[Dict[str, Any]] = None,
    ):
        """Initialize throttling error.

        Args:
            message: Error message
            retry_after: Seconds to wait before retrying if available
            response: Full response data if available
        """
        super().__init__(message, status_code=429, response=response)
        self.retry_after = retry_after


class NovaModelError(NovaError):
    """Exception raised for internal model errors (HTTP 500).

    This occurs when:
    - Model encounters an internal error
    - Service is temporarily unavailable
    - Unexpected model behavior
    """

    def __init__(
        self,
        message: str,
        response: Optional[Dict[str, Any]] = None,
    ):
        """Initialize model error.

        Args:
            message: Error message
            response: Full response data if available
        """
        super().__init__(message, status_code=500, response=response)


class NovaToolCallError(NovaError):
    """Exception raised for tool calling errors.

    This occurs when:
    - Model doesn't support tool calling
    - Tool definition is invalid
    - Tool response format is incorrect
    """

    def __init__(
        self,
        message: str,
        model_name: Optional[str] = None,
    ):
        """Initialize tool call error.

        Args:
            message: Error message
            model_name: Model name if relevant
        """
        super().__init__(message)
        self.model_name = model_name


class NovaConfigurationError(NovaError):
    """Exception raised for configuration errors.

    This occurs when:
    - API key is missing or invalid
    - Base URL is invalid
    - Environment variables are not set properly
    """

    def __init__(self, message: str):
        """Initialize configuration error.

        Args:
            message: Error message
        """
        super().__init__(message)


def map_http_error_to_nova_exception(
    error: Exception,
    model_name: Optional[str] = None,
) -> NovaError:
    """Map HTTP/OpenAI errors to Nova-specific exceptions.

    Args:
        error: Original exception from OpenAI SDK or HTTP client
        model_name: Model name for context

    Returns:
        Appropriate NovaError subclass
    """
    # Try to extract status code from various error types
    status_code = None
    message = str(error)
    response = None

    # Handle OpenAI SDK errors
    if hasattr(error, "status_code"):
        status_code = error.status_code
    if hasattr(error, "response"):
        response = getattr(error, "response", None)
    if hasattr(error, "message"):
        message = error.message

    # Map by status code
    if status_code == 400:
        return NovaValidationError(message, response=response)
    elif status_code == 404:
        return NovaModelNotFoundError(
            model_name=model_name or "unknown",
            message=message,
            response=response,
        )
    elif status_code == 429:
        retry_after = None
        if response and isinstance(response, dict):
            retry_after = response.get("retry_after")
        return NovaThrottlingError(message, retry_after=retry_after, response=response)
    elif status_code and status_code >= 500:
        return NovaModelError(message, response=response)

    # Default to base NovaError
    return NovaError(message, status_code=status_code, response=response)


__all__ = [
    "NovaError",
    "NovaValidationError",
    "NovaModelNotFoundError",
    "NovaThrottlingError",
    "NovaModelError",
    "NovaToolCallError",
    "NovaConfigurationError",
    "map_http_error_to_nova_exception",
]
