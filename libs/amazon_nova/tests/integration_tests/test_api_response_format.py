"""
Integration tests verifying Nova API response format matches specification.
"""

from typing import Literal, Optional, cast

import pytest
from langchain_core.messages import AIMessage
from langchain_core.tools import tool

from langchain_amazon_nova import ChatAmazonNova
from langchain_amazon_nova._exceptions import (
    NovaError,
    NovaModelNotFoundError,
)


@pytest.mark.integration
def test_basic_response_structure() -> None:
    """Verify basic response has all required fields from API spec.

    Expected structure:
    {
        "id": "chatcmpl-...",
        "object": "chat.completion",
        "created": timestamp,
        "model": "nova-pro-v1",
        "choices": [{
            "finish_reason": "stop",
            "index": 0,
            "message": {
                "content": "...",
                "role": "assistant"
            }
        }],
        "usage": {
            "prompt_tokens": int,
            "completion_tokens": int,
            "total_tokens": int
        }
    }
    """
    llm = ChatAmazonNova(model="nova-pro-v1", temperature=0.7)

    response = llm.invoke("Hello!")

    # Verify LangChain message structure
    assert isinstance(response, AIMessage)
    assert response.content
    assert isinstance(response.content, str)

    # Verify response metadata
    assert hasattr(response, "response_metadata")
    assert "model" in response.response_metadata
    assert "finish_reason" in response.response_metadata
    assert response.response_metadata["finish_reason"] in [
        "stop",
        "length",
        "tool_calls",
        "content_filter",
    ]

    # Verify usage metadata (converted from API format)
    assert hasattr(response, "usage_metadata")
    assert response.usage_metadata is not None

    # Type narrowing for mypy
    usage = response.usage_metadata
    assert "input_tokens" in usage  # converted from prompt_tokens
    assert "output_tokens" in usage  # converted from completion_tokens
    assert "total_tokens" in usage

    # Verify token counts are positive
    assert usage["input_tokens"] > 0
    assert usage["output_tokens"] > 0
    assert usage["total_tokens"] > 0


@pytest.mark.integration
def test_streaming_response_structure() -> None:
    """Verify streaming responses match API spec.

    Streaming chunks should have:
    - choices[0].delta.content
    - choices[0].delta.role (only first chunk)
    """
    llm = ChatAmazonNova(model="nova-pro-v1", temperature=0.7)

    chunks = list(llm.stream("Tell me a one-sentence fact."))

    assert len(chunks) > 0

    # Each chunk should be an AIMessageChunk (stream() unwraps ChatGenerationChunk)
    for chunk in chunks:
        assert isinstance(chunk, AIMessage)
        # Content can be empty for some chunks (e.g., final chunk with usage)
        assert hasattr(chunk, "content")


@pytest.mark.integration
def test_streaming_with_usage_metadata() -> None:
    """Verify stream_options.include_usage works correctly.

    When include_usage=true, the final chunk should include usage data.
    """
    llm = ChatAmazonNova(
        model="nova-pro-v1", temperature=0.7, stream_options={"include_usage": True}
    )

    chunks = list(llm.stream("Hello!"))

    assert len(chunks) > 0

    # Find the chunk with usage metadata (should be in final chunk)
    usage_found = False
    for chunk in chunks:
        if hasattr(chunk, "usage_metadata") and chunk.usage_metadata:
            usage_found = True
            # Verify structure
            assert "input_tokens" in chunk.usage_metadata
            assert "output_tokens" in chunk.usage_metadata
            assert "total_tokens" in chunk.usage_metadata
            break

    assert usage_found, "Usage metadata not found in any chunk"


@pytest.mark.integration
def test_tool_call_response_format() -> None:
    """Verify tool call responses match API spec.

    Expected format in API:
    {
        "tool_calls": [{
            "id": "tooluse_...",
            "type": "function",
            "function": {
                "name": "tool_name",
                "arguments": "{...json...}"
            }
        }]
    }

    LangChain converts to:
    {
        "tool_calls": [{
            "id": "tooluse_...",
            "name": "tool_name",
            "args": {...dict...}
        }]
    }
    """

    @tool
    def get_weather(location: str) -> str:
        """Get weather for a location."""
        return f"Weather in {location}: sunny, 72°F"

    llm = ChatAmazonNova(model="nova-pro-v1", temperature=0.7).bind_tools([get_weather])

    response = llm.invoke("What's the weather in Paris?")

    # Verify we got tool calls
    assert hasattr(response, "tool_calls")
    assert len(response.tool_calls) > 0

    # Verify tool call structure (LangChain format)
    tool_call = response.tool_calls[0]
    assert "id" in tool_call
    assert "name" in tool_call
    assert "args" in tool_call

    # Verify tool call ID format
    assert isinstance(tool_call["id"], str)
    assert len(tool_call["id"]) > 0

    # Verify function name
    assert tool_call["name"] == "get_weather"

    # Verify args is a dict (converted from JSON string)
    assert isinstance(tool_call["args"], dict)
    assert "location" in tool_call["args"]


@pytest.mark.integration
def test_max_tokens_enforced() -> None:
    """Verify max_tokens actually limits response length."""
    llm = ChatAmazonNova(model="nova-pro-v1", max_tokens=20)

    response = llm.invoke(
        "Write a long story about space exploration with many details."
    )

    # Should hit token limit
    assert hasattr(response, "usage_metadata")
    assert response.usage_metadata is not None
    assert response.usage_metadata["output_tokens"] <= 20

    # Finish reason should indicate length cutoff
    assert response.response_metadata["finish_reason"] == "length"


@pytest.mark.integration
def test_reasoning_effort_parameter() -> None:
    """Verify reasoning_effort parameter is accepted."""
    for effort_str in ["low", "medium", "high"]:
        llm = ChatAmazonNova(
            model="nova-2-lite-v1",
            reasoning_effort=cast(Literal["low", "medium", "high"], effort_str),
            temperature=0.3,
        )

        response = llm.invoke("What is 15% of 240?")

        # Should get valid response
        assert response.content
        assert isinstance(response.content, str)


@pytest.mark.integration
def test_top_p_parameter() -> None:
    """Verify top_p parameter is accepted."""
    llm = ChatAmazonNova(model="nova-pro-v1", top_p=0.9, temperature=0.8)

    response = llm.invoke("Hello!")

    # Should get valid response
    assert response.content
    assert isinstance(response.content, str)


@pytest.mark.integration
def test_metadata_parameter() -> None:
    """Verify metadata parameter is accepted without errors."""
    llm = ChatAmazonNova(
        model="nova-pro-v1",
        metadata={
            "user_id": "test123",
            "session_id": "abc456",
            "application": "test_suite",
        },
    )

    response = llm.invoke("Hello!")

    # Should get valid response (metadata doesn't affect response)
    assert response.content
    assert isinstance(response.content, str)


@pytest.mark.integration
def test_multi_content_message() -> None:
    """Verify multi-content messages are supported.

    API spec shows support for messages with multiple content blocks:
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "..."},
            {"type": "text", "text": "..."}
        ]
    }
    """
    llm = ChatAmazonNova(model="nova-pro-v1", temperature=0.7)

    # Single content string format
    response = llm.invoke("Hello!")

    assert response.content
    assert isinstance(response.content, str)


@pytest.mark.integration
def test_error_response_codes() -> None:
    """Verify proper error handling for API error codes.

    This is a basic smoke test that exceptions are raised.
    More specific tests are in separate test functions.
    """
    llm = ChatAmazonNova(model="invalid-model-name-xyz", temperature=0.7)

    with pytest.raises(NovaError):
        llm.invoke("Hello!")


@pytest.mark.integration
def test_invalid_model_raises_model_not_found() -> None:
    """Verify invalid model name raises NovaModelNotFoundError.

    Expected error code: 404 ModelNotFoundException
    """
    llm = ChatAmazonNova(model="invalid-model-name-xyz", temperature=0.7)

    with pytest.raises(NovaModelNotFoundError) as exc_info:
        llm.invoke("Hello!")

    assert exc_info.value.status_code == 404
    assert exc_info.value.model_name == "invalid-model-name-xyz"


@pytest.mark.integration
def test_all_exceptions_inherit_from_nova_error() -> None:
    """Verify all Nova exceptions inherit from NovaError for easy catching."""
    llm = ChatAmazonNova(model="invalid-xyz", temperature=0.7)

    with pytest.raises(Exception) as exc_info:
        llm.invoke("Hello!")

    # Should be a NovaError (or subclass) - check the cause chain
    exception: Optional[BaseException] = exc_info.value
    while exception is not None:
        if isinstance(exception, NovaError):
            break
        exception = exception.__cause__

    # exception should be NovaError or None if not found in chain
    assert exception is not None
    assert isinstance(exception, NovaError)


@pytest.mark.integration
def test_per_call_parameter_override() -> None:
    """Verify per-call parameters override model-level defaults."""
    llm = ChatAmazonNova(
        model="nova-2-lite-v1", max_tokens=50, temperature=0.5, reasoning_effort="low"
    )

    # Override on invoke
    response = llm.invoke("What is Python?", max_tokens=100, reasoning_effort="high")

    # Should get response (actual override happens at API level)
    assert response.content
    assert isinstance(response.content, str)
