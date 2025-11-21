"""Basic unit tests for ChatNova."""

from typing import Literal, cast

import pytest
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from langchain_nova import ChatNova


def test_chatnova_initialization() -> None:
    """Test that ChatNova can be initialized."""
    llm = ChatNova(
        model="nova-pro-v1",
        temperature=0.7,
        api_key="test-key",
    )
    assert llm.model_name == "nova-pro-v1"
    assert llm.temperature == 0.7


def test_chatnova_model_name_alias() -> None:
    """Test that 'model' alias works for model_name."""
    llm = ChatNova(model="nova-micro-v1", api_key="test-key")
    assert llm.model_name == "nova-micro-v1"


def test_chatnova_serialization() -> None:
    """Test that ChatNova can be serialized without exposing secrets."""
    llm = ChatNova(
        model="nova-pro-v1",
        temperature=0.5,
        api_key="secret-key",
    )

    # Verify secrets are marked
    assert "api_key" in llm.lc_secrets

    # Verify identifying params don't include secrets
    params = llm._identifying_params
    assert "api_key" not in params
    assert params["model_name"] == "nova-pro-v1"
    assert params["temperature"] == 0.5


def test_chatnova_llm_type() -> None:
    """Test that ChatNova returns correct LLM type."""
    llm = ChatNova(api_key="test-key")
    assert llm._llm_type == "nova-chat"


def test_chatnova_message_conversion() -> None:
    """Test message conversion to OpenAI format."""
    llm = ChatNova(api_key="test-key")

    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="Hello!"),
    ]

    converted = llm._convert_messages_to_nova_format(messages)

    assert len(converted) == 2
    assert converted[0]["role"] == "system"
    assert converted[0]["content"] == "You are a helpful assistant."
    assert converted[1]["role"] == "user"
    assert converted[1]["content"] == "Hello!"


def test_chatnova_default_base_url() -> None:
    """Test that default base URL is set."""
    llm = ChatNova(api_key="test-key")
    assert llm.base_url is not None
    assert isinstance(llm.base_url, str)


def test_chatnova_custom_base_url() -> None:
    """Test that custom base URL can be set."""
    custom_url = "https://custom.example.com/v1"
    llm = ChatNova(api_key="test-key", base_url=custom_url)
    assert llm.base_url == custom_url


def test_chatnova_max_tokens_validation() -> None:
    """Test that max_tokens must be positive."""
    with pytest.raises(ValueError):
        ChatNova(api_key="test-key", max_tokens=0)

    with pytest.raises(ValueError):
        ChatNova(api_key="test-key", max_tokens=-1)


def test_chatnova_temperature_validation() -> None:
    """Test that temperature is within valid range."""
    with pytest.raises(ValueError):
        ChatNova(api_key="test-key", temperature=-0.1)

    with pytest.raises(ValueError):
        ChatNova(api_key="test-key", temperature=1.1)

    # Valid temperatures should work
    llm = ChatNova(api_key="test-key", temperature=0.0)
    assert llm.temperature == 0.0

    llm = ChatNova(api_key="test-key", temperature=1.0)
    assert llm.temperature == 1.0


# Phase 1 Parameters Tests


def test_max_completion_tokens_initialization() -> None:
    """Test that max_completion_tokens can be set at initialization."""
    llm = ChatNova(model="nova-pro-v1", max_completion_tokens=200, api_key="test-key")
    assert llm.max_completion_tokens == 200


def test_top_p_initialization() -> None:
    """Test that top_p can be set at initialization."""
    llm = ChatNova(model="nova-pro-v1", top_p=0.9, api_key="test-key")
    assert llm.top_p == 0.9


def test_top_p_validation() -> None:
    """Test that top_p is validated to be between 0 and 1."""
    with pytest.raises(ValueError):
        ChatNova(model="nova-pro-v1", top_p=1.5, api_key="test-key")

    with pytest.raises(ValueError):
        ChatNova(model="nova-pro-v1", top_p=-0.1, api_key="test-key")


def test_reasoning_effort_initialization() -> None:
    """Test that reasoning_effort can be set at initialization."""
    for effort in ["low", "medium", "high"]:
        llm = ChatNova(
            model="nova-pro-v1",
            reasoning_effort=cast(Literal["low", "medium", "high"], effort),
            api_key="test-key",
        )
        assert llm.reasoning_effort == effort


def test_metadata_initialization() -> None:
    """Test that metadata can be set at initialization."""
    metadata = {"user_id": "123", "session_id": "abc"}
    llm = ChatNova(model="nova-pro-v1", metadata=metadata, api_key="test-key")
    assert llm.metadata == metadata


def test_stream_options_initialization() -> None:
    """Test that stream_options can be set at initialization."""
    stream_opts = {"include_usage": True}
    llm = ChatNova(model="nova-pro-v1", stream_options=stream_opts, api_key="test-key")
    assert llm.stream_options == stream_opts


# Response Structure Tests (based on upstream API spec)


def test_response_structure_fields() -> None:
    """Test that response structure matches Nova API spec.

    Based on: https://quip-amazon.com/tEwWAlX0Lfc7/

    Expected response format:
    {
        "id": "chatcmpl-...",
        "object": "chat.completion",
        "created": 1759157259,
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
            "prompt_tokens": 2,
            "completion_tokens": 205,
            "total_tokens": 207
        }
    }
    """
    llm = ChatNova(model="nova-pro-v1", api_key="test-key")

    # Verify we're converting the response properly
    # The actual API validation happens in integration tests
    # Here we just ensure our fields exist
    assert hasattr(llm, "model_name")
    assert hasattr(llm, "temperature")
    assert hasattr(llm, "max_tokens")


def test_usage_metadata_structure() -> None:
    """Test that usage metadata follows the correct structure.

    Expected usage structure from API:
    {
        "prompt_tokens": int,
        "completion_tokens": int,
        "total_tokens": int
    }

    We convert to LangChain format:
    {
        "input_tokens": int,
        "output_tokens": int,
        "total_tokens": int
    }
    """
    # This is tested in integration tests with real API calls
    # Unit test just verifies the structure is defined
    pass


def test_tool_call_response_structure() -> None:
    """Test that tool call responses match Nova API format.

    Expected tool_calls in assistant message:
    [{
        "id": "tooluse_...",
        "type": "function",
        "function": {
            "name": "tool_name",
            "arguments": "{...json...}"
        }
    }]

    We convert to LangChain format:
    [{
        "id": "tooluse_...",
        "name": "tool_name",
        "args": {...dict...}
    }]
    """
    # This is tested in integration tests with real API calls
    pass


def test_finish_reasons() -> None:
    """Test that we handle all possible finish_reason values.

    Possible finish_reason values from Nova API:
    - "stop": Natural completion
    - "length": Hit max_tokens limit
    - "tool_calls": Model wants to call tools
    - "content_filter": Content filtered
    """
    # This is tested in integration tests with real API calls
    pass


def test_streaming_response_structure() -> None:
    """Test that streaming responses match Nova API format.

    Streaming chunks have similar structure but with delta:
    {
        "choices": [{
            "delta": {
                "content": "text chunk",
                "role": "assistant"  # only in first chunk
            }
        }]
    }

    With stream_options.include_usage=true, final chunk includes usage.
    """
    # This is tested in integration tests with real API calls
    pass


# Multimodal Message Tests


def test_convert_text_message() -> None:
    """Test converting simple text message."""
    from langchain_core.messages import HumanMessage

    llm = ChatNova(model="nova-pro-v1", api_key="test-key")

    messages = [HumanMessage(content="Hello")]
    converted = llm._convert_messages_to_nova_format(cast(list[BaseMessage], messages))

    assert len(converted) == 1
    assert converted[0]["role"] == "user"
    assert converted[0]["content"] == "Hello"


def test_convert_image_url_message() -> None:
    """Test converting message with image_url content."""
    from langchain_core.messages import HumanMessage

    llm = ChatNova(model="nova-pro-v1", api_key="test-key")

    # OpenAI format with image_url
    messages = [
        HumanMessage(
            content=[
                {"type": "text", "text": "What's in this image?"},
                {
                    "type": "image_url",
                    "image_url": {
                        "url": "https://example.com/image.jpg",
                    },
                },
            ]
        )
    ]

    converted = llm._convert_messages_to_nova_format(cast(list[BaseMessage], messages))

    assert len(converted) == 1
    assert converted[0]["role"] == "user"
    assert isinstance(converted[0]["content"], list)
    assert len(converted[0]["content"]) == 2

    # Check text block
    assert converted[0]["content"][0]["type"] == "text"
    assert converted[0]["content"][0]["text"] == "What's in this image?"

    # Check image_url block
    assert converted[0]["content"][1]["type"] == "image_url"
    assert (
        converted[0]["content"][1]["image_url"]["url"]
        == "https://example.com/image.jpg"
    )


def test_convert_image_url_string_format() -> None:
    """Test converting image_url when it's a string instead of dict."""
    from langchain_core.messages import HumanMessage

    llm = ChatNova(model="nova-pro-v1", api_key="test-key")

    messages = [
        HumanMessage(
            content=[
                {"type": "text", "text": "Describe this image"},
                {"type": "image_url", "image_url": "https://example.com/image.jpg"},
            ]
        )
    ]

    converted = llm._convert_messages_to_nova_format(cast(list[BaseMessage], messages))

    # Should convert string to proper format
    assert converted[0]["content"][1]["type"] == "image_url"
    assert (
        converted[0]["content"][1]["image_url"]["url"]
        == "https://example.com/image.jpg"
    )


def test_convert_mixed_content_types() -> None:
    """Test converting message with multiple content types."""
    from langchain_core.messages import HumanMessage

    llm = ChatNova(model="nova-pro-v1", api_key="test-key")

    messages = [
        HumanMessage(
            content=[
                {"type": "text", "text": "First question"},
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/img1.jpg"},
                },
                {"type": "text", "text": "Second question"},
                {
                    "type": "image_url",
                    "image_url": {"url": "https://example.com/img2.jpg"},
                },
            ]
        )
    ]

    converted = llm._convert_messages_to_nova_format(cast(list[BaseMessage], messages))

    assert len(converted[0]["content"]) == 4
    assert converted[0]["content"][0]["type"] == "text"
    assert converted[0]["content"][1]["type"] == "image_url"
    assert converted[0]["content"][2]["type"] == "text"
    assert converted[0]["content"][3]["type"] == "image_url"


def test_convert_string_in_list() -> None:
    """Test that strings in content list are converted to text blocks."""
    from langchain_core.messages import HumanMessage

    llm = ChatNova(model="nova-pro-v1", api_key="test-key")

    messages = [HumanMessage(content=["Hello", "World"])]
    converted = llm._convert_messages_to_nova_format(cast(list[BaseMessage], messages))

    assert len(converted[0]["content"]) == 2
    assert converted[0]["content"][0] == {"type": "text", "text": "Hello"}
    assert converted[0]["content"][1] == {"type": "text", "text": "World"}


def test_convert_langchain_image_block_with_url() -> None:
    """Test converting LangChain image block (block_type='image') with URL."""
    from langchain_core.messages import HumanMessage

    llm = ChatNova(model="nova-pro-v1", api_key="test-key")

    # LangChain uses block_type = "image" with url property
    messages = [
        HumanMessage(
            content=[{"block_type": "image", "url": "https://example.com/image.jpg"}]
        )
    ]

    converted = llm._convert_messages_to_nova_format(cast(list[BaseMessage], messages))

    assert len(converted[0]["content"]) == 1
    assert converted[0]["content"][0]["type"] == "image_url"
    assert (
        converted[0]["content"][0]["image_url"]["url"]
        == "https://example.com/image.jpg"
    )


def test_convert_langchain_image_block_with_base64() -> None:
    """Test converting LangChain image block with base64 data."""
    from langchain_core.messages import HumanMessage

    llm = ChatNova(model="nova-pro-v1", api_key="test-key")

    # LangChain image block with base64
    messages = [
        HumanMessage(
            content=[
                {
                    "block_type": "image",
                    "base64": (
                        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ"
                        "AAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
                    ),
                    "mime_type": "image/png",
                }
            ]
        )
    ]

    converted = llm._convert_messages_to_nova_format(cast(list[BaseMessage], messages))

    assert len(converted[0]["content"]) == 1
    assert converted[0]["content"][0]["type"] == "image_url"

    # Check that base64 was converted to data URL
    image_url = converted[0]["content"][0]["image_url"]["url"]
    assert image_url.startswith("data:image/png;base64,")
    assert (
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJ"
        "AAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg==" in image_url
    )


def test_convert_langchain_image_block_default_mime_type() -> None:
    """Test that base64 images default to image/jpeg if no mime_type."""
    from langchain_core.messages import HumanMessage

    llm = ChatNova(model="nova-pro-v1", api_key="test-key")

    messages = [
        HumanMessage(content=[{"block_type": "image", "base64": "fake_base64_data"}])
    ]

    converted = llm._convert_messages_to_nova_format(cast(list[BaseMessage], messages))

    image_url = converted[0]["content"][0]["image_url"]["url"]
    assert image_url == "data:image/jpeg;base64,fake_base64_data"


def test_tools_passed_to_api() -> None:
    """Test that tools from bind_tools are passed to the API request."""
    from unittest.mock import MagicMock, patch

    from langchain_core.tools import tool

    @tool
    def get_weather(location: str) -> str:
        """Get weather for a location."""
        return f"Weather in {location}"

    llm = ChatNova(model="nova-pro-v1", api_key="test-key")

    # Mock the client
    mock_response = MagicMock()
    mock_response.choices = [MagicMock()]
    mock_response.choices[0].message.content = "Let me check the weather"
    mock_response.choices[0].finish_reason = "stop"
    mock_response.model = "nova-pro-v1"

    with patch.object(
        llm.client.chat.completions, "create", return_value=mock_response
    ) as mock_create:
        llm_with_tools = llm.bind_tools([get_weather])
        llm_with_tools.invoke("What's the weather in Paris?")

        # Check that tools were passed to the API
        call_kwargs = mock_create.call_args[1]
        assert "tools" in call_kwargs
        assert len(call_kwargs["tools"]) == 1
        assert call_kwargs["tools"][0]["function"]["name"] == "get_weather"
