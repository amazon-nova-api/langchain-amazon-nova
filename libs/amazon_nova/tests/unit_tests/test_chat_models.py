"""Basic unit tests for ChatNova."""

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

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


def test_max_completion_tokens_initialization():
    """Test that max_completion_tokens can be set at initialization."""
    llm = ChatNova(model="nova-pro-v1", max_completion_tokens=200, api_key="test-key")
    assert llm.max_completion_tokens == 200


def test_top_p_initialization():
    """Test that top_p can be set at initialization."""
    llm = ChatNova(model="nova-pro-v1", top_p=0.9, api_key="test-key")
    assert llm.top_p == 0.9


def test_top_p_validation():
    """Test that top_p is validated to be between 0 and 1."""
    with pytest.raises(ValueError):
        ChatNova(model="nova-pro-v1", top_p=1.5, api_key="test-key")

    with pytest.raises(ValueError):
        ChatNova(model="nova-pro-v1", top_p=-0.1, api_key="test-key")


def test_reasoning_effort_initialization():
    """Test that reasoning_effort can be set at initialization."""
    for effort in ["low", "medium", "high"]:
        llm = ChatNova(model="nova-pro-v1", reasoning_effort=effort, api_key="test-key")
        assert llm.reasoning_effort == effort


def test_metadata_initialization():
    """Test that metadata can be set at initialization."""
    metadata = {"user_id": "123", "session_id": "abc"}
    llm = ChatNova(model="nova-pro-v1", metadata=metadata, api_key="test-key")
    assert llm.metadata == metadata


def test_stream_options_initialization():
    """Test that stream_options can be set at initialization."""
    stream_opts = {"include_usage": True}
    llm = ChatNova(model="nova-pro-v1", stream_options=stream_opts, api_key="test-key")
    assert llm.stream_options == stream_opts


def test_merge_params_max_tokens():
    """Test that _merge_params correctly handles max_tokens."""
    llm = ChatNova(model="nova-pro-v1", max_tokens=100, api_key="test-key")

    # Model-level default
    params = llm._merge_params({})
    assert params["max_tokens"] == 100

    # Invoke-level override
    params = llm._merge_params({"max_tokens": 200})
    assert params["max_tokens"] == 200


def test_merge_params_max_completion_tokens_precedence():
    """Test that max_completion_tokens takes precedence over max_tokens."""
    llm = ChatNova(model="nova-pro-v1", max_tokens=100, max_completion_tokens=150, api_key="test-key")

    # max_completion_tokens should win
    params = llm._merge_params({})
    assert "max_completion_tokens" in params
    assert params["max_completion_tokens"] == 150
    assert "max_tokens" not in params

    # Invoke-level max_completion_tokens overrides both
    params = llm._merge_params({"max_completion_tokens": 200})
    assert params["max_completion_tokens"] == 200
    assert "max_tokens" not in params

    # Without max_completion_tokens at either level, use max_tokens
    llm2 = ChatNova(model="nova-pro-v1", max_tokens=100, api_key="test-key")
    params = llm2._merge_params({})
    assert "max_tokens" in params
    assert params["max_tokens"] == 100


def test_merge_params_top_p():
    """Test that _merge_params correctly handles top_p."""
    llm = ChatNova(model="nova-pro-v1", top_p=0.8, api_key="test-key")

    params = llm._merge_params({})
    assert params["top_p"] == 0.8

    params = llm._merge_params({"top_p": 0.95})
    assert params["top_p"] == 0.95

    llm2 = ChatNova(model="nova-pro-v1", api_key="test-key")
    params = llm2._merge_params({})
    assert "top_p" not in params


def test_merge_params_reasoning_effort():
    """Test that _merge_params correctly handles reasoning_effort."""
    llm = ChatNova(model="nova-pro-v1", reasoning_effort="medium", api_key="test-key")

    params = llm._merge_params({})
    assert params["reasoning_effort"] == "medium"

    params = llm._merge_params({"reasoning_effort": "high"})
    assert params["reasoning_effort"] == "high"

    llm2 = ChatNova(model="nova-pro-v1", api_key="test-key")
    params = llm2._merge_params({})
    assert "reasoning_effort" not in params


def test_merge_params_metadata():
    """Test that _merge_params correctly handles metadata."""
    llm = ChatNova(model="nova-pro-v1", metadata={"app": "test"}, api_key="test-key")

    params = llm._merge_params({})
    assert params["metadata"] == {"app": "test"}

    params = llm._merge_params({"metadata": {"user": "123"}})
    assert params["metadata"] == {"user": "123"}

    llm2 = ChatNova(model="nova-pro-v1", api_key="test-key")
    params = llm2._merge_params({})
    assert "metadata" not in params
