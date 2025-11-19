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

    converted = llm._convert_messages_to_openai_format(messages)

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
