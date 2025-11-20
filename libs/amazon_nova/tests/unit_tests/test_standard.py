"""Standard unit tests for ChatNova using langchain-tests."""

from typing import Type

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_tests.unit_tests import ChatModelUnitTests

from langchain_nova import ChatNova


class TestChatNovaUnit(ChatModelUnitTests):
    """Standard unit tests for ChatNova."""

    @property
    def chat_model_class(self) -> Type[BaseChatModel]:
        """Return the chat model class to test."""
        return ChatNova

    @property
    def has_structured_output(self) -> bool:
        """Structured output not yet implemented."""
        return False

    @property
    def chat_model_params(self) -> dict:
        """Return initialization parameters for the chat model."""
        return {
            "model": "nova-pro-v1",
            "temperature": 0,
            "api_key": "test-api-key",
        }

    @pytest.mark.xfail(reason="Not yet implemented")
    def test_bind_tools(self) -> None:
        """Tool binding not yet implemented."""
        pass

    @pytest.mark.xfail(reason="Not yet implemented")
    def test_tool_calling(self) -> None:
        """Tool calling not yet implemented."""
        pass

    @pytest.mark.xfail(reason="Not yet implemented")
    def test_structured_output(self) -> None:
        """Structured output not yet implemented."""
        pass

    @pytest.mark.xfail(reason="with_structured_output not yet implemented")
    def test_with_structured_output(self) -> None:
        """Structured output test not yet implemented."""
        pass
