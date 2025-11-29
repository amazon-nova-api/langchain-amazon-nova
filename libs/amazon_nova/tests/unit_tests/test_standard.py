"""Standard unit tests for ChatAmazonNova using langchain-tests."""

from typing import Type

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_tests.unit_tests import ChatModelUnitTests

from langchain_amazon_nova import ChatAmazonNova


class TestChatAmazonNovaUnit(ChatModelUnitTests):
    """Standard unit tests for ChatAmazonNova."""

    @property
    def chat_model_class(self) -> Type[BaseChatModel]:
        """Return the chat model class to test."""
        return ChatAmazonNova

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

    @pytest.mark.xfail(reason="with_structured_output not yet implemented")
    def test_with_structured_output(self) -> None:
        """Structured output test not yet implemented."""
        pass
