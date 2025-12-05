"""Standard unit tests for ChatAmazonNova using langchain-tests."""

from typing import Any, Type

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
        """Structured output is implemented."""
        return True

    @property
    def chat_model_params(self) -> dict:
        """Return initialization parameters for the chat model."""
        return {
            "model": "nova-pro-v1",
            "temperature": 0.7,
            "api_key": "test-api-key",
        }

    @property
    def supports_tool_choice_values(self) -> list:
        """Tool choice values supported by Nova."""
        return ["auto", "required", "none"]

    @pytest.mark.xfail(
        reason=(
            "tool_choice='any' not supported - "
            "Nova only supports 'auto', 'required', 'none'"
        )
    )
    def test_bind_tool_pydantic(self, *args: Any, **kwargs: Any) -> None:
        super().test_bind_tool_pydantic(*args, **kwargs)
