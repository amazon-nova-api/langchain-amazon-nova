"""Standard integration tests for ChatAmazonNova using langchain-tests."""

from typing import Any, Type

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_tests.integration_tests import ChatModelIntegrationTests

from langchain_amazon_nova import ChatAmazonNova


class TestChatAmazonNovaIntegration(ChatModelIntegrationTests):
    """Standard integration tests for ChatAmazonNova."""

    @property
    def chat_model_class(self) -> Type[BaseChatModel]:
        """Return the chat model class to test."""
        return ChatAmazonNova

    @property
    def chat_model_params(self) -> dict:
        """Return initialization parameters for the chat model.

        Note: Requires NOVA_API_KEY and NOVA_BASE_URL environment variables.
        """
        return {
            "model": "nova-pro-v1",
            "temperature": 0.7,
            "stream_options": {"include_usage": True},
        }

    @property
    def has_tool_calling(self) -> bool:
        """Whether the model supports tool calling."""
        return True

    @property
    def supports_image_inputs(self) -> bool:
        """Whether the model supports image inputs."""
        return True  # Nova Pro, Lite, and Premier support vision

    @property
    def supports_video_inputs(self) -> bool:
        """Whether the model supports video inputs."""
        return False  # Video support not yet implemented

    @property
    def supports_audio_inputs(self) -> bool:
        """Whether the model supports audio inputs."""
        return False  # Audio support not yet implemented

    @property
    def supports_anthropic_inputs(self) -> bool:
        """Whether the model supports Anthropic-specific input formats."""
        return False

    @property
    def returns_usage_metadata(self) -> bool:
        """Whether the model returns usage metadata."""
        return True

    # Override to indicate structured output is not supported
    @pytest.mark.xfail(reason="with_structured_output not implemented")
    def test_structured_output(self, *args: Any, **kwargs: Any) -> None:
        super().test_structured_output(*args, **kwargs)

    @pytest.mark.xfail(reason="with_structured_output not implemented")
    async def test_structured_output_async(self, *args: Any, **kwargs: Any) -> None:
        await super().test_structured_output_async(*args, **kwargs)

    @pytest.mark.xfail(reason="with_structured_output not implemented")
    def test_structured_output_pydantic_2_v1(self, *args: Any, **kwargs: Any) -> None:
        super().test_structured_output_pydantic_2_v1(*args, **kwargs)

    @pytest.mark.xfail(reason="with_structured_output not implemented")
    def test_structured_output_optional_param(self, *args: Any, **kwargs: Any) -> None:
        super().test_structured_output_optional_param(*args, **kwargs)

    @pytest.mark.xfail(reason="with_structured_output not implemented")
    def test_structured_few_shot_examples(self, *args: Any, **kwargs: Any) -> None:
        super().test_structured_few_shot_examples(*args, **kwargs)

    @pytest.mark.xfail(reason="JSON mode not implemented")
    def test_json_mode(self, *args: Any, **kwargs: Any) -> None:
        super().test_json_mode(*args, **kwargs)

    @pytest.mark.xfail(reason="tool_choice format needs investigation")
    def test_unicode_tool_call_integration(self, *args: Any, **kwargs: Any) -> None:
        super().test_unicode_tool_call_integration(*args, **kwargs)

    @pytest.mark.xfail(reason="Tool calling test parametrization needs investigation")
    def test_tool_calling(self, *args: Any, **kwargs: Any) -> None:
        super().test_tool_calling(*args, **kwargs)

    @pytest.mark.xfail(reason="Tool calling test parametrization needs investigation")
    async def test_tool_calling_async(self, *args: Any, **kwargs: Any) -> None:
        await super().test_tool_calling_async(*args, **kwargs)

    @pytest.mark.xfail(reason="Tools with no arguments not supported")
    def test_tool_calling_with_no_arguments(self, *args: Any, **kwargs: Any) -> None:
        super().test_tool_calling_with_no_arguments(*args, **kwargs)

    @pytest.mark.xfail(reason="Tools with no arguments not supported")
    async def test_tool_calling_with_no_arguments_async(
        self, *args: Any, **kwargs: Any
    ) -> None:
        super().test_tool_calling_with_no_arguments_async(*args, **kwargs)
