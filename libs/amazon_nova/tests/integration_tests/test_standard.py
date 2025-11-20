"""Standard integration tests for ChatNova using langchain-tests."""

from typing import Type

from langchain_core.language_models import BaseChatModel
from langchain_tests.integration_tests import ChatModelIntegrationTests

from langchain_nova import ChatNova


class TestChatNovaIntegration(ChatModelIntegrationTests):
    """Standard integration tests for ChatNova."""

    @property
    def chat_model_class(self) -> Type[BaseChatModel]:
        """Return the chat model class to test."""
        return ChatNova

    @property
    def chat_model_params(self) -> dict:
        """Return initialization parameters for the chat model.

        Note: Requires NOVA_API_KEY and NOVA_BASE_URL environment variables.
        """
        return {
            "model": "nova-pro-v1",
            "temperature": 0,
        }

    @property
    def has_tool_calling(self) -> bool:
        """Whether the model supports tool calling."""
        return True

    @property
    def supports_image_inputs(self) -> bool:
        """Whether the model supports image inputs."""
        return False  # TODO: Update based on Nova's capabilities

    @property
    def supports_video_inputs(self) -> bool:
        """Whether the model supports video inputs."""
        return False  # TODO: Update based on Nova's capabilities

    @property
    def supports_audio_inputs(self) -> bool:
        """Whether the model supports audio inputs."""
        return False  # TODO: Update based on Nova's capabilities

    @property
    def supports_anthropic_inputs(self) -> bool:
        """Whether the model supports Anthropic-specific input formats."""
        return False

    @property
    def returns_usage_metadata(self) -> bool:
        """Whether the model returns usage metadata."""
        return True
