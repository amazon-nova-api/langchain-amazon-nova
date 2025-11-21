"""Unit tests for model capabilities."""

import pytest

from langchain_nova.models import (
    MODEL_CAPABILITIES,
    ModelCapabilities,
    get_model_capabilities,
    is_image_generation_model,
    is_multimodal_model,
    validate_image_generation,
    validate_tool_calling,
    validate_vision_input,
)


class TestModelCapabilities:
    """Test model capabilities dataclass and registry."""

    def test_model_capabilities_defaults(self) -> None:
        """Test ModelCapabilities default values."""
        caps = ModelCapabilities()
        assert caps.supports_text is True
        assert caps.supports_vision is False
        assert caps.supports_tool_calling is True
        assert caps.supports_image_generation is False
        assert caps.supports_streaming is True
        assert caps.max_context_tokens is None
        assert caps.modality == "text"

    def test_model_capabilities_custom(self) -> None:
        """Test ModelCapabilities with custom values."""
        caps = ModelCapabilities(
            supports_text=True,
            supports_vision=True,
            supports_tool_calling=True,
            supports_image_generation=False,
            supports_streaming=True,
            max_context_tokens=300000,
            modality="multimodal",
        )
        assert caps.supports_text is True
        assert caps.supports_vision is True
        assert caps.supports_tool_calling is True
        assert caps.supports_image_generation is False
        assert caps.max_context_tokens == 300000
        assert caps.modality == "multimodal"


class TestModelRegistry:
    """Test model registry."""

    def test_registry_has_expected_models(self) -> None:
        """Test that registry contains expected models."""
        expected_models = [
            "nova-micro-v1",
            "nova-lite-v1",
            "nova-pro-v1",
            "nova-premier-v1",
        ]
        for model in expected_models:
            assert model in MODEL_CAPABILITIES

    def test_text_only_models(self) -> None:
        """Test text-only model capabilities."""
        text_models = ["nova-micro-v1"]
        for model_name in text_models:
            caps = MODEL_CAPABILITIES[model_name]
            assert caps.supports_text is True
            assert caps.supports_vision is False
            assert caps.supports_tool_calling is True
            assert caps.supports_image_generation is False
            assert caps.modality == "text"

    def test_multimodal_models(self) -> None:
        """Test multimodal model capabilities."""
        # nova-lite-v1, nova-pro-v1, and nova-premier-v1 support vision
        caps = MODEL_CAPABILITIES["nova-premier-v1"]
        assert caps.supports_text is True
        assert caps.supports_vision is True
        assert caps.supports_tool_calling is True
        assert caps.supports_image_generation is False
        assert caps.modality == "multimodal"

        caps = MODEL_CAPABILITIES["nova-pro-v1"]
        assert caps.supports_text is True
        assert caps.supports_vision is True
        assert caps.supports_tool_calling is True
        assert caps.supports_image_generation is False
        assert caps.modality == "multimodal"

        caps = MODEL_CAPABILITIES["nova-lite-v1"]
        assert caps.supports_text is True
        assert caps.supports_vision is True
        assert caps.supports_tool_calling is True
        assert caps.supports_image_generation is False
        assert caps.modality == "multimodal"

    def test_image_generation_models(self) -> None:
        """Test image generation model capabilities."""
        if "nova-canvas-v1" in MODEL_CAPABILITIES:
            caps = MODEL_CAPABILITIES["nova-canvas-v1"]
            assert caps.supports_text is True
            assert caps.supports_vision is False
            assert caps.supports_tool_calling is False
            assert caps.supports_image_generation is True
            assert caps.modality == "image-generation"


class TestGetModelCapabilities:
    """Test get_model_capabilities function."""

    def test_get_known_model(self) -> None:
        """Test getting capabilities for known model."""
        caps = get_model_capabilities("nova-pro-v1")
        assert isinstance(caps, ModelCapabilities)
        assert caps.supports_text is True
        assert caps.supports_tool_calling is True

    def test_get_unknown_model(self) -> None:
        """Test getting capabilities for unknown model returns defaults."""
        caps = get_model_capabilities("unknown-model-v99")
        assert isinstance(caps, ModelCapabilities)
        # Should return default capabilities
        assert caps.supports_text is True
        assert caps.supports_vision is False
        assert caps.supports_tool_calling is False

    def test_all_registered_models_retrievable(self) -> None:
        """Test that all registered models can be retrieved."""
        for model_name in MODEL_CAPABILITIES.keys():
            caps = get_model_capabilities(model_name)
            assert isinstance(caps, ModelCapabilities)
            assert caps == MODEL_CAPABILITIES[model_name]


class TestHelperFunctions:
    """Test helper functions."""

    def test_is_multimodal_model(self) -> None:
        """Test is_multimodal_model function."""
        assert is_multimodal_model("nova-premier-v1") is True
        assert is_multimodal_model("nova-pro-v1") is True
        assert is_multimodal_model("nova-lite-v1") is True
        assert is_multimodal_model("nova-micro-v1") is False

    def test_is_image_generation_model(self) -> None:
        """Test is_image_generation_model function."""
        if "nova-canvas-v1" in MODEL_CAPABILITIES:
            assert is_image_generation_model("nova-canvas-v1") is True
        assert is_image_generation_model("nova-pro-v1") is False
        assert is_image_generation_model("nova-premier-v1") is False


class TestValidation:
    """Test validation functions."""

    def test_validate_tool_calling_success(self) -> None:
        """Test tool calling validation succeeds for supporting models."""
        # Should not raise
        validate_tool_calling("nova-pro-v1")
        validate_tool_calling("nova-lite-v1")
        validate_tool_calling("nova-premier-v1")

    def test_validate_tool_calling_failure(self) -> None:
        """Test tool calling validation fails for non-supporting models."""
        if "nova-canvas-v1" in MODEL_CAPABILITIES:
            with pytest.raises(ValueError, match="does not support tool calling"):
                validate_tool_calling("nova-canvas-v1")

    def test_validate_vision_input_success(self) -> None:
        """Test vision input validation succeeds for supporting models."""
        # Should not raise
        validate_vision_input("nova-pro-v1")
        validate_vision_input("nova-premier-v1")
        validate_vision_input("nova-lite-v1")

    def test_validate_vision_input_failure(self) -> None:
        """Test vision input validation fails for non-supporting models."""
        with pytest.raises(ValueError, match="does not support image/video input"):
            validate_vision_input("nova-micro-v1")

    def test_validate_image_generation_success(self) -> None:
        """Test image generation validation succeeds for supporting models."""
        if "nova-canvas-v1" in MODEL_CAPABILITIES:
            # Should not raise
            validate_image_generation("nova-canvas-v1")

    def test_validate_image_generation_failure(self) -> None:
        """Test image generation validation fails for non-supporting models."""
        with pytest.raises(ValueError, match="does not support image generation"):
            validate_image_generation("nova-pro-v1")


class TestContextWindows:
    """Test context window information."""

    def test_micro_context_window(self) -> None:
        """Test nova-micro-v1 context window."""
        caps = get_model_capabilities("nova-micro-v1")
        assert caps.max_context_tokens == 128000

    def test_lite_context_window(self) -> None:
        """Test nova-lite-v1 context window."""
        caps = get_model_capabilities("nova-lite-v1")
        assert caps.max_context_tokens == 300000

    def test_pro_context_window(self) -> None:
        """Test nova-pro-v1 context window."""
        caps = get_model_capabilities("nova-pro-v1")
        assert caps.max_context_tokens == 300000

    def test_premier_context_window(self) -> None:
        """Test nova-premier-v1 context window."""
        caps = get_model_capabilities("nova-premier-v1")
        assert caps.max_context_tokens == 1000000
