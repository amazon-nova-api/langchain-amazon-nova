"""Model capabilities registry for Amazon Nova models."""

from dataclasses import dataclass
from typing import Dict, Literal, Optional


@dataclass
class ModelCapabilities:
    """Capabilities for a Nova model.

    Attributes:
        supports_text: Whether the model supports text input/output
        supports_vision: Whether the model can process images/video as input
        supports_tool_calling: Whether the model supports function calling
        supports_image_generation: Whether the model can generate images
        supports_streaming: Whether the model supports streaming responses
        max_context_tokens: Maximum context window size
        modality: Primary modality of the model
    """

    supports_text: bool = True
    supports_vision: bool = False
    supports_tool_calling: bool = True
    supports_image_generation: bool = False
    supports_streaming: bool = True
    max_context_tokens: Optional[int] = None
    modality: Literal["text", "multimodal", "image-generation"] = "text"


# Registry of known Nova models and their capabilities
MODEL_CAPABILITIES: Dict[str, ModelCapabilities] = {
    # Text-only models
    "nova-micro-v1": ModelCapabilities(
        supports_text=True,
        supports_vision=False,
        supports_tool_calling=True,
        supports_image_generation=False,
        supports_streaming=True,
        max_context_tokens=128000,
        modality="text",
    ),
    # Multimodal models (vision input - text, images, videos)
    "nova-lite-v1": ModelCapabilities(
        supports_text=True,
        supports_vision=True,
        supports_tool_calling=True,
        supports_image_generation=False,
        supports_streaming=True,
        max_context_tokens=300000,
        modality="multimodal",
    ),
    "nova-pro-v1": ModelCapabilities(
        supports_text=True,
        supports_vision=True,
        supports_tool_calling=True,
        supports_image_generation=False,
        supports_streaming=True,
        max_context_tokens=300000,
        modality="multimodal",
    ),
    "nova-premier-v1": ModelCapabilities(
        supports_text=True,
        supports_vision=True,
        supports_tool_calling=True,
        supports_image_generation=False,
        supports_streaming=True,
        max_context_tokens=1000000,
        modality="multimodal",
    ),
    # Canvas models (image generation)
    "nova-canvas-v1": ModelCapabilities(
        supports_text=True,
        supports_vision=False,
        supports_tool_calling=False,
        supports_image_generation=True,
        supports_streaming=False,
        max_context_tokens=None,
        modality="image-generation",
    ),
    # Research models
    "nova-deep-research-v1": ModelCapabilities(
        supports_text=True,
        supports_vision=False,
        supports_tool_calling=True,
        supports_image_generation=False,
        supports_streaming=True,
        max_context_tokens=300000,
        modality="text",
    ),
}

# Default capabilities for unknown models (assume basic text model)
DEFAULT_CAPABILITIES = ModelCapabilities(
    supports_text=True,
    supports_vision=False,
    supports_tool_calling=False,
    supports_image_generation=False,
    supports_streaming=True,
    max_context_tokens=None,
    modality="text",
)


def get_model_capabilities(model_name: str) -> ModelCapabilities:
    """Get capabilities for a model.

    Args:
        model_name: Name of the Nova model

    Returns:
        ModelCapabilities instance for the model. Returns default capabilities
        for unknown models.
    """
    return MODEL_CAPABILITIES.get(model_name, DEFAULT_CAPABILITIES)


def validate_tool_calling(model_name: str) -> None:
    """Validate that a model supports tool calling.

    Args:
        model_name: Name of the Nova model

    Raises:
        ValueError: If the model doesn't support tool calling
    """
    capabilities = get_model_capabilities(model_name)
    if not capabilities.supports_tool_calling:
        raise ValueError(
            f"Model '{model_name}' does not support tool calling. "
            f"Tool calling is supported by: {', '.join(m for m, c in MODEL_CAPABILITIES.items() if c.supports_tool_calling)}"
        )


def validate_vision_input(model_name: str) -> None:
    """Validate that a model supports vision input.

    Args:
        model_name: Name of the Nova model

    Raises:
        ValueError: If the model doesn't support vision input
    """
    capabilities = get_model_capabilities(model_name)
    if not capabilities.supports_vision:
        raise ValueError(
            f"Model '{model_name}' does not support image/video input. "
            f"Vision is supported by: {', '.join(m for m, c in MODEL_CAPABILITIES.items() if c.supports_vision)}"
        )


def validate_image_generation(model_name: str) -> None:
    """Validate that a model supports image generation.

    Args:
        model_name: Name of the Nova model

    Raises:
        ValueError: If the model doesn't support image generation
    """
    capabilities = get_model_capabilities(model_name)
    if not capabilities.supports_image_generation:
        raise ValueError(
            f"Model '{model_name}' does not support image generation. "
            f"Image generation is supported by: {', '.join(m for m, c in MODEL_CAPABILITIES.items() if c.supports_image_generation)}"
        )


def is_multimodal_model(model_name: str) -> bool:
    """Check if a model is multimodal (accepts vision input).

    Args:
        model_name: Name of the Nova model

    Returns:
        True if the model supports vision input
    """
    return get_model_capabilities(model_name).supports_vision


def is_image_generation_model(model_name: str) -> bool:
    """Check if a model generates images.

    Args:
        model_name: Name of the Nova model

    Returns:
        True if the model supports image generation
    """
    return get_model_capabilities(model_name).supports_image_generation


__all__ = [
    "ModelCapabilities",
    "MODEL_CAPABILITIES",
    "DEFAULT_CAPABILITIES",
    "get_model_capabilities",
    "validate_tool_calling",
    "validate_vision_input",
    "validate_image_generation",
    "is_multimodal_model",
    "is_image_generation_model",
]
