"""Example demonstrating vision/multimodal capabilities with images.

This script shows how to use Nova models with image inputs using the
OpenAI-compatible image_url format.

Supports:
- Image URLs (http/https)
- Base64 encoded images (data:image/...)
- Multiple images in one request
- Mix of text and images
"""

import argparse
import base64
from pathlib import Path

from langchain_core.messages import HumanMessage

from langchain_nova import ChatNova


def encode_image_to_base64(image_path: str) -> str:
    """Encode a local image file to base64 data URL.

    Args:
        image_path: Path to local image file

    Returns:
        Base64 encoded data URL string
    """
    path = Path(image_path)
    mime_type = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }.get(path.suffix.lower(), "image/jpeg")

    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")

    return f"data:{mime_type};base64,{image_data}"


def example_single_image_url(model: str = "nova-pro-v1", verbose: bool = False):
    """Example 1: Simple image URL with question."""
    print("Example 1: Single Image URL\n")

    llm = ChatNova(model=model, temperature=0.7)

    # Create message with image URL
    message = HumanMessage(
        content=[
            {"type": "text", "text": "What's in this image? Describe it in detail."},
            {
                "type": "image_url",
                "image_url": {
                    "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
                    "detail": "high"
                }
            }
        ]
    )

    if verbose:
        print(f"[DEBUG] Sending image URL: {message.content[1]['image_url']['url'][:60]}...")

    response = llm.invoke([message])

    print(f"Response: {response.content}\n")

    if verbose and hasattr(response, "usage_metadata"):
        print(f"[DEBUG] Tokens: {response.usage_metadata}")


def example_multiple_images(model: str = "nova-pro-v1", verbose: bool = False):
    """Example 2: Multiple images in one request."""
    print("Example 2: Multiple Images\n")

    llm = ChatNova(model=model, temperature=0.7)

    message = HumanMessage(
        content=[
            {"type": "text", "text": "Compare these two images. What are the similarities and differences?"},
            {
                "type": "image_url",
                "image_url": {
                    "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
                }
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/3/3f/Fronalpstock_big.jpg/2560px-Fronalpstock_big.jpg"
                }
            }
        ]
    )

    if verbose:
        print(f"[DEBUG] Sending {len([c for c in message.content if c.get('type') == 'image_url'])} images")

    response = llm.invoke([message])

    print(f"Response: {response.content}\n")


def example_image_with_detail_level(model: str = "nova-pro-v1", verbose: bool = False):
    """Example 3: Image with different detail levels."""
    print("Example 3: Image Detail Levels\n")

    llm = ChatNova(model=model, temperature=0.7)

    # Try with low detail (faster, cheaper)
    print("Low detail analysis:")
    message_low = HumanMessage(
        content=[
            {"type": "text", "text": "What's the main subject of this image?"},
            {
                "type": "image_url",
                "image_url": {
                    "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
                    "detail": "low"
                }
            }
        ]
    )

    response_low = llm.invoke([message_low])
    print(f"  {response_low.content[:100]}...\n")

    # Try with high detail (more accurate)
    print("High detail analysis:")
    message_high = HumanMessage(
        content=[
            {"type": "text", "text": "What's the main subject of this image?"},
            {
                "type": "image_url",
                "image_url": {
                    "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg",
                    "detail": "high"
                }
            }
        ]
    )

    response_high = llm.invoke([message_high])
    print(f"  {response_high.content[:100]}...\n")


def example_base64_image(image_path: str, model: str = "nova-pro-v1", verbose: bool = False):
    """Example 4: Base64 encoded local image."""
    print(f"Example 4: Base64 Encoded Image\n")

    llm = ChatNova(model=model, temperature=0.7)

    # Encode local image to base64
    if verbose:
        print(f"[DEBUG] Encoding image from {image_path}")

    base64_url = encode_image_to_base64(image_path)

    if verbose:
        print(f"[DEBUG] Base64 URL length: {len(base64_url)} characters")

    message = HumanMessage(
        content=[
            {"type": "text", "text": "Describe this image in detail."},
            {
                "type": "image_url",
                "image_url": {
                    "url": base64_url
                }
            }
        ]
    )

    response = llm.invoke([message])

    print(f"Response: {response.content}\n")


def example_conversation_with_images(model: str = "nova-pro-v1", verbose: bool = False):
    """Example 5: Multi-turn conversation with images."""
    print("Example 5: Conversation with Images\n")

    llm = ChatNova(model=model, temperature=0.7)

    # First turn: Ask about image
    message1 = HumanMessage(
        content=[
            {"type": "text", "text": "What do you see in this image?"},
            {
                "type": "image_url",
                "image_url": {
                    "url": "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
                }
            }
        ]
    )

    response1 = llm.invoke([message1])
    print(f"Turn 1:")
    print(f"  User: What do you see in this image?")
    print(f"  Assistant: {response1.content}\n")

    # Second turn: Follow-up question (no image needed, uses context)
    message2 = HumanMessage(content="What time of day do you think it was taken?")

    response2 = llm.invoke([message1, response1, message2])
    print(f"Turn 2:")
    print(f"  User: What time of day do you think it was taken?")
    print(f"  Assistant: {response2.content}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Demonstrate Nova vision/multimodal capabilities"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="nova-pro-v1",
        help="Nova model to use (default: nova-pro-v1)",
    )
    parser.add_argument(
        "--example",
        type=int,
        choices=[1, 2, 3, 4, 5],
        help="Which example to run (1-5)",
    )
    parser.add_argument(
        "--image",
        type=str,
        help="Path to local image file (for example 4)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output",
    )

    args = parser.parse_args()

    # Check if model supports vision
    from langchain_nova.models import get_model_capabilities
    caps = get_model_capabilities(args.model)
    if not caps.supports_vision:
        print(f"Warning: Model {args.model} may not support vision inputs.")
        print(f"Consider using: nova-lite-v1, nova-pro-v1, or nova-premier-v1\n")

    if args.example == 1 or args.example is None:
        example_single_image_url(args.model, args.verbose)

    if args.example == 2 or args.example is None:
        example_multiple_images(args.model, args.verbose)

    if args.example == 3 or args.example is None:
        example_image_with_detail_level(args.model, args.verbose)

    if args.example == 4 and args.image:
        example_base64_image(args.image, args.model, args.verbose)
    elif args.example == 4:
        print("Example 4 requires --image parameter\n")

    if args.example == 5 or args.example is None:
        example_conversation_with_images(args.model, args.verbose)


if __name__ == "__main__":
    main()
