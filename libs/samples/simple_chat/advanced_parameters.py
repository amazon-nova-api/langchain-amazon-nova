"""Demonstrates advanced Nova API parameters.

Shows usage of Phase 1 parameters:
- max_tokens / max_completion_tokens
- top_p (nucleus sampling)
- reasoning_effort
- metadata
- stream_options

Requirements:
    - NOVA_API_KEY environment variable
    - NOVA_BASE_URL environment variable

Usage:
    python advanced_parameters.py
    python advanced_parameters.py --model nova-pro-v1
    python advanced_parameters.py --reasoning high --top-p 0.95
"""

import argparse
import json

from langchain_nova import ChatNova


def demo_basic_parameters():
    """Demonstrate basic parameter usage."""
    print("\n=== Basic Parameters Demo ===\n")

    llm = ChatNova(
        model="nova-pro-v1",
        temperature=0.7,
        max_tokens=150,
        top_p=0.9,
        metadata={"demo": "basic_params", "user_id": "demo_user"},
    )

    messages = [("human", "Explain quantum computing in one sentence.")]

    response = llm.invoke(messages)
    print(f"Response: {response.content}\n")

    if hasattr(response, "usage_metadata"):
        print(f"Tokens used: {response.usage_metadata}")


def demo_reasoning_effort():
    """Demonstrate reasoning effort levels."""
    print("\n=== Reasoning Effort Demo ===\n")

    test_question = "What is 15% of 240?"

    for effort in ["low", "medium", "high"]:
        print(f"\nReasoning effort: {effort}")
        print("-" * 40)

        llm = ChatNova(
            model="nova-pro-v1",
            temperature=0.3,
            reasoning_effort=effort,
            max_tokens=100,
        )

        response = llm.invoke(test_question)
        print(f"Answer: {response.content}")


def demo_token_limits():
    """Demonstrate token limit controls."""
    print("\n=== Token Limits Demo ===\n")

    prompt = "Write a story about a space explorer."

    for max_tokens in [50, 100, 200]:
        print(f"\nMax tokens: {max_tokens}")
        print("-" * 40)

        llm = ChatNova(
            model="nova-pro-v1",
            temperature=0.8,
            max_tokens=max_tokens,
        )

        response = llm.invoke(prompt)
        print(f"{response.content}\n")

        if hasattr(response, "usage_metadata"):
            print(f"Actual tokens: {response.usage_metadata.get('output_tokens')}")


def demo_top_p_sampling():
    """Demonstrate top-p nucleus sampling."""
    print("\n=== Top-P Sampling Demo ===\n")

    prompt = "Generate 3 creative names for a coffee shop."

    for top_p in [0.5, 0.9, 0.99]:
        print(f"\nTop-p: {top_p}")
        print("-" * 40)

        llm = ChatNova(
            model="nova-pro-v1",
            temperature=0.9,
            top_p=top_p,
            max_tokens=100,
        )

        response = llm.invoke(prompt)
        print(response.content)


def demo_streaming_with_usage():
    """Demonstrate streaming with usage metadata."""
    print("\n=== Streaming with Usage Metadata ===\n")

    llm = ChatNova(
        model="nova-pro-v1",
        temperature=0.7,
        max_tokens=150,
        stream_options={"include_usage": True},
    )

    messages = [("human", "Write a haiku about programming.")]

    print("Streaming response:\n")
    for chunk in llm.stream(messages):
        print(chunk.content, end="", flush=True)

    print("\n\nNote: Usage metadata included in final chunk")


def demo_per_call_overrides():
    """Demonstrate per-call parameter overrides."""
    print("\n=== Per-Call Parameter Overrides ===\n")

    llm = ChatNova(
        model="nova-pro-v1",
        temperature=0.5,
        max_tokens=100,
        reasoning_effort="low",
        metadata={"default": "model_level"},
    )

    print("Using model defaults:")
    print("-" * 40)
    response1 = llm.invoke("What is Python?")
    print(f"{response1.content}\n")

    print("\nOverriding max_tokens and reasoning_effort:")
    print("-" * 40)
    response2 = llm.invoke(
        "What is Python?",
        max_tokens=200,
        reasoning_effort="high",
        metadata={"override": "call_level"},
    )
    print(response2.content)


def demo_metadata_tracking():
    """Demonstrate metadata for request tracking."""
    print("\n=== Metadata Tracking Demo ===\n")

    llm = ChatNova(
        model="nova-pro-v1",
        temperature=0.7,
        metadata={
            "application": "demo_app",
            "version": "1.0",
            "environment": "development",
        },
    )

    messages = [("human", "Hello!")]

    print("Metadata sent with request:")
    print(json.dumps(llm.metadata, indent=2))

    response = llm.invoke(messages)
    print(f"\nResponse: {response.content}")


def demo_max_completion_tokens():
    """Demonstrate max_completion_tokens vs max_tokens."""
    print("\n=== max_completion_tokens vs max_tokens ===\n")

    print("Using max_tokens:")
    llm1 = ChatNova(model="nova-pro-v1", max_tokens=50)
    response1 = llm1.invoke("Count from 1 to 100.")
    print(f"{response1.content}\n")

    print("Using max_completion_tokens (OpenAI compatible):")
    llm2 = ChatNova(model="nova-pro-v1", max_completion_tokens=50)
    response2 = llm2.invoke("Count from 1 to 100.")
    print(f"{response2.content}\n")

    print("Note: max_completion_tokens takes precedence when both are set")


def main():
    parser = argparse.ArgumentParser(description="Advanced Nova parameters demo")
    parser.add_argument(
        "--demo",
        type=str,
        choices=[
            "all",
            "basic",
            "reasoning",
            "tokens",
            "sampling",
            "streaming",
            "overrides",
            "metadata",
            "completion_tokens",
        ],
        default="all",
        help="Which demo to run",
    )
    parser.add_argument("--model", type=str, default="nova-pro-v1")
    parser.add_argument("--reasoning", type=str, choices=["low", "medium", "high"])
    parser.add_argument("--top-p", type=float)
    args = parser.parse_args()

    demos = {
        "basic": demo_basic_parameters,
        "reasoning": demo_reasoning_effort,
        "tokens": demo_token_limits,
        "sampling": demo_top_p_sampling,
        "streaming": demo_streaming_with_usage,
        "overrides": demo_per_call_overrides,
        "metadata": demo_metadata_tracking,
        "completion_tokens": demo_max_completion_tokens,
    }

    print("\n" + "=" * 60)
    print("Advanced Nova Parameters Demo")
    print("=" * 60)

    if args.demo == "all":
        for demo in demos.values():
            demo()
    else:
        demos[args.demo]()

    print("\n" + "=" * 60)
    print("Demo Complete")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
