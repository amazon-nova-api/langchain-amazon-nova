"""Basic ChatNova usage example."""

import argparse
import time

from langchain_nova import ChatNova


def main():
    parser = argparse.ArgumentParser(description="Basic ChatNova usage example")
    parser.add_argument(
        "--model",
        type=str,
        default="nova-pro-v1",
        help="Nova model to use (default: nova-pro-v1)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output",
    )
    parser.add_argument(
        "--reasoning",
        type=str,
        choices=["low", "medium", "high"],
        help="Reasoning effort level",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        help="Top-p sampling (0.0-1.0)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Maximum tokens to generate (default: 2048)",
    )
    args = parser.parse_args()

    # Initialize the model
    if args.verbose:
        print(f"\n[DEBUG] Initializing ChatNova with model: {args.model}")

    llm = ChatNova(
        model=args.model,
        temperature=0.7,
        max_tokens=args.max_tokens,
        reasoning_effort=args.reasoning,
        top_p=args.top_p,
        metadata={"example": "basic_chat", "version": "1.0"},
    )

    if args.verbose:
        print(f"[DEBUG] Model initialized successfully")
        print(f"[DEBUG] Base URL: {llm.base_url}")
        print(f"[DEBUG] Temperature: {llm.temperature}")
        print(f"[DEBUG] Max tokens: {llm.max_tokens}")
        if args.reasoning:
            print(f"[DEBUG] Reasoning effort: {args.reasoning}")
        if args.top_p:
            print(f"[DEBUG] Top-p: {args.top_p}")

    # Simple invoke
    messages = [
        ("system", "You are a helpful assistant."),
        ("human", "What is the capital of France?"),
    ]

    if args.verbose:
        print(f"\n[DEBUG] Sending {len(messages)} messages")
        print(f"[DEBUG] Message content: {messages}")

    start_time = time.time()
    response = llm.invoke(messages)
    elapsed = time.time() - start_time

    print(f"\nResponse: {response.content}")

    # Check usage metadata
    if hasattr(response, "usage_metadata"):
        print(f"\nTokens used: {response.usage_metadata}")

    if args.verbose:
        print(f"\n[DEBUG] Response time: {elapsed:.2f}s")
        print(f"[DEBUG] Response metadata: {response.response_metadata}")


if __name__ == "__main__":
    main()
