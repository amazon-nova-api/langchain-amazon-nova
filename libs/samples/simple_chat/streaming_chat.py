"""Streaming ChatNova example."""

import argparse
import time

from langchain_nova import ChatNova


def main():
    parser = argparse.ArgumentParser(description="Streaming ChatNova example")
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
    args = parser.parse_args()

    # Initialize the model
    if args.verbose:
        print(f"\n[DEBUG] Initializing ChatNova with model: {args.model}")

    llm = ChatNova(
        model=args.model,
        temperature=0.7,
        max_tokens=2048,
    )

    if args.verbose:
        print(f"[DEBUG] Model initialized successfully")
        print(f"[DEBUG] Streaming enabled")

    # Stream responses
    messages = [
        ("system", "You are a helpful assistant."),
        ("human", "Tell me a short story about a robot learning to paint."),
    ]

    if args.verbose:
        print(f"[DEBUG] Sending {len(messages)} messages\n")

    print("Streaming response:\n")

    start_time = time.time()
    chunk_count = 0
    for chunk in llm.stream(messages):
        print(chunk.content, end="", flush=True)
        chunk_count += 1
    elapsed = time.time() - start_time

    if args.verbose:
        print(f"\n\n[DEBUG] Received {chunk_count} chunks in {elapsed:.2f}s")

    print("\n\nDone!")


if __name__ == "__main__":
    main()
