"""Async ChatAmazonNova usage example."""

import argparse
import asyncio
import time

from langchain_amazon_nova import ChatAmazonNova


async def main(args):
    # Initialize the model
    if args.verbose:
        print(f"\n[DEBUG] Initializing ChatAmazonNova with model: {args.model}")

    llm = ChatAmazonNova(
        model=args.model,
        temperature=0.7,
        max_tokens=2048,
    )

    if args.verbose:
        print(f"[DEBUG] Model initialized successfully")
        print(f"[DEBUG] Base URL: {llm.base_url}")
        print(f"[DEBUG] Using async operations")

    # Async invoke
    messages = [
        ("system", "You are a helpful assistant."),
        ("human", "What are three benefits of async programming?"),
    ]

    if args.verbose:
        print(f"\n[DEBUG] Making async invoke request with {len(messages)} messages")

    print("\nMaking async request...")
    start_time = time.time()
    response = await llm.ainvoke(messages)
    elapsed = time.time() - start_time

    print(f"Response: {response.content}")

    if args.verbose:
        print(f"\n[DEBUG] Async invoke took {elapsed:.2f}s")
        if hasattr(response, "usage_metadata"):
            print(f"[DEBUG] Usage: {response.usage_metadata}")

    # Async streaming
    print("\n\nStreaming async response:")
    stream_messages = [
        ("system", "You are a helpful assistant."),
        ("human", "Count from 1 to 5 with a brief description of each number."),
    ]

    if args.verbose:
        print(f"[DEBUG] Starting async stream with {len(stream_messages)} messages\n")

    start_time = time.time()
    chunk_count = 0
    async for chunk in llm.astream(stream_messages):
        print(chunk.content, end="", flush=True)
        chunk_count += 1
    elapsed = time.time() - start_time

    if args.verbose:
        print(
            f"\n\n[DEBUG] Async stream received {chunk_count} chunks in {elapsed:.2f}s"
        )

    print("\n\nDone!\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Async ChatAmazonNova usage example")
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

    if args.verbose:
        print(f"[DEBUG] Starting async example with model: {args.model}")

    asyncio.run(main(args))
