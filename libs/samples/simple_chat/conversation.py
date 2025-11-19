"""Multi-turn conversation example with ChatNova."""

import argparse
import time

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from langchain_nova import ChatNova


def main():
    parser = argparse.ArgumentParser(
        description="Multi-turn conversation example with ChatNova"
    )
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
        print(f"[DEBUG] Building multi-turn conversation with context")

    # Start with system message
    messages = [
        SystemMessage(content="You are a helpful assistant. Keep responses concise."),
    ]

    print("\n=== Multi-turn Conversation ===\n")

    total_time = 0.0

    # Turn 1
    user_input = "What is Python?"
    messages.append(HumanMessage(content=user_input))
    print(f"User: {user_input}")

    if args.verbose:
        print(f"[DEBUG] Turn 1: {len(messages)} messages in context")

    start = time.time()
    response = llm.invoke(messages)
    elapsed = time.time() - start
    total_time += elapsed

    messages.append(AIMessage(content=response.content))
    print(f"Assistant: {response.content}\n")

    if args.verbose:
        print(f"[DEBUG] Response time: {elapsed:.2f}s\n")

    # Turn 2 - with context
    user_input = "What are its main uses?"
    messages.append(HumanMessage(content=user_input))
    print(f"User: {user_input}")

    if args.verbose:
        print(f"[DEBUG] Turn 2: {len(messages)} messages in context")

    start = time.time()
    response = llm.invoke(messages)
    elapsed = time.time() - start
    total_time += elapsed

    messages.append(AIMessage(content=response.content))
    print(f"Assistant: {response.content}\n")

    if args.verbose:
        print(f"[DEBUG] Response time: {elapsed:.2f}s\n")

    # Turn 3 - with full context
    user_input = "How does it compare to JavaScript?"
    messages.append(HumanMessage(content=user_input))
    print(f"User: {user_input}")

    if args.verbose:
        print(f"[DEBUG] Turn 3: {len(messages)} messages in context")

    start = time.time()
    response = llm.invoke(messages)
    elapsed = time.time() - start
    total_time += elapsed

    messages.append(AIMessage(content=response.content))
    print(f"Assistant: {response.content}\n")

    if args.verbose:
        print(f"[DEBUG] Response time: {elapsed:.2f}s")
        print(f"[DEBUG] Total conversation time: {total_time:.2f}s")
        print(f"[DEBUG] Average response time: {total_time / 3:.2f}s")

    print(f"=== Conversation complete ({len(messages)} messages total) ===\n")


if __name__ == "__main__":
    main()
