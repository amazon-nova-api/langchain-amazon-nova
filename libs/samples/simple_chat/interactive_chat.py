"""Interactive chat interface for Amazon Nova using LangChain.

This script provides a simple command-line chat interface for interacting
with Amazon Nova models through the LangChain integration.

Requirements:
    - NOVA_API_KEY environment variable set
    - NOVA_BASE_URL environment variable set (optional)

Usage:
    python interactive_chat.py
    python interactive_chat.py --model nova-lite-v1
    python interactive_chat.py --streaming
    python interactive_chat.py --verbose
"""

import argparse
import sys
import time
from typing import List

from langchain_core.callbacks import StreamingStdOutCallbackHandler
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage

from langchain_nova import ChatNova


def print_banner(verbose: bool = False) -> None:
    """Print welcome banner."""
    print("\n" + "=" * 60)
    print("  Amazon Nova Interactive Chat")
    print("=" * 60)
    print("  Commands:")
    print("    /help    - Show this help message")
    print("    /clear   - Clear conversation history")
    print("    /history - Show conversation history")
    print("    /stats   - Show conversation statistics")
    print("    /exit    - Exit the chat")
    if verbose:
        print("\n  Verbose mode: ON")
    print("=" * 60 + "\n")


def print_help() -> None:
    """Print help message."""
    print("\nAvailable commands:")
    print("  /help    - Show this help message")
    print("  /clear   - Clear conversation history")
    print("  /history - Show conversation history")
    print("  /stats   - Show conversation statistics")
    print("  /exit    - Exit the chat")
    print()


def print_history(messages: List[BaseMessage]) -> None:
    """Print conversation history."""
    print("\nConversation History:")
    print("-" * 60)
    for i, msg in enumerate(messages, 1):
        role = msg.type.upper()
        content = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
        print(f"{i}. [{role}] {content}")
    print("-" * 60 + "\n")


def print_stats(messages: List[BaseMessage], total_time: float) -> None:
    """Print conversation statistics."""
    user_msgs = sum(1 for m in messages if m.type == "human")
    ai_msgs = sum(1 for m in messages if m.type == "ai")
    total_chars = sum(len(m.content) for m in messages)

    print("\nConversation Statistics:")
    print("-" * 60)
    print(f"  Total messages: {len(messages)}")
    print(f"  User messages: {user_msgs}")
    print(f"  Assistant messages: {ai_msgs}")
    print(f"  Total characters: {total_chars}")
    print(f"  Session time: {total_time:.1f}s")
    if ai_msgs > 0:
        print(f"  Avg response time: {total_time / ai_msgs:.1f}s")
    print("-" * 60 + "\n")


def chat_streaming(
    llm: ChatNova,
    messages: List[BaseMessage],
    user_input: str,
    verbose: bool = False,
) -> tuple[str, float]:
    """Handle streaming chat interaction."""
    messages.append(HumanMessage(content=user_input))

    if verbose:
        print(f"\n[DEBUG] Sending {len(messages)} messages to model")
        print(f"[DEBUG] User message length: {len(user_input)} chars")

    start_time = time.time()
    print("\nAssistant: ", end="", flush=True)

    full_response = ""
    chunk_count = 0
    for chunk in llm.stream(messages):
        content = chunk.content
        if content:
            print(content, end="", flush=True)
            full_response += content
            chunk_count += 1

    elapsed = time.time() - start_time

    if verbose:
        print(f"\n[DEBUG] Received {chunk_count} chunks in {elapsed:.2f}s")
        print(f"[DEBUG] Response length: {len(full_response)} chars")
        print(f"[DEBUG] Chars/sec: {len(full_response) / elapsed:.1f}")

    print("\n")
    return full_response, elapsed


def chat_non_streaming(
    llm: ChatNova,
    messages: List[BaseMessage],
    user_input: str,
    verbose: bool = False,
) -> tuple[str, float]:
    """Handle non-streaming chat interaction."""
    messages.append(HumanMessage(content=user_input))

    if verbose:
        print(f"\n[DEBUG] Sending {len(messages)} messages to model")
        print(f"[DEBUG] User message length: {len(user_input)} chars")

    start_time = time.time()
    response = llm.invoke(messages)
    elapsed = time.time() - start_time

    print(f"\nAssistant: {response.content}\n")

    if verbose:
        print(f"[DEBUG] Response time: {elapsed:.2f}s")
        print(f"[DEBUG] Response length: {len(response.content)} chars")
        if hasattr(response, "response_metadata"):
            print(f"[DEBUG] Metadata: {response.response_metadata}")
        if hasattr(response, "usage_metadata"):
            usage = response.usage_metadata
            print(f"[DEBUG] Usage: {usage}")
            if "total_tokens" in usage and elapsed > 0:
                print(f"[DEBUG] Tokens/sec: {usage['total_tokens'] / elapsed:.1f}")

    return response.content, elapsed


def main() -> None:
    """Run interactive chat."""
    parser = argparse.ArgumentParser(
        description="Interactive chat with Amazon Nova"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="nova-pro-v1",
        help="Nova model to use (default: nova-pro-v1)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for sampling (default: 0.7)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Maximum tokens to generate (default: 2048)",
    )
    parser.add_argument(
        "--streaming",
        action="store_true",
        help="Enable streaming responses",
    )
    parser.add_argument(
        "--system-message",
        type=str,
        default="You are a helpful AI assistant.",
        help="System message to set assistant behavior",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output with detailed information",
    )

    args = parser.parse_args()

    # Initialize the model
    try:
        llm = ChatNova(
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        if args.verbose:
            print(f"\n[DEBUG] Model initialized:")
            print(f"[DEBUG]   Model: {args.model}")
            print(f"[DEBUG]   Temperature: {args.temperature}")
            print(f"[DEBUG]   Max tokens: {args.max_tokens}")
            print(f"[DEBUG]   Base URL: {llm.base_url}")
        else:
            print(f"\nInitialized {args.model} (streaming: {args.streaming})")
    except Exception as e:
        print(f"\nError initializing Nova: {e}")
        print("\nMake sure you have set the following environment variables:")
        print("  - NOVA_API_KEY")
        print("  - NOVA_BASE_URL (optional)\n")
        sys.exit(1)

    # Initialize conversation with system message
    messages: List[BaseMessage] = [SystemMessage(content=args.system_message)]

    if args.verbose:
        print(f"[DEBUG] System message: {args.system_message}")

    print_banner(args.verbose)

    # Track session time
    session_start = time.time()
    total_response_time = 0.0

    # Main chat loop
    while True:
        try:
            # Get user input
            user_input = input("You: ").strip()

            if not user_input:
                continue

            # Handle commands
            if user_input.startswith("/"):
                command = user_input.lower()

                if command == "/exit":
                    session_time = time.time() - session_start
                    if args.verbose:
                        print(f"\n[DEBUG] Session duration: {session_time:.1f}s")
                        print(f"[DEBUG] Total response time: {total_response_time:.1f}s")
                    print("\nGoodbye!\n")
                    break

                elif command == "/help":
                    print_help()
                    continue

                elif command == "/clear":
                    messages = [SystemMessage(content=args.system_message)]
                    total_response_time = 0.0
                    print("\nConversation history cleared.\n")
                    if args.verbose:
                        print("[DEBUG] Message history reset")
                    continue

                elif command == "/history":
                    print_history(messages)
                    continue

                elif command == "/stats":
                    session_time = time.time() - session_start
                    print_stats(messages, session_time)
                    continue

                else:
                    print(f"\nUnknown command: {user_input}")
                    print("Type /help for available commands.\n")
                    continue

            # Process chat message
            if args.streaming:
                response_content, elapsed = chat_streaming(
                    llm, messages, user_input, args.verbose
                )
            else:
                response_content, elapsed = chat_non_streaming(
                    llm, messages, user_input, args.verbose
                )

            total_response_time += elapsed

            # Add assistant response to history
            messages.append(AIMessage(content=response_content))

        except KeyboardInterrupt:
            print("\n\nInterrupted. Type /exit to quit or continue chatting.\n")
            continue

        except Exception as e:
            print(f"\nError: {e}\n")
            if args.verbose:
                import traceback
                print("[DEBUG] Full traceback:")
                traceback.print_exc()
            continue


if __name__ == "__main__":
    main()
