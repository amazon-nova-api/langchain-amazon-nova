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
"""

import argparse
import sys
from typing import List

from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage

from langchain_nova import ChatNova


def print_banner() -> None:
    """Print welcome banner."""
    print("\n" + "=" * 60)
    print("  Amazon Nova Interactive Chat")
    print("=" * 60)
    print("  Commands:")
    print("    /help    - Show this help message")
    print("    /clear   - Clear conversation history")
    print("    /history - Show conversation history")
    print("    /exit    - Exit the chat")
    print("=" * 60 + "\n")


def print_help() -> None:
    """Print help message."""
    print("\nAvailable commands:")
    print("  /help    - Show this help message")
    print("  /clear   - Clear conversation history")
    print("  /history - Show conversation history")
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


def chat_streaming(
    llm: ChatNova,
    messages: List[BaseMessage],
    user_input: str,
) -> str:
    """Handle streaming chat interaction."""
    messages.append(HumanMessage(content=user_input))

    print("\nAssistant: ", end="", flush=True)

    full_response = ""
    for chunk in llm.stream(messages):
        content = chunk.content
        if content:
            print(content, end="", flush=True)
            full_response += content

    print("\n")
    return full_response


def chat_non_streaming(
    llm: ChatNova,
    messages: List[BaseMessage],
    user_input: str,
) -> str:
    """Handle non-streaming chat interaction."""
    messages.append(HumanMessage(content=user_input))

    response = llm.invoke(messages)

    print(f"\nAssistant: {response.content}\n")

    return response.content


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

    args = parser.parse_args()

    # Initialize the model
    try:
        llm = ChatNova(
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        print(f"\nInitialized {args.model} (streaming: {args.streaming})")
    except Exception as e:
        print(f"\nError initializing Nova: {e}")
        print("\nMake sure you have set the following environment variables:")
        print("  - NOVA_API_KEY")
        print("  - NOVA_BASE_URL (optional)\n")
        sys.exit(1)

    # Initialize conversation with system message
    messages: List[BaseMessage] = [SystemMessage(content=args.system_message)]

    print_banner()

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
                    print("\nGoodbye!\n")
                    break

                elif command == "/help":
                    print_help()
                    continue

                elif command == "/clear":
                    messages = [SystemMessage(content=args.system_message)]
                    print("\nConversation history cleared.\n")
                    continue

                elif command == "/history":
                    print_history(messages)
                    continue

                else:
                    print(f"\nUnknown command: {user_input}")
                    print("Type /help for available commands.\n")
                    continue

            # Process chat message
            if args.streaming:
                response_content = chat_streaming(llm, messages, user_input)
            else:
                response_content = chat_non_streaming(llm, messages, user_input)

            # Add assistant response to history
            from langchain_core.messages import AIMessage
            messages.append(AIMessage(content=response_content))

        except KeyboardInterrupt:
            print("\n\nInterrupted. Type /exit to quit or continue chatting.\n")
            continue

        except Exception as e:
            print(f"\nError: {e}\n")
            continue


if __name__ == "__main__":
    main()
