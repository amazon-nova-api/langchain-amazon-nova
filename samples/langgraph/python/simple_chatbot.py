"""Simple stateful chatbot using LangGraph and ChatAmazonNova.

This example demonstrates a basic chatbot that maintains conversation history
using LangGraph's state management.
"""

import argparse
from typing import Annotated, TypedDict

from langchain_amazon_nova import ChatAmazonNova
from langchain_core.messages import HumanMessage, SystemMessage, trim_messages
from langgraph.graph import START, MessagesState, StateGraph
from langgraph.graph.message import add_messages


def call_model(state: MessagesState, llm, verbose: bool = False) -> dict:
    """Call the model with conversation history."""
    messages = state["messages"]

    if verbose:
        print(f"\n[Chatbot] Processing conversation with {len(messages)} messages")

    # Add system message if this is the first turn
    if len(messages) == 1:
        system_msg = SystemMessage(
            content="You are a helpful assistant. Be conversational and remember context from earlier in the conversation."
        )
        messages = [system_msg] + messages

    response = llm.invoke(messages)

    return {"messages": [response]}


def create_chatbot_graph(llm, verbose: bool = False):
    """Create a simple stateful chatbot graph."""
    # Create the graph
    workflow = StateGraph(MessagesState)

    # Add single node that calls the model
    workflow.add_node("chatbot", lambda state: call_model(state, llm, verbose))

    # Set entry point
    workflow.set_entry_point("chatbot")
    workflow.set_finish_point("chatbot")

    return workflow.compile()


def main() -> None:
    parser = argparse.ArgumentParser(description="Simple stateful chatbot with LangGraph")
    parser.add_argument("--model", type=str, default="nova-pro-v1", help="Nova model to use")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    parser.add_argument(
        "--reasoning", type=str, choices=["low", "medium", "high"], help="Reasoning effort"
    )
    parser.add_argument("--top-p", type=float, help="Top-p sampling (0.0-1.0)")
    parser.add_argument(
        "--max-history", type=int, default=10, help="Maximum messages to keep in history"
    )
    args = parser.parse_args()

    # Initialize model
    llm = ChatAmazonNova(
        model=args.model,
        temperature=0.7,
        reasoning_effort=args.reasoning,
        top_p=args.top_p,
    )

    # Create chatbot
    chatbot = create_chatbot_graph(llm, args.verbose)

    # Initialize conversation state
    conversation_state = {"messages": []}

    if args.verbose:
        print(f"\n[DEBUG] Using model: {args.model}")
        print(f"[DEBUG] Max history: {args.max_history} messages")
        print("[DEBUG] Conversation state persists across turns\n")

    print("\n=== Simple Chatbot ===")
    print("Chat with me! I'll remember our conversation.")
    print("Commands: /exit to quit, /clear to reset conversation, /history to show history")
    print()

    while True:
        user_input = input("You: ").strip()

        if not user_input:
            continue

        # Handle commands
        if user_input.lower() in ["/exit", "/quit"]:
            print("\nGoodbye!\n")
            break

        if user_input.lower() == "/clear":
            conversation_state = {"messages": []}
            print("\n[Conversation cleared]\n")
            continue

        if user_input.lower() == "/history":
            print(f"\n[Conversation history: {len(conversation_state['messages'])} messages]")
            for i, msg in enumerate(conversation_state["messages"]):
                role = "User" if msg.type == "human" else "Assistant"
                content = msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
                print(f"{i + 1}. {role}: {content}")
            print()
            continue

        try:
            # Add user message to state
            conversation_state["messages"].append(HumanMessage(content=user_input))

            # Trim history if needed (keep most recent messages)
            if (
                len(conversation_state["messages"]) > args.max_history * 2
            ):  # *2 for user+assistant pairs
                # Keep system message if present, then most recent messages
                if conversation_state["messages"][0].type == "system":
                    system_msg = conversation_state["messages"][0]
                    conversation_state["messages"] = [system_msg] + conversation_state["messages"][
                        -(args.max_history * 2) :
                    ]
                else:
                    conversation_state["messages"] = conversation_state["messages"][
                        -(args.max_history * 2) :
                    ]

                if args.verbose:
                    print("[DEBUG] Trimmed conversation history")

            # Get response
            result = chatbot.invoke(conversation_state)

            # Update state with response
            conversation_state = result

            # Print response
            assistant_message = result["messages"][-1]
            print(f"\nAssistant: {assistant_message.content}\n")

        except Exception as e:
            print(f"\nError: {e}\n")
            # Remove the failed user message
            if (
                conversation_state["messages"]
                and conversation_state["messages"][-1].content == user_input
            ):
                conversation_state["messages"].pop()


if __name__ == "__main__":
    main()
