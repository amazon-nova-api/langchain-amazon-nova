"""Reflection pattern using LangGraph and ChatAmazonNova.

This example demonstrates an agent that critiques and improves its own outputs
through multiple iterations of generation and reflection.
"""

import argparse
from typing import List, TypedDict

from langchain_amazon_nova import ChatAmazonNova
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph


class ReflectionState(TypedDict):
    """State for the reflection loop."""

    messages: List
    iterations: int
    max_iterations: int
    initial_query: str


def create_generator(llm, verbose: bool = False):
    """Create the content generator."""

    system_message = SystemMessage(
        content="""You are a helpful assistant that generates content based on user requests.
        If you receive critique or feedback, revise your previous response to address the concerns."""
    )

    def generate(state: ReflectionState) -> dict:
        messages = [system_message] + state["messages"]

        if verbose:
            print(f"\n[Generator] Iteration {state['iterations'] + 1}")
            print(f"[Generator] Generating/revising content...")

        response = llm.invoke(messages)

        return {
            "messages": state["messages"] + [response],
            "iterations": state["iterations"] + 1,
        }

    return generate


def create_reflector(llm, verbose: bool = False):
    """Create the content reflector/critic."""

    system_message = SystemMessage(
        content="""You are a thoughtful critic. Review the assistant's response and provide specific,
        constructive feedback on how it could be improved. Consider:
        - Accuracy and completeness
        - Clarity and structure
        - Tone and style appropriateness
        - Any missing important details

        If the response is excellent and needs no improvements, respond with exactly: "APPROVED"
        Otherwise, provide specific suggestions for improvement."""
    )

    def reflect(state: ReflectionState) -> dict:
        # Get the last AI message to critique
        last_response = state["messages"][-1]

        critique_messages = [
            system_message,
            HumanMessage(content=f"Original request: {state['initial_query']}"),
            HumanMessage(content=f"Assistant's response: {last_response.content}"),
        ]

        if verbose:
            print(f"[Reflector] Reviewing iteration {state['iterations']}...")

        response = llm.invoke(critique_messages)

        if verbose:
            is_approved = "approved" in response.content.lower()
            print(f"[Reflector] {'✓ Approved' if is_approved else '✗ Needs revision'}")

        return {
            "messages": state["messages"] + [HumanMessage(content=f"Critique: {response.content}")],
        }

    return reflect


def should_continue(state: ReflectionState) -> str:
    """Decide whether to continue reflecting or end."""
    # Check if we've hit max iterations
    if state["iterations"] >= state["max_iterations"]:
        return "end"

    # Check if the last message is an approval
    if state["messages"]:
        last_message = state["messages"][-1]
        if isinstance(last_message, HumanMessage) and "APPROVED" in last_message.content:
            return "end"

    # Check if we just generated (odd iteration), go to reflect
    # If we just reflected (even iteration), go to generate
    if state["iterations"] % 2 == 1:
        return "reflect"
    else:
        return "generate"


def create_reflection_graph(llm, max_iterations: int = 3, verbose: bool = False):
    """Create the reflection agent graph."""

    generator = create_generator(llm, verbose)
    reflector = create_reflector(llm, verbose)

    # Create graph
    workflow = StateGraph(ReflectionState)

    # Add nodes
    workflow.add_node("generate", generator)
    workflow.add_node("reflect", reflector)

    # Set entry point
    workflow.add_edge(START, "generate")

    # Add conditional edges
    workflow.add_conditional_edges(
        "generate",
        should_continue,
        {
            "reflect": "reflect",
            "end": END,
        },
    )

    workflow.add_conditional_edges(
        "reflect",
        should_continue,
        {
            "generate": "generate",
            "end": END,
        },
    )

    return workflow.compile()


def main() -> None:
    parser = argparse.ArgumentParser(description="Reflection agent with LangGraph")
    parser.add_argument("--model", type=str, default="nova-pro-v1", help="Nova model to use")
    parser.add_argument("--query", type=str, help="Query to process (non-interactive mode)")
    parser.add_argument(
        "--max-iterations", type=int, default=3, help="Maximum reflection iterations"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    parser.add_argument(
        "--reasoning", type=str, choices=["low", "medium", "high"], help="Reasoning effort"
    )
    parser.add_argument("--top-p", type=float, help="Top-p sampling (0.0-1.0)")
    args = parser.parse_args()

    # Initialize model
    llm = ChatAmazonNova(
        model=args.model,
        temperature=0.7,
        reasoning_effort=args.reasoning,
        top_p=args.top_p,
    )

    # Create reflection agent
    agent = create_reflection_graph(llm, args.max_iterations, args.verbose)

    if args.verbose:
        print(f"\n[DEBUG] Using model: {args.model}")
        print(f"[DEBUG] Max iterations: {args.max_iterations}")
        print("[DEBUG] Agent will generate, reflect, and improve iteratively\n")

    # Run in interactive or non-interactive mode
    if args.query:
        # Non-interactive mode
        print(f"\nQuery: {args.query}\n")

        result = agent.invoke(
            {
                "messages": [HumanMessage(content=args.query)],
                "iterations": 0,
                "max_iterations": args.max_iterations,
                "initial_query": args.query,
            }
        )

        # Find the last AI message (final output)
        final_output = None
        for msg in reversed(result["messages"]):
            if isinstance(msg, AIMessage):
                final_output = msg.content
                break

        print("\n=== Final Output ===")
        print(final_output)
        print(f"\n(Completed in {result['iterations']} iteration(s))\n")
    else:
        # Interactive mode
        print("\n=== Reflection Agent ===")
        print("I will generate content, critique it, and improve it iteratively.")
        print(f"Maximum {args.max_iterations} reflection iterations per query.")
        print("\nExample queries:")
        print("  - Write a short essay on renewable energy")
        print("  - Explain quantum computing to a 10-year-old")
        print("  - Write a professional email declining a job offer")
        print("\nType 'exit' to quit.\n")

        while True:
            user_input = input("Query: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ["exit", "quit"]:
                print("\nGoodbye!\n")
                break

            try:
                result = agent.invoke(
                    {
                        "messages": [HumanMessage(content=user_input)],
                        "iterations": 0,
                        "max_iterations": args.max_iterations,
                        "initial_query": user_input,
                    }
                )

                # Find the last AI message (final output)
                final_output = None
                for msg in reversed(result["messages"]):
                    if isinstance(msg, AIMessage):
                        final_output = msg.content
                        break

                print("\n=== Final Output ===")
                print(final_output)
                print(f"\n(Completed in {result['iterations']} iteration(s))\n")
            except Exception as e:
                print(f"\nError: {e}\n")


if __name__ == "__main__":
    main()
