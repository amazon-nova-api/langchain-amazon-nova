"""Agent with tools using LangGraph and ChatAmazonNova.

This example demonstrates a basic agent that can use tools to answer questions.
The agent reasons about which tools to use and executes them in a loop.
"""

import argparse
from typing import Annotated, Literal, TypedDict

from langchain_amazon_nova import ChatAmazonNova
from langchain_core.messages import BaseMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode


# Define tools
@tool
def get_weather(location: str, unit: Literal["celsius", "fahrenheit"] = "fahrenheit") -> str:
    """Get the current weather for a location.

    Args:
        location: The city and state, e.g. San Francisco, CA
        unit: The temperature unit to use
    """
    # Simulated weather data
    weather_data = {
        "San Francisco, CA": {"temp": 65, "condition": "cloudy"},
        "Seattle, WA": {"temp": 52, "condition": "rainy"},
        "New York, NY": {"temp": 45, "condition": "partly cloudy"},
        "Miami, FL": {"temp": 78, "condition": "sunny"},
    }

    data = weather_data.get(location, {"temp": 70, "condition": "unknown"})

    if unit == "celsius":
        temp = round((data["temp"] - 32) * 5 / 9, 1)
        unit_str = "°C"
    else:
        temp = data["temp"]
        unit_str = "°F"

    return f"Weather in {location}: {data['condition']}, {temp}{unit_str}"


@tool
def calculate(operation: str, a: float, b: float) -> str:
    """Perform a mathematical calculation.

    Args:
        operation: The operation to perform (add, subtract, multiply, divide, power)
        a: The first number
        b: The second number
    """
    operations = {
        "add": a + b,
        "subtract": a - b,
        "multiply": a * b,
        "divide": a / b if b != 0 else "Error: Division by zero",
        "power": a**b,
    }

    if operation not in operations:
        return f"Error: Unknown operation '{operation}'"

    result = operations[operation]
    return f"{a} {operation} {b} = {result}" if not isinstance(result, str) else result


@tool
def search_web(query: str) -> str:
    """Search the web for information.

    Args:
        query: The search query
    """
    # Simulated search results
    search_results = {
        "python": "Python is a high-level programming language known for its readability.",
        "nova": "Amazon Nova is a family of foundation models from AWS.",
        "langchain": "LangChain is a framework for developing LLM-powered applications.",
        "langgraph": "LangGraph is a library for building stateful multi-actor applications.",
    }

    for key, value in search_results.items():
        if key in query.lower():
            return value

    return f"Search results for '{query}': Information not found in simulated database."


# Define the agent function
def call_model(state: MessagesState, llm_with_tools, verbose: bool = False) -> dict:
    """Call the model with tools bound."""
    messages = state["messages"]

    if verbose:
        print(f"\n[Agent] Processing {len(messages)} messages")
        print(f"[Agent] Last message: {messages[-1].content[:100]}...")

    response = llm_with_tools.invoke(messages)

    if verbose and hasattr(response, "tool_calls") and response.tool_calls:
        print(f"[Agent] Calling {len(response.tool_calls)} tool(s):")
        for tc in response.tool_calls:
            print(f"  - {tc['name']}: {tc['args']}")

    return {"messages": [response]}


# Define routing function
def should_continue(state: MessagesState) -> Literal["tools", "end"]:
    """Determine if we should continue to tools or end."""
    messages = state["messages"]
    last_message = messages[-1]

    # If there are tool calls, continue to tools
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"

    # Otherwise, end
    return "end"


def create_agent_graph(llm, tools, verbose: bool = False):
    """Create the agent graph."""
    # Bind tools to model
    llm_with_tools = llm.bind_tools(tools)

    # Create the graph
    workflow = StateGraph(MessagesState)

    # Add nodes
    workflow.add_node("agent", lambda state: call_model(state, llm_with_tools, verbose))
    workflow.add_node("tools", ToolNode(tools))

    # Set entry point
    workflow.add_edge(START, "agent")

    # Add conditional edges
    workflow.add_conditional_edges(
        "agent",
        should_continue,
        {
            "tools": "tools",
            "end": END,
        },
    )

    # After tools, always go back to agent
    workflow.add_edge("tools", "agent")

    return workflow.compile()


def main() -> None:
    parser = argparse.ArgumentParser(description="Agent with tools using LangGraph")
    parser.add_argument("--model", type=str, default="nova-pro-v1", help="Nova model to use")
    parser.add_argument("--query", type=str, help="Query to run (non-interactive mode)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    parser.add_argument(
        "--reasoning", type=str, choices=["low", "medium", "high"], help="Reasoning effort"
    )
    parser.add_argument("--top-p", type=float, help="Top-p sampling (0.0-1.0)")
    args = parser.parse_args()

    # Initialize model
    llm = ChatAmazonNova(
        model=args.model,
        temperature=0,  # Use 0 for consistent tool calling
        reasoning_effort=args.reasoning,
        top_p=args.top_p,
    )

    # Define tools
    tools = [get_weather, calculate, search_web]

    # Create agent
    agent = create_agent_graph(llm, tools, args.verbose)

    if args.verbose:
        print(f"\n[DEBUG] Using model: {args.model}")
        print(f"[DEBUG] Available tools: {[t.name for t in tools]}")
        print()

    # Run in interactive or non-interactive mode
    if args.query:
        # Non-interactive mode
        print(f"\nQuery: {args.query}\n")
        result = agent.invoke({"messages": [HumanMessage(content=args.query)]})
        print(f"Answer: {result['messages'][-1].content}\n")
    else:
        # Interactive mode
        print("\n=== Agent with Tools ===")
        print("Ask questions and I'll use tools to help answer them.")
        print("Available tools: weather, calculate, search_web")
        print("Type 'exit' to quit.\n")

        while True:
            user_input = input("You: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ["exit", "quit"]:
                print("\nGoodbye!\n")
                break

            try:
                result = agent.invoke({"messages": [HumanMessage(content=user_input)]})
                print(f"\nAssistant: {result['messages'][-1].content}\n")
            except Exception as e:
                print(f"\nError: {e}\n")


if __name__ == "__main__":
    main()
