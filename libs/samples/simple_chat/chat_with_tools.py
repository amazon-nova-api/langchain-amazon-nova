"""Interactive chat with tool use for Amazon Nova.

Requirements:
    - NOVA_API_KEY environment variable set
    - NOVA_BASE_URL environment variable set

Usage:
    python chat_with_tools.py
    python chat_with_tools.py --model nova-lite-v1
"""

import argparse
import json
from datetime import datetime
from typing import Literal

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_nova import ChatNova


@tool
def get_current_weather(
    location: str, unit: Literal["celsius", "fahrenheit"] = "fahrenheit"
) -> str:
    """Get the current weather for a location.

    Args:
        location: The city and state, e.g. San Francisco, CA
        unit: The temperature unit to use. Defaults to fahrenheit.
    """
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
def get_current_time(timezone: str = "UTC") -> str:
    """Get the current time in a specific timezone.

    Args:
        timezone: The timezone name (e.g., UTC, America/New_York, Europe/London, Asia/Tokyo)
    """
    timezone_offsets = {
        "UTC": 0,
        "America/New_York": -5,
        "America/Los_Angeles": -8,
        "Europe/London": 0,
        "Asia/Tokyo": 9,
    }

    current_time = datetime.now()
    offset = timezone_offsets.get(timezone, 0)
    adjusted_time = current_time.replace(hour=(current_time.hour + offset) % 24)

    return f"Current time in {timezone}: {adjusted_time.strftime('%I:%M %p on %B %d, %Y')}"


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
        return f"Error: Unknown operation '{operation}'. Available: {', '.join(operations.keys())}"

    result = operations[operation]
    return f"{a} {operation} {b} = {result}" if not isinstance(result, str) else result


def main():
    """Run interactive chat with tools."""
    parser = argparse.ArgumentParser(description="Interactive chat with Amazon Nova and tools")
    parser.add_argument("--model", default="nova-pro-v1", help="Nova model to use")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for sampling")
    args = parser.parse_args()

    # Initialize model with tools
    tools = [get_current_weather, get_current_time, calculate]
    llm = ChatNova(model=args.model, temperature=args.temperature).bind_tools(tools)

    # Create tool map for execution
    tool_map = {t.name: t for t in tools}

    # Initialize conversation
    messages = [
        SystemMessage(
            content="You are a helpful assistant with access to tools for weather, time, and calculations."
        )
    ]

    print(f"\nNova Chat with Tools ({args.model})")
    print("Commands: /exit to quit, /clear to reset\n")

    while True:
        user_input = input("You: ").strip()

        if not user_input:
            continue

        if user_input == "/exit":
            print("\nGoodbye!\n")
            break

        if user_input == "/clear":
            messages = [messages[0]]  # Keep system message
            print("\nConversation cleared.\n")
            continue

        # Add user message
        messages.append(HumanMessage(content=user_input))

        # Tool calling loop
        max_iterations = 5
        for _ in range(max_iterations):
            response = llm.invoke(messages)
            messages.append(response)

            # Check for tool calls
            if not response.tool_calls:
                # No tools needed, show response
                print(f"\nAssistant: {response.content}\n")
                break

            # Execute tool calls
            print(f"\n[Using {len(response.tool_calls)} tool(s)...]")
            for tool_call in response.tool_calls:
                tool_name = tool_call["name"]
                tool_args = tool_call["args"]
                tool_id = tool_call["id"]

                print(f"  → {tool_name}({json.dumps(tool_args)})")

                # Execute tool
                try:
                    result = tool_map[tool_name].invoke(tool_args)
                except Exception as e:
                    result = f"Error: {str(e)}"

                messages.append(ToolMessage(content=str(result), tool_call_id=tool_id))


if __name__ == "__main__":
    main()
