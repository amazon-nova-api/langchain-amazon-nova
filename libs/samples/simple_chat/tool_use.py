"""Tool use example with ChatNova.

Demonstrates function calling / tool use patterns.
"""

import argparse
import json
from datetime import datetime
from typing import Literal

from langchain_core.messages import HumanMessage, ToolMessage
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

    return f"The weather in {location} is {data['condition']} with a temperature of {temp}{unit_str}."


@tool
def get_current_time(timezone: str = "UTC") -> str:
    """Get the current time in a specific timezone.

    Args:
        timezone: The timezone name (e.g., UTC, America/New_York, Europe/London)
    """
    # Simplified - in real code would use pytz or zoneinfo
    current_time = datetime.now()
    return f"The current time in {timezone} is {current_time.strftime('%I:%M %p')}"


@tool
def calculate(operation: str, a: float, b: float) -> str:
    """Perform a mathematical calculation.

    Args:
        operation: The operation to perform (add, subtract, multiply, divide)
        a: The first number
        b: The second number
    """
    operations = {
        "add": a + b,
        "subtract": a - b,
        "multiply": a * b,
        "divide": a / b if b != 0 else "Error: Division by zero",
    }

    result = operations.get(operation, "Error: Unknown operation")
    return f"{a} {operation} {b} = {result}"


def main():
    parser = argparse.ArgumentParser(description="Tool use with Nova")
    parser.add_argument("--model", type=str, default="nova-pro-v1")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    llm = ChatNova(model=args.model, temperature=0)

    # Bind tools to the model
    tools = [get_current_weather, get_current_time, calculate]
    llm_with_tools = llm.bind_tools(tools)

    if args.verbose:
        print(f"\n[DEBUG] Using model: {args.model}")
        print(f"[DEBUG] Bound {len(tools)} tools: {[t.name for t in tools]}\n")

    print("=== 1. Simple Tool Call ===\n")

    messages = [HumanMessage(content="What's the weather in Seattle?")]

    response = llm_with_tools.invoke(messages)

    if args.verbose:
        print(f"[DEBUG] Response type: {type(response)}")
        print(f"[DEBUG] Has tool calls: {bool(response.tool_calls)}\n")

    print(f"User: {messages[0].content}")

    if response.tool_calls:
        print(f"Assistant wants to call tool: {response.tool_calls[0]['name']}")
        print(f"  Arguments: {json.dumps(response.tool_calls[0]['args'], indent=2)}\n")

        # Execute the tool
        tool_call = response.tool_calls[0]
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]

        # Find and execute the tool
        tool_map = {t.name: t for t in tools}
        if tool_name in tool_map:
            tool_result = tool_map[tool_name].invoke(tool_args)
            print(f"Tool result: {tool_result}\n")

            # Send result back to model
            messages.append(response)
            messages.append(
                ToolMessage(content=tool_result, tool_call_id=tool_call["id"])
            )

            if args.verbose:
                print(f"[DEBUG] Messages to send: {len(messages)}")
                for i, msg in enumerate(messages):
                    print(
                        f"[DEBUG]   {i}. {msg.type}: content='{msg.content[:50] if msg.content else '(empty)'}...'"
                    )

            final_response = llm_with_tools.invoke(messages)
            print(f"Final answer: {final_response.content}\n")
    else:
        print(f"Assistant: {response.content}\n")

    print("=== 2. Multiple Tool Calls ===\n")

    messages = [
        HumanMessage(
            content="What's the weather in San Francisco and Miami? Also what time is it?"
        )
    ]

    print(f"User: {messages[0].content}\n")

    response = llm_with_tools.invoke(messages)

    if response.tool_calls:
        print(f"Assistant wants to call {len(response.tool_calls)} tools:")
        for i, tool_call in enumerate(response.tool_calls, 1):
            print(f"  {i}. {tool_call['name']} with args: {tool_call['args']}")
        print()

        # Execute all tools
        messages.append(response)
        tool_map = {t.name: t for t in tools}

        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]

            if tool_name in tool_map:
                result = tool_map[tool_name].invoke(tool_args)

                if args.verbose:
                    print(f"[DEBUG] Executed {tool_name}: {result}")

                messages.append(
                    ToolMessage(content=result, tool_call_id=tool_call["id"])
                )

        if args.verbose:
            print()

        # Get final response
        final_response = llm_with_tools.invoke(messages)
        print(f"Final answer: {final_response.content}\n")

    print("=== 3. Mathematical Calculation ===\n")

    messages = [HumanMessage(content="What is 42 multiplied by 7?")]

    print(f"User: {messages[0].content}")

    response = llm_with_tools.invoke(messages)

    if response.tool_calls:
        tool_call = response.tool_calls[0]
        print(f"Assistant calling: {tool_call['name']}")
        print(f"  Arguments: {tool_call['args']}\n")

        # Execute
        tool_map = {t.name: t for t in tools}
        result = tool_map[tool_call["name"]].invoke(tool_call["args"])

        messages.append(response)
        messages.append(ToolMessage(content=result, tool_call_id=tool_call["id"]))

        final_response = llm_with_tools.invoke(messages)
        print(f"Final answer: {final_response.content}\n")

    print("=== 4. Multi-Turn with Tools ===\n")

    conversation = [HumanMessage(content="What's the weather like in New York?")]

    max_turns = 10
    turn = 0

    while turn < max_turns:
        turn += 1

        if args.verbose:
            print(f"[DEBUG] Turn {turn}")

        response = llm_with_tools.invoke(conversation)

        # Check if done
        if not response.tool_calls:
            print(f"Assistant: {response.content}\n")
            break

        # Execute tools
        conversation.append(response)
        tool_map = {t.name: t for t in tools}

        for tool_call in response.tool_calls:
            if args.verbose:
                print(f"[DEBUG] Calling {tool_call['name']} with {tool_call['args']}")

            result = tool_map[tool_call["name"]].invoke(tool_call["args"])
            conversation.append(
                ToolMessage(content=result, tool_call_id=tool_call["id"])
            )

    if turn >= max_turns:
        print("[WARNING] Reached maximum turns\n")

    if args.verbose:
        print(f"[DEBUG] Conversation had {turn} turns")
        print("[DEBUG] Tool use examples completed")


if __name__ == "__main__":
    main()
