"""Interactive chat interface with tool use for Amazon Nova using LangChain.

This script provides an enhanced command-line chat interface for interacting
with Amazon Nova models through the LangChain integration, with comprehensive
tool use capabilities including weather, time, calculations, web search, and more.

Requirements:
    - NOVA_API_KEY environment variable set
    - NOVA_BASE_URL environment variable set (optional)

Usage:
    python chat_with_tools.py
    python chat_with_tools.py --model nova-lite-v1
    python chat_with_tools.py --streaming
    python chat_with_tools.py --verbose
    python chat_with_tools.py --disable-tools
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Literal

from langchain_core.callbacks import StreamingStdOutCallbackHandler
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langchain_nova import ChatNova


# Define comprehensive tool set
@tool
def get_current_weather(
    location: str, unit: Literal["celsius", "fahrenheit"] = "fahrenheit"
) -> str:
    """Get the current weather for a location.

    Args:
        location: The city and state, e.g. San Francisco, CA
        unit: The temperature unit to use. Defaults to fahrenheit.
    """
    # Simulated weather data - in production, use a real weather API
    weather_data = {
        "San Francisco, CA": {
            "temp": 65,
            "condition": "cloudy",
            "humidity": 75,
            "wind": "10 mph",
        },
        "Seattle, WA": {
            "temp": 52,
            "condition": "rainy",
            "humidity": 85,
            "wind": "8 mph",
        },
        "New York, NY": {
            "temp": 45,
            "condition": "partly cloudy",
            "humidity": 60,
            "wind": "12 mph",
        },
        "Miami, FL": {
            "temp": 78,
            "condition": "sunny",
            "humidity": 70,
            "wind": "5 mph",
        },
        "Los Angeles, CA": {
            "temp": 72,
            "condition": "sunny",
            "humidity": 65,
            "wind": "7 mph",
        },
        "Chicago, IL": {
            "temp": 38,
            "condition": "snowy",
            "humidity": 80,
            "wind": "15 mph",
        },
        "London, UK": {
            "temp": 50,
            "condition": "foggy",
            "humidity": 90,
            "wind": "6 mph",
        },
        "Tokyo, Japan": {
            "temp": 68,
            "condition": "clear",
            "humidity": 55,
            "wind": "4 mph",
        },
    }

    data = weather_data.get(
        location, {"temp": 70, "condition": "unknown", "humidity": 50, "wind": "5 mph"}
    )

    if unit == "celsius":
        temp = round((data["temp"] - 32) * 5 / 9, 1)
        unit_str = "°C"
    else:
        temp = data["temp"]
        unit_str = "°F"

    return f"Weather in {location}: {data['condition']}, {temp}{unit_str}, {data['humidity']}% humidity, wind {data['wind']}"


@tool
def get_current_time(timezone: str = "UTC") -> str:
    """Get the current time in a specific timezone.

    Args:
        timezone: The timezone name (e.g., UTC, America/New_York, Europe/London, Asia/Tokyo)
    """
    # Simplified timezone handling - in production use pytz or zoneinfo
    current_time = datetime.now()

    # Simulate different timezones with offsets
    timezone_offsets = {
        "UTC": 0,
        "America/New_York": -5,
        "America/Los_Angeles": -8,
        "Europe/London": 0,
        "Europe/Paris": 1,
        "Asia/Tokyo": 9,
        "Asia/Shanghai": 8,
        "Australia/Sydney": 11,
    }

    offset = timezone_offsets.get(timezone, 0)
    adjusted_time = current_time.replace(hour=(current_time.hour + offset) % 24)

    return (
        f"Current time in {timezone}: {adjusted_time.strftime('%I:%M %p on %B %d, %Y')}"
    )


@tool
def calculate(operation: str, a: float, b: float) -> str:
    """Perform a mathematical calculation.

    Args:
        operation: The operation to perform (add, subtract, multiply, divide, power, modulo)
        a: The first number
        b: The second number
    """
    try:
        operations = {
            "add": a + b,
            "subtract": a - b,
            "multiply": a * b,
            "divide": a / b if b != 0 else "Error: Division by zero",
            "power": a**b,
            "modulo": a % b if b != 0 else "Error: Modulo by zero",
        }

        if operation not in operations:
            return f"Error: Unknown operation '{operation}'. Available: {', '.join(operations.keys())}"

        result = operations[operation]
        if isinstance(result, str):  # Error case
            return result

        return f"{a} {operation} {b} = {result}"
    except Exception as e:
        return f"Error performing calculation: {str(e)}"


@tool
def search_web(query: str, num_results: int = 3) -> str:
    """Search the web for information (simulated).

    Args:
        query: The search query
        num_results: Number of results to return (1-5)
    """
    # Simulated search results - in production, use a real search API
    search_responses = {
        "python": [
            "Python.org - Official Python website with documentation and downloads",
            "Real Python - Comprehensive Python tutorials and articles",
            "Python Package Index (PyPI) - Repository of Python packages",
        ],
        "weather": [
            "Weather.com - Current weather conditions and forecasts",
            "AccuWeather - Detailed weather information and radar",
            "National Weather Service - Official US weather data",
        ],
        "news": [
            "BBC News - Latest international news and updates",
            "Reuters - Breaking news and business information",
            "Associated Press - Global news coverage",
        ],
        "default": [
            f"Search result 1 for '{query}' - Relevant information found",
            f"Search result 2 for '{query}' - Additional details available",
            f"Search result 3 for '{query}' - More comprehensive coverage",
        ],
    }

    # Find matching responses
    results = None
    for key in search_responses:
        if key.lower() in query.lower():
            results = search_responses[key]
            break

    if not results:
        results = [
            r.replace("'{query}'", f"'{query}'") for r in search_responses["default"]
        ]

    # Limit results
    num_results = max(1, min(5, num_results))
    results = results[:num_results]

    formatted_results = "\n".join(
        [f"{i + 1}. {result}" for i, result in enumerate(results)]
    )
    return f"Search results for '{query}':\n{formatted_results}"


@tool
def get_file_info(filename: str) -> str:
    """Get information about a file in the current directory.

    Args:
        filename: The name of the file to check
    """
    try:
        if os.path.exists(filename):
            stat = os.stat(filename)
            size = stat.st_size
            modified = datetime.fromtimestamp(stat.st_mtime)

            # Determine file type
            if os.path.isdir(filename):
                file_type = "directory"
            elif filename.endswith((".py", ".js", ".java", ".cpp", ".c")):
                file_type = "source code file"
            elif filename.endswith((".txt", ".md", ".rst")):
                file_type = "text file"
            elif filename.endswith((".json", ".xml", ".yaml", ".yml")):
                file_type = "configuration file"
            else:
                file_type = "file"

            return f"{filename}: {file_type}, {size} bytes, last modified {modified.strftime('%Y-%m-%d %H:%M:%S')}"
        else:
            return f"File '{filename}' does not exist in the current directory"
    except Exception as e:
        return f"Error accessing file '{filename}': {str(e)}"


@tool
def text_analysis(text: str, analysis_type: str = "stats") -> str:
    """Analyze text and provide statistics or insights.

    Args:
        text: The text to analyze
        analysis_type: Type of analysis (stats, words, chars, sentiment)
    """
    try:
        if analysis_type == "stats":
            char_count = len(text)
            word_count = len(text.split())
            line_count = text.count("\n") + 1
            return f"Text statistics: {char_count} characters, {word_count} words, {line_count} lines"

        elif analysis_type == "words":
            words = text.lower().split()
            word_freq = {}
            for word in words:
                word_freq[word] = word_freq.get(word, 0) + 1

            top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
            result = "Most frequent words:\n"
            for word, count in top_words:
                result += f"  '{word}': {count} times\n"
            return result.strip()

        elif analysis_type == "chars":
            char_freq = {}
            for char in text.lower():
                if char.isalpha():
                    char_freq[char] = char_freq.get(char, 0) + 1

            top_chars = sorted(char_freq.items(), key=lambda x: x[1], reverse=True)[:5]
            result = "Most frequent characters:\n"
            for char, count in top_chars:
                result += f"  '{char}': {count} times\n"
            return result.strip()

        elif analysis_type == "sentiment":
            # Simple sentiment analysis based on keywords
            positive_words = [
                "good",
                "great",
                "excellent",
                "amazing",
                "wonderful",
                "fantastic",
                "love",
                "like",
                "happy",
                "joy",
            ]
            negative_words = [
                "bad",
                "terrible",
                "awful",
                "horrible",
                "hate",
                "dislike",
                "sad",
                "angry",
                "frustrated",
                "disappointed",
            ]

            text_lower = text.lower()
            positive_count = sum(1 for word in positive_words if word in text_lower)
            negative_count = sum(1 for word in negative_words if word in text_lower)

            if positive_count > negative_count:
                sentiment = "positive"
            elif negative_count > positive_count:
                sentiment = "negative"
            else:
                sentiment = "neutral"

            return f"Sentiment analysis: {sentiment} (positive: {positive_count}, negative: {negative_count})"

        else:
            return f"Unknown analysis type '{analysis_type}'. Available: stats, words, chars, sentiment"

    except Exception as e:
        return f"Error analyzing text: {str(e)}"


@tool
def system_info(info_type: str = "basic") -> str:
    """Get system information.

    Args:
        info_type: Type of info to get (basic, time, env, python)
    """
    try:
        if info_type == "basic":
            return f"Operating System: {os.name}, Current Directory: {os.getcwd()}"

        elif info_type == "time":
            now = datetime.now()
            return f"System time: {now.strftime('%Y-%m-%d %H:%M:%S')}, Timezone: {time.tzname[0]}"

        elif info_type == "env":
            important_vars = [
                "PATH",
                "HOME",
                "USER",
                "SHELL",
                "NOVA_API_KEY",
                "NOVA_BASE_URL",
            ]
            env_info = []
            for var in important_vars:
                value = os.environ.get(var, "Not set")
                if var in ["NOVA_API_KEY", "NOVA_BASE_URL"] and value != "Not set":
                    value = f"Set ({len(value)} characters)"
                env_info.append(f"{var}: {value}")
            return "Environment variables:\n" + "\n".join(env_info)

        elif info_type == "python":
            return f"Python version: {sys.version}, Executable: {sys.executable}"

        else:
            return (
                f"Unknown info type '{info_type}'. Available: basic, time, env, python"
            )

    except Exception as e:
        return f"Error getting system info: {str(e)}"


def print_banner(verbose: bool = False, tools_enabled: bool = True) -> None:
    """Print welcome banner."""
    print("\n" + "=" * 70)
    print("  Amazon Nova Interactive Chat with Tools")
    print("=" * 70)
    print("  Commands:")
    print("    /help     - Show this help message")
    print("    /clear    - Clear conversation history")
    print("    /history  - Show conversation history")
    print("    /stats    - Show conversation statistics")
    if tools_enabled:
        print("    /tools    - List available tools")
        print("    /toolhelp - Show detailed tool information")
    print("    /exit     - Exit the chat")
    if verbose:
        print("\n  Verbose mode: ON")
    if tools_enabled:
        print(f"  Tools: ENABLED ({len(get_available_tools())} available)")
    else:
        print("  Tools: DISABLED")
    print("=" * 70 + "\n")


def print_help(tools_enabled: bool = True) -> None:
    """Print help message."""
    print("\nAvailable commands:")
    print("  /help     - Show this help message")
    print("  /clear    - Clear conversation history")
    print("  /history  - Show conversation history")
    print("  /stats    - Show conversation statistics")
    if tools_enabled:
        print("  /tools    - List available tools")
        print("  /toolhelp - Show detailed tool information")
    print("  /exit     - Exit the chat")
    if tools_enabled:
        print("\nWith tools enabled, the assistant can:")
        print("  • Get weather information for cities")
        print("  • Check current time in different timezones")
        print("  • Perform mathematical calculations")
        print("  • Search the web (simulated)")
        print("  • Analyze files and text")
        print("  • Get system information")
    print()


def get_available_tools():
    """Get list of available tools."""
    return [
        get_current_weather,
        get_current_time,
        calculate,
        search_web,
        get_file_info,
        text_analysis,
        system_info,
    ]


def print_tools() -> None:
    """Print available tools."""
    tools = get_available_tools()
    print("\nAvailable Tools:")
    print("-" * 50)
    for i, tool in enumerate(tools, 1):
        print(f"{i}. {tool.name}")
        print(f"   Description: {tool.description.split('.')[0]}.")
    print("-" * 50 + "\n")


def print_tool_help() -> None:
    """Print detailed tool help."""
    tools = get_available_tools()
    print("\nDetailed Tool Information:")
    print("=" * 60)
    for tool in tools:
        print(f"\n{tool.name.upper()}")
        print("-" * len(tool.name))
        print(f"Description: {tool.description}")

        # Extract args from tool schema
        if hasattr(tool, "args_schema") and tool.args_schema:
            print("Arguments:")
            schema = tool.args_schema.schema()
            if "properties" in schema:
                for arg_name, arg_info in schema["properties"].items():
                    arg_type = arg_info.get("type", "unknown")
                    arg_desc = arg_info.get("description", "No description")
                    required = arg_name in schema.get("required", [])
                    req_str = " (required)" if required else " (optional)"
                    print(f"  • {arg_name}: {arg_type}{req_str}")
                    print(f"    {arg_desc}")
    print("=" * 60 + "\n")


def print_history(messages: List[BaseMessage]) -> None:
    """Print conversation history."""
    print("\nConversation History:")
    print("-" * 60)
    for i, msg in enumerate(messages, 1):
        role = msg.type.upper()
        if msg.type == "tool":
            # Tool messages are usually long, show abbreviated
            content = (
                f"Tool result: {msg.content[:50]}..."
                if len(msg.content) > 50
                else f"Tool result: {msg.content}"
            )
        else:
            content = (
                msg.content[:100] + "..." if len(msg.content) > 100 else msg.content
            )
        print(f"{i}. [{role}] {content}")
    print("-" * 60 + "\n")


def print_stats(
    messages: List[BaseMessage], total_time: float, tool_calls_count: int = 0
) -> None:
    """Print conversation statistics."""
    user_msgs = sum(1 for m in messages if m.type == "human")
    ai_msgs = sum(1 for m in messages if m.type == "ai")
    tool_msgs = sum(1 for m in messages if m.type == "tool")
    total_chars = sum(len(m.content) for m in messages if m.content)

    print("\nConversation Statistics:")
    print("-" * 60)
    print(f"  Total messages: {len(messages)}")
    print(f"  User messages: {user_msgs}")
    print(f"  Assistant messages: {ai_msgs}")
    print(f"  Tool messages: {tool_msgs}")
    print(f"  Tool calls made: {tool_calls_count}")
    print(f"  Total characters: {total_chars}")
    print(f"  Session time: {total_time:.1f}s")
    if ai_msgs > 0:
        print(f"  Avg response time: {total_time / ai_msgs:.1f}s")
    print("-" * 60 + "\n")


def execute_tools(
    tool_calls: List[Dict[str, Any]], tools: List, verbose: bool = False
) -> List[ToolMessage]:
    """Execute tool calls and return tool messages."""
    tool_map = {t.name: t for t in tools}
    tool_messages = []

    for tool_call in tool_calls:
        tool_name = tool_call["name"]
        tool_args = tool_call["args"]
        tool_id = tool_call["id"]

        if verbose:
            print(f"[DEBUG] Executing tool: {tool_name}")
            print(f"[DEBUG] Arguments: {json.dumps(tool_args, indent=2)}")

        try:
            if tool_name in tool_map:
                result = tool_map[tool_name].invoke(tool_args)
                if verbose:
                    print(
                        f"[DEBUG] Tool result: {result[:100]}..."
                        if len(str(result)) > 100
                        else f"[DEBUG] Tool result: {result}"
                    )
            else:
                result = f"Error: Unknown tool '{tool_name}'"
                if verbose:
                    print(f"[DEBUG] {result}")

            tool_messages.append(ToolMessage(content=str(result), tool_call_id=tool_id))

        except Exception as e:
            error_msg = f"Error executing tool '{tool_name}': {str(e)}"
            if verbose:
                print(f"[DEBUG] {error_msg}")
            tool_messages.append(ToolMessage(content=error_msg, tool_call_id=tool_id))

    return tool_messages


def chat_streaming_with_tools(
    llm: ChatNova,
    messages: List[BaseMessage],
    user_input: str,
    tools: List,
    verbose: bool = False,
) -> tuple[str, float, int]:
    """Handle streaming chat interaction with tools."""
    messages.append(HumanMessage(content=user_input))
    tool_calls_made = 0
    max_iterations = 5  # Prevent infinite tool calling loops

    for iteration in range(max_iterations):
        if verbose:
            print(
                f"\n[DEBUG] Iteration {iteration + 1}, sending {len(messages)} messages to model"
            )

        start_time = time.time()
        print("\nAssistant: ", end="", flush=True)

        # Stream the response
        full_response = ""
        chunk_count = 0
        response = None

        for chunk in llm.stream(messages):
            content = chunk.content
            if content:
                print(content, end="", flush=True)
                full_response += content
                chunk_count += 1
            response = chunk  # Keep the last chunk which has tool_calls

        elapsed = time.time() - start_time

        if verbose:
            print(f"\n[DEBUG] Received {chunk_count} chunks in {elapsed:.2f}s")
            print(f"[DEBUG] Response length: {len(full_response)} chars")

        # Add the AI response to messages
        ai_message = AIMessage(content=full_response)
        if hasattr(response, "tool_calls") and response.tool_calls:
            ai_message.tool_calls = response.tool_calls
        messages.append(ai_message)

        # Check for tool calls
        if hasattr(response, "tool_calls") and response.tool_calls:
            tool_calls_made += len(response.tool_calls)

            print(f"\n[TOOLS] Calling {len(response.tool_calls)} tool(s)...")
            for i, tool_call in enumerate(response.tool_calls, 1):
                print(
                    f"  {i}. {tool_call['name']} with args: {json.dumps(tool_call['args'])}"
                )

            # Execute tools
            tool_messages = execute_tools(response.tool_calls, tools, verbose)
            messages.extend(tool_messages)

            print("[TOOLS] Tool execution completed, getting final response...\n")
            continue  # Go to next iteration to get final response
        else:
            # No more tool calls, we're done
            break

    print()
    return full_response, elapsed, tool_calls_made


def chat_non_streaming_with_tools(
    llm: ChatNova,
    messages: List[BaseMessage],
    user_input: str,
    tools: List,
    verbose: bool = False,
) -> tuple[str, float, int]:
    """Handle non-streaming chat interaction with tools."""
    messages.append(HumanMessage(content=user_input))
    tool_calls_made = 0
    max_iterations = 5  # Prevent infinite tool calling loops

    if verbose:
        print(f"\n[DEBUG] Sending {len(messages)} messages to model")
        print(f"[DEBUG] User message length: {len(user_input)} chars")

    for iteration in range(max_iterations):
        start_time = time.time()
        response = llm.invoke(messages)
        elapsed = time.time() - start_time

        messages.append(response)

        if verbose:
            print(f"[DEBUG] Iteration {iteration + 1} response time: {elapsed:.2f}s")
            print(f"[DEBUG] Response length: {len(response.content)} chars")

        # Check for tool calls
        if hasattr(response, "tool_calls") and response.tool_calls:
            tool_calls_made += len(response.tool_calls)

            print(f"\n[TOOLS] Calling {len(response.tool_calls)} tool(s)...")
            for i, tool_call in enumerate(response.tool_calls, 1):
                print(
                    f"  {i}. {tool_call['name']} with args: {json.dumps(tool_call['args'])}"
                )

            # Execute tools
            tool_messages = execute_tools(response.tool_calls, tools, verbose)
            messages.extend(tool_messages)

            if verbose:
                print(f"[DEBUG] Added {len(tool_messages)} tool messages")

            continue  # Go to next iteration to get final response
        else:
            # No tool calls, show final response
            print(f"\nAssistant: {response.content}\n")

            if verbose:
                if hasattr(response, "response_metadata"):
                    print(f"[DEBUG] Metadata: {response.response_metadata}")
                if hasattr(response, "usage_metadata"):
                    usage = response.usage_metadata
                    print(f"[DEBUG] Usage: {usage}")
                    if "total_tokens" in usage and elapsed > 0:
                        print(
                            f"[DEBUG] Tokens/sec: {usage['total_tokens'] / elapsed:.1f}"
                        )

            break

    return response.content, elapsed, tool_calls_made


def main() -> None:
    """Run interactive chat with tools."""
    parser = argparse.ArgumentParser(
        description="Interactive chat with Amazon Nova and comprehensive tools"
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
        default="You are a helpful AI assistant with access to various tools. Use tools when appropriate to provide accurate, up-to-date information and perform specific tasks.",
        help="System message to set assistant behavior",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose output with detailed information",
    )
    parser.add_argument(
        "--disable-tools",
        action="store_true",
        help="Disable tool use (run as regular chat)",
    )

    args = parser.parse_args()

    # Initialize the model
    try:
        llm = ChatNova(
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )

        # Bind tools if enabled
        tools_enabled = not args.disable_tools
        tools = []
        if tools_enabled:
            tools = get_available_tools()
            llm = llm.bind_tools(tools)

        if args.verbose:
            print(f"\n[DEBUG] Model initialized:")
            print(f"[DEBUG]   Model: {args.model}")
            print(f"[DEBUG]   Temperature: {args.temperature}")
            print(f"[DEBUG]   Max tokens: {args.max_tokens}")
            print(f"[DEBUG]   Base URL: {llm.base_url}")
            print(f"[DEBUG]   Tools enabled: {tools_enabled}")
            if tools_enabled:
                print(f"[DEBUG]   Available tools: {[t.name for t in tools]}")
        else:
            mode = "with tools" if tools_enabled else "tools disabled"
            print(f"\nInitialized {args.model} ({mode}, streaming: {args.streaming})")
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

    print_banner(args.verbose, tools_enabled)

    # Track session statistics
    session_start = time.time()
    total_response_time = 0.0
    total_tool_calls = 0

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
                        print(
                            f"[DEBUG] Total response time: {total_response_time:.1f}s"
                        )
                        print(f"[DEBUG] Total tool calls: {total_tool_calls}")
                    print("\nGoodbye!\n")
                    break

                elif command == "/help":
                    print_help(tools_enabled)
                    continue

                elif command == "/clear":
                    messages = [SystemMessage(content=args.system_message)]
                    total_response_time = 0.0
                    total_tool_calls = 0
                    print("\nConversation history cleared.\n")
                    if args.verbose:
                        print("[DEBUG] Message history and stats reset")
                    continue

                elif command == "/history":
                    print_history(messages)
                    continue

                elif command == "/stats":
                    session_time = time.time() - session_start
                    print_stats(messages, session_time, total_tool_calls)
                    continue

                elif command == "/tools" and tools_enabled:
                    print_tools()
                    continue

                elif command == "/toolhelp" and tools_enabled:
                    print_tool_help()
                    continue

                elif command in ["/tools", "/toolhelp"] and not tools_enabled:
                    print(
                        "\nTools are disabled. Use --help to see how to enable them.\n"
                    )
                    continue

                else:
                    print(f"\nUnknown command: {user_input}")
                    print("Type /help for available commands.\n")
                    continue

            # Process chat message
            if tools_enabled:
                if args.streaming:
                    response_content, elapsed, tool_calls = chat_streaming_with_tools(
                        llm, messages, user_input, tools, args.verbose
                    )
                else:
                    response_content, elapsed, tool_calls = (
                        chat_non_streaming_with_tools(
                            llm, messages, user_input, tools, args.verbose
                        )
                    )
                total_tool_calls += tool_calls
            else:
                # Regular chat without tools
                if args.streaming:
                    response_content, elapsed, _ = chat_streaming_with_tools(
                        llm, messages, user_input, [], args.verbose
                    )
                else:
                    response_content, elapsed, _ = chat_non_streaming_with_tools(
                        llm, messages, user_input, [], args.verbose
                    )

            total_response_time += elapsed

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
