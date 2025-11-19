#!/usr/bin/env python3
"""
Configurable Nova API testing script.
Usage: python nova_test.py --help
"""

import os
import sys
import json
import argparse
import requests
from typing import List, Dict, Any, Optional


NOVA_API_URL = "https://api.nova.amazon.com/v1/chat/completions"


def call_nova(
    api_key: str,
    messages: List[Dict[str, str]],
    model: str = "nova-pro-v1",
    temperature: float = 0.7,
    max_tokens: int = 10000,
    top_p: float = 0.7,
    stream: bool = True,
    tools: Optional[List[Dict[str, Any]]] = None,
    tool_choice: str = "auto",
):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
        "stream": stream,
    }

    if stream:
        payload["stream_options"] = {"include_usage": True}

    if tools:
        payload["tools"] = tools
        payload["tool_choice"] = tool_choice

    print("Request payload:")
    print(json.dumps(payload, indent=2))
    print("\n" + "=" * 50 + "\n")

    response = requests.post(NOVA_API_URL, headers=headers, json=payload, stream=stream)
    response.raise_for_status()

    if stream:
        handle_stream_response(response)
    else:
        result = response.json()
        print("Response:")
        print(json.dumps(result, indent=2))


def handle_stream_response(response):
    print("Streaming response:\n")
    full_content = ""

    for line in response.iter_lines():
        if not line:
            continue

        line = line.decode("utf-8")
        if line.startswith("data: "):
            data = line[6:]
            if data == "[DONE]":
                print("\n[DONE]")
                break

            try:
                chunk = json.loads(data)
                if "choices" in chunk and len(chunk["choices"]) > 0:
                    delta = chunk["choices"][0].get("delta", {})
                    if "content" in delta:
                        content = delta["content"]
                        print(content, end="", flush=True)
                        full_content += content
                    elif "tool_calls" in delta:
                        print(f"\n[Tool call: {json.dumps(delta['tool_calls'], indent=2)}]")
                elif "usage" in chunk:
                    print(f"\n\n[Usage: {json.dumps(chunk['usage'], indent=2)}]")
            except json.JSONDecodeError as e:
                print(f"\nJSON decode error: {e}")
                continue

    print("\n\nFull content:")
    print(full_content)


def load_example_tools() -> List[Dict[str, Any]]:
    return [
        {
            "type": "function",
            "function": {
                "name": "getCurrentWeather",
                "description": "Get the current weather in a given city",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {
                            "type": "string",
                            "description": "City and country e.g. Bogotá, Colombia",
                        }
                    },
                    "required": ["location"],
                },
            },
        }
    ]


def main():
    parser = argparse.ArgumentParser(description="Test Nova API with custom parameters")
    parser.add_argument("prompt", nargs="?", help="User prompt to send")
    parser.add_argument("--model", default="nova-pro-v1", help="Model to use")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature")
    parser.add_argument("--max-tokens", type=int, default=10000, help="Max tokens")
    parser.add_argument("--top-p", type=float, default=0.7, help="Top-p")
    parser.add_argument("--no-stream", action="store_true", help="Disable streaming")
    parser.add_argument("--tools", action="store_true", help="Include example tools")
    parser.add_argument("--tool-choice", default="auto", help="Tool choice strategy")
    parser.add_argument(
        "--messages-file",
        help="JSON file containing messages array",
    )
    parser.add_argument(
        "--system",
        help="System message to prepend",
    )

    args = parser.parse_args()

    api_key = os.getenv("NOVA_API_KEY")
    if not api_key:
        print("Error: NOVA_API_KEY environment variable not set")
        sys.exit(1)

    if args.messages_file:
        with open(args.messages_file, "r") as f:
            messages = json.load(f)
    elif args.prompt:
        messages = []
        if args.system:
            messages.append({"role": "system", "content": args.system})
        messages.append({"role": "user", "content": args.prompt})
    else:
        print("Error: Either provide a prompt or --messages-file")
        sys.exit(1)

    tools = load_example_tools() if args.tools else None

    call_nova(
        api_key=api_key,
        messages=messages,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=args.top_p,
        stream=not args.no_stream,
        tools=tools,
        tool_choice=args.tool_choice,
    )


if __name__ == "__main__":
    main()
