#!/usr/bin/env python3
"""
Nova API web search testing script.
Usage: python nova_web_search.py --help
"""

import os
import sys
import json
import argparse
import requests
from typing import Dict, Any, Optional


NOVA_API_URL = "https://api.nova.amazon.com/v1/chat/completions"


def call_nova_with_web_search(
    api_key: str,
    user_prompt: str,
    model: str = "nova-premier-v1",
    system_prompt: Optional[str] = None,
    search_context_size: str = "low",
    temperature: float = 0.7,
    max_tokens: int = 10000,
    top_p: float = 0.7,
    stream: bool = False,
):
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }

    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})

    payload = {
        "model": model,
        "messages": messages,
        "web_search_options": {"search_context_size": search_context_size},
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
        "stream": stream,
    }

    if stream:
        payload["stream_options"] = {"include_usage": True}

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

        if "choices" in result and len(result["choices"]) > 0:
            content = result["choices"][0]["message"]["content"]
            print("\n" + "=" * 50)
            print("Assistant response:")
            print("=" * 50)
            print(content)


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
                elif "usage" in chunk:
                    print(f"\n\n[Usage: {json.dumps(chunk['usage'], indent=2)}]")
            except json.JSONDecodeError as e:
                print(f"\nJSON decode error: {e}")
                continue

    print("\n\nFull content:")
    print(full_content)


def main():
    parser = argparse.ArgumentParser(
        description="Test Nova API with web search capabilities"
    )
    parser.add_argument("prompt", help="User prompt to send")
    parser.add_argument(
        "--model",
        default="nova-premier-v1",
        help="Model to use (default: nova-premier-v1)",
    )
    parser.add_argument(
        "--system",
        help="System prompt",
    )
    parser.add_argument(
        "--search-context-size",
        choices=["low", "medium", "high"],
        default="low",
        help="Web search context size (default: low)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature (default: 0.7)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=10000,
        help="Max tokens (default: 10000)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.7,
        help="Top-p (default: 0.7)",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Enable streaming",
    )

    args = parser.parse_args()

    api_key = os.getenv("NOVA_API_KEY")
    if not api_key:
        print("Error: NOVA_API_KEY environment variable not set")
        sys.exit(1)

    call_nova_with_web_search(
        api_key=api_key,
        user_prompt=args.prompt,
        model=args.model,
        system_prompt=args.system,
        search_context_size=args.search_context_size,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        top_p=args.top_p,
        stream=args.stream,
    )


if __name__ == "__main__":
    main()
