#!/usr/bin/env python3
"""
Interactive chat script for experimenting with Nova API.
Usage: python nova_chat.py
Set NOVA_API_KEY environment variable before running.
"""

import os
import sys
import json
import requests
from typing import List, Dict, Any, Optional

NOVA_API_URL = "https://api.nova.amazon.com/v1/chat/completions"


class NovaChat:
    def __init__(self, api_key: str, model: str = "nova-pro-v1"):
        self.api_key = api_key
        self.model = model
        self.messages: List[Dict[str, str]] = []
        self.tools: Optional[List[Dict[str, Any]]] = None

    def add_message(self, role: str, content: str):
        self.messages.append({"role": role, "content": content})

    def set_tools(self, tools: List[Dict[str, Any]]):
        self.tools = tools

    def chat(
        self,
        temperature: float = 0.7,
        max_tokens: int = 10000,
        top_p: float = 0.7,
        stream: bool = True,
        tool_choice: str = "auto",
    ) -> str:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        payload = {
            "model": self.model,
            "messages": self.messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "top_p": top_p,
            "stream": stream,
        }

        if stream:
            payload["stream_options"] = {"include_usage": True}

        if self.tools:
            payload["tools"] = self.tools
            payload["tool_choice"] = tool_choice

        response = requests.post(
            NOVA_API_URL, headers=headers, json=payload, stream=stream
        )
        response.raise_for_status()

        if stream:
            return self._handle_stream(response)
        else:
            result = response.json()
            return result["choices"][0]["message"]["content"]

    def _handle_stream(self, response) -> str:
        full_content = ""
        print("Assistant: ", end="", flush=True)

        for line in response.iter_lines():
            if not line:
                continue

            line = line.decode("utf-8")
            if line.startswith("data: "):
                data = line[6:]
                if data == "[DONE]":
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
                            print(f"\n[Tool call: {delta['tool_calls']}]", flush=True)
                    elif "usage" in chunk:
                        print(f"\n[Usage: {chunk['usage']}]", flush=True)
                except json.JSONDecodeError:
                    continue

        print()
        return full_content


def main():
    api_key = os.getenv("NOVA_API_KEY")
    if not api_key:
        print("Error: NOVA_API_KEY environment variable not set")
        sys.exit(1)

    print("Nova Chat - Interactive Mode")
    print("Commands: /reset, /tools, /params, /history, /quit")
    print("-" * 50)

    chat = NovaChat(api_key)
    temperature = 0.7
    max_tokens = 10000
    top_p = 0.7
    stream = True

    while True:
        try:
            user_input = input("\nYou: ").strip()

            if not user_input:
                continue

            if user_input == "/quit":
                break
            elif user_input == "/reset":
                chat.messages.clear()
                print("Conversation reset.")
                continue
            elif user_input == "/history":
                print(json.dumps(chat.messages, indent=2))
                continue
            elif user_input == "/params":
                print(f"Model: {chat.model}")
                print(f"Temperature: {temperature}")
                print(f"Max tokens: {max_tokens}")
                print(f"Top-p: {top_p}")
                print(f"Stream: {stream}")
                continue
            elif user_input == "/tools":
                print(
                    "Current tools:",
                    json.dumps(chat.tools, indent=2) if chat.tools else "None",
                )
                continue

            chat.add_message("user", user_input)
            response = chat.chat(
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                stream=stream,
            )
            chat.add_message("assistant", response)

        except KeyboardInterrupt:
            print("\nExiting...")
            break
        except Exception as e:
            print(f"\nError: {e}")


if __name__ == "__main__":
    main()
