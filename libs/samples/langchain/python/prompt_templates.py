"""Prompt template examples with ChatNova."""

import argparse

from langchain_core.prompts import (
    ChatPromptTemplate,
    FewShotChatMessagePromptTemplate,
    MessagesPlaceholder,
)

from langchain_nova import ChatNova


def main():
    parser = argparse.ArgumentParser(description="Prompt templates with Nova")
    parser.add_argument("--model", type=str, default="nova-pro-v1")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    llm = ChatNova(model=args.model, temperature=0.7)

    if args.verbose:
        print(f"\n[DEBUG] Using model: {args.model}")
        print("[DEBUG] Demonstrating prompt templates\n")

    print("=== 1. Simple Template ===\n")

    template = ChatPromptTemplate.from_template(
        "You are an expert in {subject}. Explain {concept} in simple terms."
    )

    if args.verbose:
        print(f"[DEBUG] Template: {template.input_variables}")

    chain = template | llm
    result = chain.invoke({"subject": "physics", "concept": "quantum entanglement"})
    print(f"Result: {result.content}\n")

    print("=== 2. Chat Template with System Message ===\n")

    chat_template = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a {role}. Keep responses {style}."),
            ("human", "{input}"),
        ]
    )

    if args.verbose:
        print(f"[DEBUG] Chat template with {len(chat_template.messages)} messages")

    chain = chat_template | llm
    result = chain.invoke(
        {
            "role": "pirate",
            "style": "brief and in character",
            "input": "What's the weather like?",
        }
    )
    print(f"Result: {result.content}\n")

    print("=== 3. Few-Shot Examples ===\n")

    examples = [
        {"input": "happy", "output": "sad"},
        {"input": "tall", "output": "short"},
        {"input": "hot", "output": "cold"},
    ]

    example_prompt = ChatPromptTemplate.from_messages(
        [
            ("human", "{input}"),
            ("ai", "{output}"),
        ]
    )

    few_shot_prompt = FewShotChatMessagePromptTemplate(
        example_prompt=example_prompt,
        examples=examples,
    )

    final_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are providing antonyms."),
            few_shot_prompt,
            ("human", "{input}"),
        ]
    )

    if args.verbose:
        print(f"[DEBUG] Few-shot prompt with {len(examples)} examples")

    chain = final_prompt | llm
    result = chain.invoke({"input": "light"})
    print(f"Input: 'light'\nOutput: {result.content}\n")

    print("=== 4. Template with Message Placeholder ===\n")

    prompt_with_history = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
        ]
    )

    from langchain_core.messages import AIMessage, HumanMessage

    history = [
        HumanMessage(content="My name is Alice"),
        AIMessage(content="Nice to meet you, Alice!"),
    ]

    if args.verbose:
        print(f"[DEBUG] Using message placeholder with {len(history)} history messages")

    chain = prompt_with_history | llm
    result = chain.invoke({"chat_history": history, "input": "What's my name?"})
    print(f"Result: {result.content}\n")

    if args.verbose:
        print("[DEBUG] All prompt template examples completed")


if __name__ == "__main__":
    main()
