"""Memory and conversation context examples with ChatNova."""

import argparse

from langchain.memory import ConversationBufferMemory, ConversationSummaryMemory
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain_nova import ChatNova


def main():
    parser = argparse.ArgumentParser(description="Memory examples with Nova")
    parser.add_argument("--model", type=str, default="nova-pro-v1")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    llm = ChatNova(model=args.model, temperature=0.7)

    if args.verbose:
        print(f"\n[DEBUG] Using model: {args.model}")
        print("[DEBUG] Demonstrating conversation memory\n")

    print("=== 1. Manual Message History ===\n")

    messages = [
        SystemMessage(content="You are a helpful assistant."),
    ]

    # Turn 1
    messages.append(HumanMessage(content="My favorite color is blue."))
    response = llm.invoke(messages)
    messages.append(AIMessage(content=response.content))
    print(f"User: My favorite color is blue.")
    print(f"Assistant: {response.content}\n")

    # Turn 2
    messages.append(HumanMessage(content="What's my favorite color?"))
    response = llm.invoke(messages)
    messages.append(AIMessage(content=response.content))
    print(f"User: What's my favorite color?")
    print(f"Assistant: {response.content}\n")

    if args.verbose:
        print(f"[DEBUG] Message history has {len(messages)} messages\n")

    print("=== 2. Conversation Buffer Memory ===\n")

    memory = ConversationBufferMemory(return_messages=True)

    # Save conversations
    memory.save_context(
        {"input": "Hi, I'm Alice"},
        {"output": "Hello Alice! Nice to meet you."}
    )
    memory.save_context(
        {"input": "I live in Seattle"},
        {"output": "Seattle is a beautiful city!"}
    )

    if args.verbose:
        print(f"[DEBUG] Buffer memory has {len(memory.chat_memory.messages)} messages")

    # Use memory in a chain
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant."),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}"),
    ])

    chain = prompt | llm

    result = chain.invoke({
        "history": memory.chat_memory.messages,
        "input": "What's my name and where do I live?"
    })

    print(f"Result: {result.content}\n")

    print("=== 3. Conversation Summary Memory ===\n")

    summary_memory = ConversationSummaryMemory(llm=llm, return_messages=True)

    # Add multiple conversations
    conversations = [
        ("Tell me about quantum physics", "Quantum physics studies matter at atomic scale..."),
        ("How does it relate to computing?", "Quantum computing uses quantum bits or qubits..."),
        ("What are practical applications?", "Applications include cryptography, drug discovery..."),
    ]

    for user_input, ai_output in conversations:
        summary_memory.save_context({"input": user_input}, {"output": ai_output})

    if args.verbose:
        print(f"[DEBUG] Summary memory created from {len(conversations)} exchanges")
        print(f"[DEBUG] Current summary: {summary_memory.buffer[:100]}...")

    print(f"Summary: {summary_memory.buffer}\n")

    print("=== 4. Context Window Management ===\n")

    # Demonstrating keeping conversation within context limits
    messages = [SystemMessage(content="You are a helpful assistant. Be concise.")]

    MAX_MESSAGES = 10

    exchanges = [
        "What is 2+2?",
        "What is 5*5?",
        "What is 10-3?",
        "What is 100/4?",
        "What is 8^2?",
        "What is sqrt of 144?",
    ]

    for i, question in enumerate(exchanges, 1):
        messages.append(HumanMessage(content=question))
        response = llm.invoke(messages)
        messages.append(AIMessage(content=response.content))

        if args.verbose:
            print(f"[DEBUG] Turn {i}: {len(messages)} messages in context")

        # Keep only last N messages (plus system message)
        if len(messages) > MAX_MESSAGES:
            messages = [messages[0]] + messages[-(MAX_MESSAGES-1):]
            if args.verbose:
                print(f"[DEBUG] Trimmed to {len(messages)} messages")

    print(f"Final context has {len(messages)} messages")
    print(f"Last question: {exchanges[-1]}")
    print(f"Last answer: {messages[-1].content}\n")

    if args.verbose:
        print("[DEBUG] All memory examples completed")


if __name__ == "__main__":
    main()
