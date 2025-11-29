"""Memory and conversation context examples with ChatAmazonNova.

Modern conversation memory patterns for LangChain.
For more details, see:
- Short-term memory: https://python.langchain.com/docs/how_to/chatbots_memory/
- Long-term memory: https://python.langchain.com/docs/how_to/chatbots_long_term_memory/
"""

import argparse

from langchain_amazon_nova import ChatAmazonNova
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, trim_messages
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory


def main():
    parser = argparse.ArgumentParser(description="Memory examples with Nova")
    parser.add_argument("--model", type=str, default="nova-pro-v1")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    llm = ChatAmazonNova(model=args.model, temperature=0.7)

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

    print("=== 2. RunnableWithMessageHistory (Short-term Memory) ===\n")

    # Modern approach using RunnableWithMessageHistory
    store = {}

    def get_session_history(session_id: str):
        if session_id not in store:
            store[session_id] = InMemoryChatMessageHistory()
        return store[session_id]

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant."),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )

    chain = prompt | llm
    with_message_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="messages",
    )

    # Conversation 1
    response = with_message_history.invoke(
        {"messages": [HumanMessage(content="Hi, I'm Alice from Seattle")]},
        config={"configurable": {"session_id": "user_123"}},
    )
    print(f"User: Hi, I'm Alice from Seattle")
    print(f"Assistant: {response.content}\n")

    # Conversation 2 - remembers context
    response = with_message_history.invoke(
        {"messages": [HumanMessage(content="What's my name and where am I from?")]},
        config={"configurable": {"session_id": "user_123"}},
    )
    print(f"User: What's my name and where am I from?")
    print(f"Assistant: {response.content}\n")

    if args.verbose:
        history = get_session_history("user_123")
        print(f"[DEBUG] Session history has {len(history.messages)} messages\n")

    print("=== 3. Message Trimming (Managing Context Window) ===\n")

    # Use trim_messages to manage context window
    messages = [SystemMessage(content="You are a helpful assistant. Be concise.")]

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

    # Trim to keep recent conversation (keep system + last 6 messages)
    trimmed_messages = trim_messages(
        messages,
        max_tokens=100,  # Approximate token budget
        strategy="last",
        token_counter=len,  # Simple counter for demo
        include_system=True,
        allow_partial=False,
    )

    print(f"Original context: {len(messages)} messages")
    print(f"Trimmed context: {len(trimmed_messages)} messages")
    print(f"Last question: {exchanges[-1]}")
    print(f"Last answer: {messages[-1].content}\n")

    print("=== 4. Summarization (Long-term Memory Pattern) ===\n")

    # For long conversations, periodically summarize older messages
    conversation_messages = []

    # Simulate a longer conversation
    conversation_pairs = [
        ("Tell me about quantum physics", "Quantum physics studies matter at atomic scale..."),
        ("How does it relate to computing?", "Quantum computing uses quantum bits or qubits..."),
        ("What are practical applications?", "Applications include cryptography, drug discovery..."),
    ]

    for user_msg, ai_msg in conversation_pairs:
        conversation_messages.append(HumanMessage(content=user_msg))
        conversation_messages.append(AIMessage(content=ai_msg))

    # Create a summary of the conversation
    summary_prompt = ChatPromptTemplate.from_messages([
        ("system", "Provide a concise summary of the conversation below."),
        MessagesPlaceholder(variable_name="messages"),
    ])

    summary_chain = summary_prompt | llm
    summary_response = summary_chain.invoke({"messages": conversation_messages})

    print(f"Conversation summary: {summary_response.content}\n")

    # In a real application, you would:
    # 1. Replace old messages with the summary
    # 2. Store summaries in a database for long-term memory
    # 3. Retrieve relevant summaries based on current context

    if args.verbose:
        print(f"[DEBUG] Summarized {len(conversation_messages)} messages")
        print("[DEBUG] All memory examples completed")


if __name__ == "__main__":
    main()
