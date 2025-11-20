"""LCEL chains example with ChatNova."""

import argparse

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from langchain_nova import ChatNova


def main():
    parser = argparse.ArgumentParser(description="LangChain chains with Nova")
    parser.add_argument("--model", type=str, default="nova-pro-v1")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--reasoning", type=str, choices=["low", "medium", "high"])
    parser.add_argument("--top-p", type=float, help="Top-p sampling (0.0-1.0)")
    args = parser.parse_args()

    llm = ChatNova(
        model=args.model,
        temperature=0.7,
        reasoning_effort=args.reasoning,
        top_p=args.top_p,
    )

    if args.verbose:
        print(f"\n[DEBUG] Using model: {args.model}")
        print("[DEBUG] Demonstrating LCEL chains\n")

    print("=== 1. Simple Chain ===\n")

    # Simple chain: prompt | model | parser
    prompt = ChatPromptTemplate.from_template("Tell me a joke about {topic}")
    chain = prompt | llm | StrOutputParser()

    if args.verbose:
        print("[DEBUG] Chain structure: prompt | llm | parser")

    result = chain.invoke({"topic": "programming"})
    print(f"Result: {result}\n")

    print("=== 2. Sequential Chain ===\n")

    # Sequential processing with multiple steps
    translate_prompt = ChatPromptTemplate.from_template(
        "Translate this to {language}: {text}"
    )
    summarize_prompt = ChatPromptTemplate.from_template(
        "Summarize in one sentence: {text}"
    )

    translate_chain = translate_prompt | llm | StrOutputParser()
    summarize_chain = summarize_prompt | llm | StrOutputParser()

    if args.verbose:
        print("[DEBUG] Two-step chain: translate -> summarize")

    # Chain them together
    full_chain = (
        {"text": translate_chain, "language": lambda x: "English"}
        | RunnablePassthrough.assign(text=lambda x: x["text"])
        | summarize_chain
    )

    result = full_chain.invoke(
        {
            "text": "LangChain is great for building AI applications",
            "language": "Spanish",
        }
    )
    print(f"Result: {result}\n")

    print("=== 3. Parallel Execution ===\n")

    # Run multiple chains in parallel
    from langchain_core.runnables import RunnableParallel

    joke_chain = (
        ChatPromptTemplate.from_template("Tell a joke about {topic}")
        | llm
        | StrOutputParser()
    )

    poem_chain = (
        ChatPromptTemplate.from_template("Write a haiku about {topic}")
        | llm
        | StrOutputParser()
    )

    parallel_chain = RunnableParallel(joke=joke_chain, poem=poem_chain)

    if args.verbose:
        print("[DEBUG] Running joke and poem chains in parallel")

    results = parallel_chain.invoke({"topic": "clouds"})
    print(f"Joke: {results['joke']}")
    print(f"\nPoem: {results['poem']}\n")

    print("=== 4. Chain with Fallbacks ===\n")

    # Fallback to different model if first fails
    primary = ChatNova(
        model=args.model,
        temperature=0.7,
        max_tokens=500,
        reasoning_effort=args.reasoning,
    )
    fallback_model = ChatNova(model="nova-lite-v1", temperature=0.7, max_tokens=500)

    prompt = ChatPromptTemplate.from_template("What is {thing}?")
    chain_with_fallback = (prompt | primary | StrOutputParser()).with_fallbacks(
        [prompt | fallback_model | StrOutputParser()]
    )

    if args.verbose:
        print(f"[DEBUG] Chain with fallback: {args.model} -> nova-lite-v1")

    result = chain_with_fallback.invoke({"thing": "LangChain"})
    print(f"Result: {result}\n")

    if args.verbose:
        print("[DEBUG] All chain examples completed successfully")


if __name__ == "__main__":
    main()
