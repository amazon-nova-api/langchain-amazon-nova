"""Basic RAG (Retrieval Augmented Generation) example with ChatNova."""

import argparse

from langchain.text_splitter import CharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough

from langchain_nova import ChatNova


def format_docs(docs):
    """Format documents for context."""
    return "\n\n".join(doc.page_content for doc in docs)


def main():
    parser = argparse.ArgumentParser(description="RAG example with Nova")
    parser.add_argument("--model", type=str, default="nova-pro-v1")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    llm = ChatNova(model=args.model, temperature=0)

    if args.verbose:
        print(f"\n[DEBUG] Using model: {args.model}")
        print("[DEBUG] Demonstrating basic RAG pattern\n")

    print("=== 1. Create Sample Documents ===\n")

    # Sample documents about LangChain
    documents = [
        Document(
            page_content="LangChain is a framework for developing applications powered by language models. It provides tools for prompt management, chains, and agents.",
            metadata={"source": "intro", "page": 1}
        ),
        Document(
            page_content="LCEL (LangChain Expression Language) is a declarative way to compose chains. It uses the pipe operator to connect components.",
            metadata={"source": "lcel", "page": 2}
        ),
        Document(
            page_content="LangChain supports multiple model providers including OpenAI, Anthropic, and others. It provides a unified interface.",
            metadata={"source": "providers", "page": 3}
        ),
        Document(
            page_content="Retrieval Augmented Generation (RAG) combines retrieval with generation. Documents are retrieved and used as context.",
            metadata={"source": "rag", "page": 4}
        ),
    ]

    if args.verbose:
        print(f"[DEBUG] Created {len(documents)} documents")
        for doc in documents:
            print(f"[DEBUG]   - {doc.metadata['source']}: {len(doc.page_content)} chars")
        print()

    print("=== 2. Simple Retrieval (Manual) ===\n")

    # Manual retrieval based on keyword matching
    query = "What is LCEL?"

    def simple_retriever(query: str):
        """Simple keyword-based retrieval."""
        relevant = [doc for doc in documents if "LCEL" in doc.page_content]
        return relevant

    retrieved_docs = simple_retriever(query)

    if args.verbose:
        print(f"[DEBUG] Query: '{query}'")
        print(f"[DEBUG] Retrieved {len(retrieved_docs)} documents")

    # Build prompt with retrieved context
    context = format_docs(retrieved_docs)
    prompt = ChatPromptTemplate.from_template(
        "Answer the question based on the following context:\n\n{context}\n\nQuestion: {question}"
    )

    chain = prompt | llm | StrOutputParser()
    result = chain.invoke({"context": context, "question": query})

    print(f"Question: {query}")
    print(f"Answer: {result}\n")

    print("=== 3. RAG Chain with Runnable ===\n")

    # More sophisticated RAG chain
    def retrieve_top_k(query: str, k: int = 2):
        """Retrieve top k documents by simple keyword matching."""
        query_words = set(query.lower().split())

        scored_docs = []
        for doc in documents:
            doc_words = set(doc.page_content.lower().split())
            score = len(query_words & doc_words)
            scored_docs.append((score, doc))

        scored_docs.sort(reverse=True, key=lambda x: x[0])
        return [doc for score, doc in scored_docs[:k]]

    query = "How does LangChain support different models?"

    if args.verbose:
        print(f"[DEBUG] Query: '{query}'")

    retrieved = retrieve_top_k(query, k=2)

    if args.verbose:
        print(f"[DEBUG] Retrieved {len(retrieved)} documents:")
        for doc in retrieved:
            print(f"[DEBUG]   - {doc.metadata['source']}")

    # RAG chain
    rag_prompt = ChatPromptTemplate.from_template(
        """Use the following context to answer the question. If the context doesn't contain the answer, say so.

Context:
{context}

Question: {question}

Answer:"""
    )

    rag_chain = (
        {"context": lambda x: format_docs(retrieve_top_k(x["question"])),
         "question": RunnablePassthrough()}
        | rag_prompt
        | llm
        | StrOutputParser()
    )

    result = rag_chain.invoke({"question": query})
    print(f"\nQuestion: {query}")
    print(f"Answer: {result}\n")

    print("=== 4. Multi-Query RAG ===\n")

    # Generate multiple queries and retrieve for each
    multi_query_prompt = ChatPromptTemplate.from_template(
        "Generate 2 different versions of this question:\n{question}\n\nReturn only the questions, one per line."
    )

    original_query = "What is LangChain used for?"

    if args.verbose:
        print(f"[DEBUG] Original query: '{original_query}'")

    # Generate alternative queries
    queries_result = (multi_query_prompt | llm | StrOutputParser()).invoke(
        {"question": original_query}
    )
    alternative_queries = [q.strip() for q in queries_result.split("\n") if q.strip()]

    if args.verbose:
        print(f"[DEBUG] Generated {len(alternative_queries)} alternative queries")
        for q in alternative_queries:
            print(f"[DEBUG]   - {q}")

    # Retrieve for all queries
    all_retrieved = []
    for q in [original_query] + alternative_queries[:2]:
        all_retrieved.extend(retrieve_top_k(q, k=1))

    # Deduplicate by source
    unique_docs = {doc.metadata["source"]: doc for doc in all_retrieved}.values()

    if args.verbose:
        print(f"[DEBUG] Total unique documents retrieved: {len(unique_docs)}")

    # Answer based on combined context
    context = format_docs(unique_docs)
    answer = (
        ChatPromptTemplate.from_template(
            "Based on this context:\n{context}\n\nAnswer: {question}"
        )
        | llm
        | StrOutputParser()
    ).invoke({"context": context, "question": original_query})

    print(f"\nQuestion: {original_query}")
    print(f"Answer: {answer}\n")

    if args.verbose:
        print("[DEBUG] All RAG examples completed")


if __name__ == "__main__":
    main()
