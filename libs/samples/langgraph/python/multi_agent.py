"""Multi-agent collaboration using LangGraph and ChatAmazonNova.

This example shows multiple specialized agents working together with a supervisor
to handle complex tasks. The supervisor routes tasks to appropriate agents.
"""

import argparse
from typing import Literal, TypedDict

from langchain_amazon_nova import ChatAmazonNova
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph


class AgentState(TypedDict):
    """State passed between agents."""

    messages: list
    next_agent: str
    task: str


def create_writer_agent(llm):
    """Create a writer agent that generates creative content."""
    system_message = SystemMessage(
        content="""You are a creative writer. Your task is to write engaging,
        well-structured content based on the user's request. Focus on clarity,
        creativity, and proper structure."""
    )

    def writer(state: AgentState) -> dict:
        messages = [system_message] + state["messages"]
        response = llm.invoke(messages)
        return {"messages": state["messages"] + [response], "next_agent": "supervisor"}

    return writer


def create_translator_agent(llm):
    """Create a translator agent that translates text."""
    system_message = SystemMessage(
        content="""You are an expert translator. Translate the provided text
        accurately while preserving the original meaning, tone, and style.
        Only output the translation, no explanations."""
    )

    def translator(state: AgentState) -> dict:
        messages = [system_message] + state["messages"]
        response = llm.invoke(messages)
        return {"messages": state["messages"] + [response], "next_agent": "supervisor"}

    return translator


def create_critic_agent(llm):
    """Create a critic agent that reviews and improves content."""
    system_message = SystemMessage(
        content="""You are a thoughtful critic and editor. Review the provided
        content and suggest specific improvements for clarity, accuracy, and
        engagement. Be constructive and specific."""
    )

    def critic(state: AgentState) -> dict:
        messages = [system_message] + state["messages"]
        response = llm.invoke(messages)
        return {"messages": state["messages"] + [response], "next_agent": "supervisor"}

    return critic


def create_supervisor(llm, verbose: bool = False):
    """Create a supervisor that routes tasks to appropriate agents."""

    system_message = SystemMessage(
        content="""You are a supervisor managing a team of agents: writer, translator, and critic.

        Based on the conversation, decide which agent should act next, or if the task is complete.

        - Use 'writer' for creative writing, content generation, or composition tasks
        - Use 'translator' for translation tasks
        - Use 'critic' for reviewing, editing, or improving existing content
        - Use 'finish' when the task is complete and no more agent work is needed

        Respond with ONLY ONE WORD: writer, translator, critic, or finish."""
    )

    def supervisor_node(state: AgentState) -> dict:
        messages = [system_message] + state["messages"]

        if verbose:
            print(f"\n[Supervisor] Reviewing conversation with {len(state['messages'])} messages")

        response = llm.invoke(messages)
        next_agent = response.content.strip().lower()

        # Validate response
        valid_agents = ["writer", "translator", "critic", "finish"]
        if next_agent not in valid_agents:
            if verbose:
                print(f"[Supervisor] Invalid response '{next_agent}', defaulting to 'finish'")
            next_agent = "finish"

        if verbose:
            print(f"[Supervisor] Routing to: {next_agent}")

        return {"next_agent": next_agent}

    return supervisor_node


def route_to_agent(state: AgentState) -> str:
    """Route to the next agent based on supervisor decision."""
    return state["next_agent"]


def create_multi_agent_graph(llm, verbose: bool = False):
    """Create the multi-agent collaboration graph."""

    # Create agents
    writer = create_writer_agent(llm)
    translator = create_translator_agent(llm)
    critic = create_critic_agent(llm)
    supervisor = create_supervisor(llm, verbose)

    # Create graph
    workflow = StateGraph(AgentState)

    # Add nodes
    workflow.add_node("supervisor", supervisor)
    workflow.add_node("writer", writer)
    workflow.add_node("translator", translator)
    workflow.add_node("critic", critic)

    # Set entry point
    workflow.add_edge(START, "supervisor")

    # Add conditional routing from supervisor
    workflow.add_conditional_edges(
        "supervisor",
        route_to_agent,
        {
            "writer": "writer",
            "translator": "translator",
            "critic": "critic",
            "finish": END,
        },
    )

    # All agents return to supervisor
    workflow.add_edge("writer", "supervisor")
    workflow.add_edge("translator", "supervisor")
    workflow.add_edge("critic", "supervisor")

    return workflow.compile()


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-agent collaboration with LangGraph")
    parser.add_argument("--model", type=str, default="nova-pro-v1", help="Nova model to use")
    parser.add_argument("--query", type=str, help="Task to execute (non-interactive mode)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    parser.add_argument(
        "--reasoning", type=str, choices=["low", "medium", "high"], help="Reasoning effort"
    )
    parser.add_argument("--top-p", type=float, help="Top-p sampling (0.0-1.0)")
    args = parser.parse_args()

    # Initialize model
    llm = ChatAmazonNova(
        model=args.model,
        temperature=0.7,
        reasoning_effort=args.reasoning,
        top_p=args.top_p,
    )

    # Create multi-agent system
    agent_system = create_multi_agent_graph(llm, args.verbose)

    if args.verbose:
        print(f"\n[DEBUG] Using model: {args.model}")
        print("[DEBUG] Agents: writer, translator, critic")
        print("[DEBUG] Supervisor coordinates agent collaboration\n")

    # Run in interactive or non-interactive mode
    if args.query:
        # Non-interactive mode
        print(f"\nTask: {args.query}\n")
        result = agent_system.invoke(
            {
                "messages": [HumanMessage(content=args.query)],
                "next_agent": "supervisor",
                "task": args.query,
            }
        )

        print("\n=== Final Output ===")
        print(result["messages"][-1].content)
        print()
    else:
        # Interactive mode
        print("\n=== Multi-Agent Collaboration ===")
        print("The supervisor will coordinate writer, translator, and critic agents.")
        print("\nExample tasks:")
        print("  - Write a short poem about the ocean")
        print("  - Write a haiku and then translate it to Spanish")
        print("  - Write a product description and have the critic review it")
        print("\nType 'exit' to quit.\n")

        while True:
            user_input = input("Task: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ["exit", "quit"]:
                print("\nGoodbye!\n")
                break

            try:
                result = agent_system.invoke(
                    {
                        "messages": [HumanMessage(content=user_input)],
                        "next_agent": "supervisor",
                        "task": user_input,
                    }
                )

                print("\n=== Result ===")
                print(result["messages"][-1].content)
                print()
            except Exception as e:
                print(f"\nError: {e}\n")


if __name__ == "__main__":
    main()
