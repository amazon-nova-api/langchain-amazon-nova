# LangGraph Patterns with ChatNova

This directory contains examples of using LangGraph with Amazon Nova models. LangGraph enables building stateful, multi-actor applications with LLMs.

## Prerequisites

```bash
pip install langchain-nova langgraph langchain-core
```

Set environment variables:
```bash
export NOVA_API_KEY="your-api-key"
export NOVA_BASE_URL="https://api.nova.amazon.com/v1"
```

## Examples

### 1. Simple Chatbot (`simple_chatbot.py`)
Basic stateful chatbot that remembers conversation history. Demonstrates:
- State management with MessagesState
- Conversation history tracking
- History trimming to manage context window

```bash
python simple_chatbot.py
python simple_chatbot.py --model nova-pro-v1 --verbose --max-history 10
```

### 2. Agent with Tools (`agent_with_tools.py`)
Basic agent that uses tools to answer questions. Demonstrates:
- Tool binding and execution
- Agent reasoning loop
- State management with LangGraph

```bash
python agent_with_tools.py
python agent_with_tools.py --model nova-pro-v1 --verbose
```

### 3. Multi-Agent Collaboration (`multi_agent.py`)
Multiple specialized agents working together. Demonstrates:
- Agent handoffs
- Conditional routing between agents
- Supervisor pattern

```bash
python multi_agent.py
python multi_agent.py --query "Write a poem about space and then translate it to French"
```


### 4. Reflection Pattern (`reflection_agent.py`)
Agent that critiques and improves its own outputs. Demonstrates:
- Self-reflection loops
- Quality improvement iterations
- Multi-step reasoning

```bash
python reflection_agent.py
python reflection_agent.py --max-iterations 3
```

## Jupyter Notebooks

Each example has a corresponding Jupyter notebook in this directory for interactive exploration.

## Common Patterns

All examples support:
- `--model`: Choose Nova model (default: nova-pro-v1)
- `--verbose`: Enable debug output
- `--reasoning`: Set reasoning effort (low/medium/high)
- `--top-p`: Nucleus sampling parameter

## Learn More

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [ChatNova Documentation](../../README.md)
