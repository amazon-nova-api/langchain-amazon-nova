# ChatAmazonNova Samples

Examples demonstrating LangChain and LangGraph integration with Amazon Nova models.

## Directory Structure

```
samples/
├── langchain/           # LangChain examples
│   ├── python/         # Python scripts with CLI
│   └── jupyter/        # Jupyter notebooks
├── langgraph/          # LangGraph examples
│   ├── python/         # Python scripts with CLI
│   └── jupyter/        # Jupyter notebooks
├── simple_chat/        # Basic chat examples
```

## Environment Setup

All examples require environment variables:

```bash
export NOVA_API_KEY="your-api-key"
export NOVA_BASE_URL="https://api.nova.amazon.com/v1"
```

## Quick Start

Each directory has its own `pyproject.toml` for managing virtual environments:

```bash
# Example: LangGraph Python samples
git clone https://github.com/nova-ai-api/langchain-nova.git
cd samples/simple_chat
python3 -m venv .venv
source .venv/bin/activate
pip install -e .

# Run an example
python interactive_chat.py --verbose --streaming
```

## Documentation

For detailed Nova API documentation and parameters: https://nova.amazon.com/dev/documentation

## Simple Chat Examples

Basic usage patterns located in `simple_chat/`:

- **`basic_chat.py`** - Basic usage with invoke
- **`streaming_chat.py`** - Streaming responses
- **`async_chat.py`** - Async operations (invoke and streaming)
- **`conversation.py`** - Multi-turn conversation with context

```bash
cd simple_chat
python basic_chat.py
```

## LangChain Examples

Foundational patterns for using LangChain with ChatAmazonNova:
- Chains and LCEL composition
- Prompt templates and few-shot learning
- Output parsers for structured data
- Memory and conversation history
- RAG (Retrieval Augmented Generation)

See [`langchain/README.md`](langchain/README.md) for details.

## LangGraph Examples

Advanced stateful and multi-agent patterns:
- Simple stateful chatbot
- Tool-using agents
- Multi-agent collaboration
- Reflection and self-improvement

See [`langgraph/README.md`](langgraph/README.md) for details.

## Python vs Jupyter

- **Python** (`python/` folders): CLI scripts with argument parsing, good for automation and testing
- **Jupyter** (`jupyter/` folders): Interactive notebooks with explanations, good for learning and experimentation

Each has its own dependency management via `pyproject.toml`, so you can keep environments isolated.

## Documentation

For full API documentation, see the [langchain-nova package](../amazon_nova/README.md).
