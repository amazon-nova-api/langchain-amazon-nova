# LangGraph Patterns with ChatNova

Examples demonstrating LangGraph integration patterns using Amazon Nova models. LangGraph enables building stateful, multi-actor applications with LLMs.

## Directory Structure

- **`python/`** - Python script examples with CLI interfaces
- **`jupyter/`** - Interactive Jupyter notebook examples

## Setup

Navigate to the folder you want to use and create a virtual environment:

```bash
# For Python examples
cd python/
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install -e .

# For Jupyter notebooks
cd jupyter/
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
jupyter notebook
```

Set environment variables:

```bash
export NOVA_API_KEY="your-api-key"
export NOVA_BASE_URL="https://api.nova.amazon.com/v1"
```

## Documentation

For detailed Nova API documentation and parameters: https://nova.amazon.com/dev/documentation

## Python Examples

Located in `python/` directory:

### 1. Simple Chatbot
**`simple_chatbot.py`** - Stateful chatbot with conversation history
- State management with MessagesState
- Conversation history tracking
- History trimming to manage context window

```bash
python simple_chatbot.py
python simple_chatbot.py --model nova-pro-v1 --verbose --max-history 10
```

### 2. Agent with Tools
**`agent_with_tools.py`** - Tool-using agent
- Tool binding and execution
- Agent reasoning loop
- Weather, calculator, and search tools

```bash
python agent_with_tools.py
python agent_with_tools.py --model nova-pro-v1 --verbose
```

### 3. Multi-Agent Collaboration
**`multi_agent.py`** - Multiple specialized agents working together
- Agent handoffs and routing
- Supervisor pattern
- Writer, translator, and critic agents

```bash
python multi_agent.py
python multi_agent.py --query "Write a poem about space and then translate it to French"
```

### 4. Reflection Pattern
**`reflection_agent.py`** - Self-improving agent
- Generate-critique-revise loops
- Quality improvement iterations
- Multi-step reasoning

```bash
python reflection_agent.py
python reflection_agent.py --max-iterations 3
```

## Jupyter Notebooks

Located in `jupyter/` directory. Interactive versions of all Python examples with explanatory markdown and step-by-step execution:

- `simple_chatbot.ipynb`
- `agent_with_tools.ipynb`
- `multi_agent.ipynb`
- `reflection_agent.ipynb`

## Common Options

All Python examples support:
- `--model MODEL` - Choose Nova model (default: nova-pro-v1)
- `--verbose` / `-v` - Enable debug output
- `--reasoning EFFORT` - Set reasoning effort (low/medium/high)
- `--top-p VALUE` - Nucleus sampling parameter (0.0-1.0)
- `--query QUERY` - Non-interactive mode with single query

## Learn More

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [ChatNova Documentation](../../amazon_nova/README.md)
