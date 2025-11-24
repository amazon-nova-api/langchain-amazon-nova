# LangChain Patterns with ChatNova

Examples demonstrating LangChain integration patterns using Amazon Nova models.

## Directory Structure

- **`python/`** - Python script examples with CLI interfaces
- **`jupyter/`** - Jupyter notebook examples (coming soon)

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
```

Set environment variables:

```bash
export NOVA_API_KEY="your-api-key"
export NOVA_BASE_URL="https://api.nova.amazon.com/v1"
```

## Python Examples

Located in `python/` directory:

### Chains and LCEL
**`chains_example.py`** - LangChain Expression Language (LCEL)
- Sequential chains
- Parallel execution
- Chain composition

```bash
python chains_example.py
python chains_example.py --verbose
```

### Prompt Templates
**`prompt_templates.py`** - Working with prompt templates
- Simple templates
- Chat prompt templates
- Few-shot examples

```bash
python prompt_templates.py
```

### Output Parsers
**`output_parsers.py`** - Structured output parsing
- JSON output parser
- Pydantic output parser
- List and CSV parsers

```bash
python output_parsers.py
```

### Memory and Context
**`memory_example.py`** - Conversation memory management
- Conversation buffer memory
- Message history
- Context window management

```bash
python memory_example.py
```

### Retrieval Augmented Generation (RAG)
**`rag_example.py`** - Basic RAG implementation
- Document loading
- Text splitting
- Vector store (in-memory)
- Retrieval chain

```bash
python rag_example.py
```

## Common Options

All Python examples support:
- `--model MODEL` - Choose Nova model (default: nova-pro-v1)
- `--verbose` / `-v` - Enable detailed debug output
- `--help` - Show available options
