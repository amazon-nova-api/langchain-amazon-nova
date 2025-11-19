# LangChain Patterns with Nova

Examples demonstrating typical LangChain integration patterns using Amazon Nova.

## Setup

Install dependencies:

```bash
cd libs/samples
pip install -e ".[patterns]"
```

Or install manually:

```bash
pip install langchain-nova langchain langchain-core
```

Set environment variables:

```bash
export NOVA_API_KEY="your-api-key"
export NOVA_BASE_URL="https://api.nova.amazon.com/v1"
```

## Examples

### Chains and LCEL

**`chains_example.py`** - Demonstrates LangChain Expression Language (LCEL)
- Sequential chains
- Parallel execution
- Chain composition
- Runnables and operators

```bash
python chains_example.py
python chains_example.py --verbose
```

### Prompt Templates

**`prompt_templates.py`** - Working with prompt templates
- Simple templates
- Chat prompt templates
- Few-shot examples
- Template variables

```bash
python prompt_templates.py
python prompt_templates.py --verbose
```

### Output Parsers

**`output_parsers.py`** - Structured output parsing
- JSON output parser
- Pydantic output parser
- List parser
- Comma-separated parser

```bash
python output_parsers.py
python output_parsers.py --verbose
```

### Memory and Context

**`memory_example.py`** - Conversation memory management
- Conversation buffer memory
- Conversation summary memory
- Message history
- Context window management

```bash
python memory_example.py
python memory_example.py --verbose
```

### Tool Usage

**`tools_example.py`** - Using tools with Nova (if supported)
- Function calling
- Tool binding
- Custom tools
- Tool execution

```bash
python tools_example.py
python tools_example.py --verbose
```

### Retrieval Augmented Generation (RAG)

**`rag_example.py`** - Basic RAG implementation
- Document loading
- Text splitting
- Vector store (in-memory)
- Retrieval chain

```bash
python rag_example.py
python rag_example.py --verbose
```

## Options

All examples support:
- `--model MODEL` - Choose Nova model (default: nova-pro-v1)
- `--verbose` / `-v` - Enable detailed debug output
- `--help` - Show available options
