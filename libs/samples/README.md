# LangChain Nova Samples

Sample code demonstrating how to use langchain-nova.

## Setup

Install dependencies:

```bash
cd libs/samples
pip install -e .
```

Or install langchain-nova directly:

```bash
pip install langchain-nova
```

Set environment variables:

```bash
export NOVA_API_KEY="your-api-key"
export NOVA_BASE_URL="https://api.nova.amazon.com/v1"
```

## Available Samples

### Simple Chat Examples

Located in `simple_chat/`:

- **`basic_chat.py`** - Basic usage with invoke
- **`streaming_chat.py`** - Streaming responses
- **`async_chat.py`** - Async operations (invoke and streaming)
- **`conversation.py`** - Multi-turn conversation with context

Run any example:

```bash
cd simple_chat
python basic_chat.py
```

### Interactive Chat

Located in the root:

- **`interactive_chat.py`** - Interactive command-line chat interface

Run it:

```bash
python interactive_chat.py
```

## Documentation

For full documentation, see the [langchain-nova package](../nova/README.md).
