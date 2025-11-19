# Simple Chat Examples

Basic examples demonstrating langchain-nova usage patterns.

## Setup

Make sure you have set your environment variables:

```bash
export NOVA_API_KEY="your-api-key"
export NOVA_BASE_URL="https://api.nova.amazon.com/v1"
```

## Examples

All examples support `--model` and `--verbose` flags:

```bash
# Use a different model
python basic_chat.py --model nova-lite-v1

# Enable verbose output for debugging
python basic_chat.py --verbose
python basic_chat.py -v  # short form
```

### basic_chat.py

Simple synchronous invoke with usage metadata tracking.

```bash
python basic_chat.py
python basic_chat.py --verbose
```

Shows:
- Basic model initialization
- Simple invoke pattern
- Usage metadata (token counts)

### streaming_chat.py

Streaming responses for real-time output.

```bash
python streaming_chat.py
python streaming_chat.py --verbose
```

Shows:
- Token-by-token streaming
- Chunk counting in verbose mode

### async_chat.py

Async operations (both invoke and streaming).

```bash
python async_chat.py
python async_chat.py --verbose
```

Shows:
- Async model invocation
- Async streaming
- Performance timing

### conversation.py

Multi-turn conversation with context tracking.

```bash
python conversation.py
python conversation.py --verbose
```

Shows:
- Building conversation history
- Context window management
- Response timing per turn

### interactive_chat.py

Full-featured interactive CLI chat interface.

```bash
python interactive_chat.py
python interactive_chat.py --streaming
python interactive_chat.py --verbose
```

Features:
- Interactive command-line chat
- Conversation history management
- Commands: `/help`, `/clear`, `/history`, `/stats`, `/exit`
- Optional streaming mode
- Conversation statistics tracking
- Full verbose debugging mode

Options:
- `--model MODEL` - Choose Nova model (default: nova-pro-v1)
- `--temperature TEMP` - Set sampling temperature (default: 0.7)
- `--max-tokens N` - Set max tokens (default: 2048)
- `--streaming` - Enable streaming responses
- `--system-message MSG` - Customize system prompt
- `--verbose` / `-v` - Enable detailed debug output

## Verbose Mode

All examples support `-v` or `--verbose` flag which shows:
- Model initialization details
- Request/response timing
- Token usage statistics
- Chunk counts (for streaming)
- API endpoint information
- Full error tracebacks

Example verbose output:

```bash
$ python basic_chat.py -v

[DEBUG] Initializing ChatNova with model: nova-pro-v1
[DEBUG] Model initialized successfully
[DEBUG] Base URL: https://api.nova.amazon.com/v1
[DEBUG] Temperature: 0.7
[DEBUG] Max tokens: 2048

[DEBUG] Sending 2 messages
[DEBUG] Message content: [('system', 'You are a helpful assistant.'), ('human', 'What is the capital of France?')]

Response: The capital of France is Paris.

Tokens used: {'input_tokens': 23, 'output_tokens': 8, 'total_tokens': 31}

[DEBUG] Response time: 0.87s
[DEBUG] Response metadata: {'model': 'nova-pro-v1', 'finish_reason': 'stop'}
```
