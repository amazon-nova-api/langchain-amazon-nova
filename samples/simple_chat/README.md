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

Options:
- `--reasoning` - Reasoning effort (low, medium, high)
- `--top-p` - Top-p sampling (0.0-1.0)
- `--max-tokens` - Maximum tokens to generate (default: 2048)

```bash
python basic_chat.py --reasoning high --top-p 0.95
python basic_chat.py --max-tokens 500
```

### streaming_chat.py

Streaming responses for real-time output.

```bash
python streaming_chat.py
python streaming_chat.py --verbose
```

Shows:
- Token-by-token streaming
- Chunk counting in verbose mode

Options:
- `--include-usage` - Include usage metadata in streaming response
- `--reasoning` - Reasoning effort level

```bash
python streaming_chat.py --include-usage --reasoning medium
```

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

### tool_use.py

Function calling / tool use demonstrations.

```bash
python tool_use.py
python tool_use.py --verbose
```

Shows:
- Defining tools with `@tool` decorator
- Binding tools to the model
- Single tool call execution
- Multiple parallel tool calls
- Multi-turn tool conversations
- Tool result handling

Included tools:
- `get_current_weather` - Get weather for a location
- `get_current_time` - Get current time in timezone
- `calculate` - Perform mathematical operations

Examples in the script:
1. Simple weather query with tool call
2. Multiple tool calls (weather + time)
3. Mathematical calculation via tools
4. Multi-turn conversation with tools

**Note:** Tool calling support depends on the Nova model being used. If the model doesn't support tools, it may respond directly without using them.

### advanced_parameters.py

Comprehensive demonstration of all Phase 1 API parameters.

```bash
python advanced_parameters.py
python advanced_parameters.py --demo reasoning
python advanced_parameters.py --demo all
```

Shows:
- `max_tokens` and `max_completion_tokens` usage
- `top_p` nucleus sampling
- `reasoning_effort` levels (low, medium, high)
- `metadata` for request tracking
- `stream_options` for usage data in streaming
- Per-call parameter overrides
- Token limit controls

Available demos:
- `basic` - Basic parameter usage
- `reasoning` - Reasoning effort comparison
- `tokens` - Token limit controls
- `sampling` - Top-p sampling variations
- `streaming` - Streaming with usage metadata
- `overrides` - Per-call parameter overrides
- `metadata` - Request tracking with metadata
- `completion_tokens` - max_completion_tokens vs max_tokens
- `all` - Run all demos (default)

Options:
- `--demo NAME` - Run specific demo
- `--model MODEL` - Choose Nova model
- `--reasoning LEVEL` - Set reasoning effort
- `--top-p VALUE` - Set top-p sampling

Examples:
```bash
# Run all demos
python advanced_parameters.py

# Run specific demo
python advanced_parameters.py --demo reasoning
python advanced_parameters.py --demo streaming

# With custom parameters
python advanced_parameters.py --demo basic --reasoning high
python advanced_parameters.py --demo sampling --top-p 0.95
```

### vision_chat.py

Vision/multimodal capabilities with image inputs.

```bash
python vision_chat.py                # Run all examples
python vision_chat.py --example 1    # Run specific example
python vision_chat.py --example 4 --image path/to/image.jpg  # With local image
```

Shows:
- Single image URL analysis
- Multiple images in one request
- Base64 encoded local images
- Multi-turn conversations with images

Examples:
1. Single Image URL - Simple question with image
2. Multiple Images - Compare two images
3. Base64 Image - Upload local image file
4. Conversation - Multi-turn chat about an image

**Image URL Format:**
```python
from langchain_core.messages import HumanMessage

message = HumanMessage(content=[
    {"type": "text", "text": "What's in this image?"},
    {
        "type": "image_url",
        "image_url": {"url": "https://example.com/image.jpg"}
    }
])
```

**Supported Image Formats:**
- HTTP/HTTPS URLs
- Base64 data URLs (data:image/jpeg;base64,...)
- JPEG, PNG, GIF, WebP

**Models with Vision Support:**
- nova-lite-v1
- nova-pro-v1
- nova-premier-v1

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

[DEBUG] Initializing ChatAmazonNova with model: nova-pro-v1
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

## Error Handling

### error_handling.py

Error handling patterns with Nova-specific exceptions.

```bash
python error_handling.py              # Run all examples
python error_handling.py --example 1  # Run specific example
```

Shows:
- Catching specific error types (NovaModelNotFoundError, NovaValidationError, etc.)
- Catching all errors with base NovaError
- Retry logic for throttling errors (NovaThrottlingError)
- Fallback to different models on errors
- Graceful degradation on model errors

**Exception Types:**

- `NovaError` - Base exception for all Nova errors
- `NovaValidationError` - Invalid parameters (HTTP 400)
- `NovaModelNotFoundError` - Model not found (HTTP 404)
- `NovaThrottlingError` - Rate limit exceeded (HTTP 429)
- `NovaModelError` - Internal model error (HTTP 500)
- `NovaToolCallError` - Tool calling errors
- `NovaConfigurationError` - Configuration issues

**Example Usage:**

```python
from langchain_amazon_nova import ChatAmazonNova, NovaError, NovaModelNotFoundError

try:
    llm = ChatAmazonNova(model="invalid-model")
    llm.invoke("Hello!")
except NovaModelNotFoundError as e:
    print(f"Model {e.model_name} not found (status: {e.status_code})")
except NovaError as e:
    print(f"Nova error: {e}")
```
