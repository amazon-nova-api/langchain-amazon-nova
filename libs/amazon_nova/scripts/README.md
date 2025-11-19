# Nova API Experiment Scripts

Scripts for experimenting with the Nova API.

## Setup

Set your Nova API key:
```bash
export NOVA_API_KEY="your-api-key-here"
```

## Scripts

### nova_chat.py - Interactive Chat

Interactive REPL-style chat interface for conversing with Nova models.

```bash
python samples/scripts/nova_chat.py
```

Commands:
- `/quit` - Exit the chat
- `/reset` - Clear conversation history
- `/history` - Show conversation history
- `/params` - Show current parameters
- `/tools` - Show configured tools

### nova_test.py - Configurable Testing

Test Nova API with custom parameters and messages.

Basic usage:
```bash
python samples/scripts/nova_test.py "What is the weather like?"
```

With options:
```bash
python samples/scripts/nova_test.py "Tell me about Seattle" \
  --temperature 0.9 \
  --max-tokens 500 \
  --system "You are a helpful travel guide"
```

With tools:
```bash
python samples/scripts/nova_test.py "What's the weather in Seattle?" --tools
```

From a messages file:
```bash
python samples/scripts/nova_test.py --messages-file samples/scripts/example_messages.json
```

Without streaming:
```bash
python samples/scripts/nova_test.py "Hello" --no-stream
```

All options:
```
--model MODEL          Model to use (default: nova-pro-v1)
--temperature FLOAT    Temperature (default: 0.7)
--max-tokens INT       Max tokens (default: 10000)
--top-p FLOAT          Top-p (default: 0.7)
--no-stream            Disable streaming
--tools                Include example tools
--tool-choice STRING   Tool choice strategy (default: auto)
--messages-file PATH   JSON file with messages array
--system TEXT          System message to prepend
```

### nova_web_search.py - Web Search Testing

Test Nova API with web search capabilities.

Basic usage:
```bash
python samples/scripts/nova_web_search.py "What's the Amazon stock price?"
```

With system prompt:
```bash
python samples/scripts/nova_web_search.py "What's the Amazon stock?" \
  --system "you are a smart llm"
```

With streaming:
```bash
python samples/scripts/nova_web_search.py "Latest news about AI" --stream
```

With different search context size:
```bash
python samples/scripts/nova_web_search.py "Tell me about quantum computing" \
  --search-context-size high
```

All options:
```
--model MODEL                    Model to use (default: nova-premier-v1)
--system TEXT                    System prompt
--search-context-size {low,medium,high}  Web search context size (default: low)
--temperature FLOAT              Temperature (default: 0.7)
--max-tokens INT                 Max tokens (default: 10000)
--top-p FLOAT                    Top-p (default: 0.7)
--stream                         Enable streaming
```

## Example Messages File

See `example_messages.json` for the multi-turn conversation format.
