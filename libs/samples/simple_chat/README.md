# Simple Chat Example

This example demonstrates basic usage of ChatNova as an end user would experience it after installing from PyPI.

## Setup

1. **Install the package** (simulating PyPI install):
   ```bash
   cd langchain-nova/libs/nova
   pip install -e .
   ```

2. **Set environment variables**:
   ```bash
   export NOVA_API_KEY="your-api-key"
   export NOVA_BASE_URL="https://your-nova-endpoint/v1"
   ```

3. **Run the example**:
   ```bash
   cd samples/simple_chat
   python basic_chat.py
   ```

## What's Included

- `basic_chat.py` - Simple invoke example
- `streaming_chat.py` - Streaming response example
- `async_chat.py` - Async usage example
- `conversation.py` - Multi-turn conversation example

## Notes

All examples use the package as it would be imported from PyPI:
```python
from langchain_nova import ChatNova
```

No local imports or relative paths are used.
