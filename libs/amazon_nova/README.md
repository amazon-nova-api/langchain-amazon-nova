# langchain-amazon-nova

Official LangChain integration for Amazon Nova models.

Amazon Nova is a family of state-of-the-art foundation models from Amazon that includes text, multimodal, and image generation capabilities. This integration provides access to Nova models through LangChain's standardized chat model interface.

## Installation

```bash
pip install -U langchain-amazon-nova
```

Or using uv:

```bash
uv add langchain-amazon-nova
```

## Quick Start

```python
from langchain_amazon_nova import ChatAmazonNova

# Initialize the model
model = ChatAmazonNova(
    model="nova-pro-v1",
    temperature=0.7,
)

# Use it like any LangChain chat model
response = model.invoke("What is the capital of France?")
print(response.content)
```

## Environment Setup

Set your Nova API credentials:

```bash
export NOVA_API_KEY="your-api-key"
export NOVA_BASE_URL="https://api.nova.amazon.com/v1"
```

## Documentation

For detailed API documentation and additional parameters, see: https://nova.amazon.com/dev/documentation

## Supported Models

Amazon Nova offers several model variants:

- **nova-micro-v1**: Fast, efficient text model (128K context)
- **nova-lite-v1**: Lightweight multimodal model (300K context)
- **nova-pro-v1**: Balanced multimodal model (300K context)
- **nova-premier-v1**: Most capable multimodal model (300K context)
- **nova-canvas-v1**: Image generation model
- **nova-deep-research-v1**: Research-focused text model (300K context)

## Features

- **Text Generation**: All models support text completion
- **Streaming**: Token-by-token streaming responses
- **Async Support**: Native async/await support
- **Tool Calling**: Function calling capabilities (most models)
- **Multimodal Input**: Image and video understanding (lite, pro, premier)
- **Image Generation**: Create images from text (canvas)

## Example Usage

### Basic Chat

```python
from langchain_amazon_nova import ChatAmazonNova

model = ChatAmazonNova(model="nova-pro-v1")
messages = [
    ("system", "You are a helpful assistant."),
    ("human", "Explain quantum computing in simple terms."),
]
response = model.invoke(messages)
print(response.content)
```

### Streaming

```python
for chunk in model.stream(messages):
    print(chunk.content, end="", flush=True)
```

### Tool Calling

```python
from pydantic import BaseModel, Field

class GetWeather(BaseModel):
    '''Get weather for a location.'''
    location: str = Field(description="City name")

model_with_tools = model.bind_tools([GetWeather])
response = model_with_tools.invoke("What's the weather in Tokyo?")
print(response.tool_calls)
```

### Async

```python
import asyncio

async def main():
    response = await model.ainvoke(messages)
    print(response.content)

asyncio.run(main())
```

## Model Capabilities

Check model capabilities programmatically:

```python
from langchain_amazon_nova import get_model_capabilities

caps = get_model_capabilities("nova-pro-v1")
print(f"Supports vision: {caps.supports_vision}")
print(f"Supports tools: {caps.supports_tool_calling}")
print(f"Context window: {caps.max_context_tokens}")
```

## 📖 Documentation

- [Integration Guide](https://docs.langchain.com/oss/integrations/chat/amazon_nova) - Complete usage guide
- [API Reference](https://python.langchain.com/api_reference/nova/index.html) - Detailed API documentation
- [Amazon Nova Docs](https://aws.amazon.com/bedrock/nova/) - Official Nova documentation

## 📕 Releases & Versioning

See our [Releases](https://docs.langchain.com/oss/python/release-policy) and [Versioning](https://docs.langchain.com/oss/python/versioning) policies.

## 💁 Contributing

As an open-source project in a rapidly developing field, we are extremely open to contributions, whether it be in the form of a new feature, improved infrastructure, or better documentation.

For detailed information on how to contribute, see the [Contributing Guide](https://docs.langchain.com/oss/python/contributing/overview).
