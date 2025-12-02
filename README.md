# 🦜️🔗 LangChain 🤝 Amazon Nova

This repository contains the official LangChain integration package for Amazon Nova models:

- [langchain-amazon-nova](https://pypi.org/project/langchain-amazon-nova/)

Amazon Nova is a family of state-of-the-art foundation models from Amazon that includes text, multimodal, and image generation capabilities. This integration provides seamless access to Nova models through LangChain's standardized chat model interface.

## Quick Start

```bash
pip install -U langchain-amazon-nova
```

```python
from langchain_amazon_nova import ChatAmazonNova

model = ChatAmazonNova(model="nova-pro-v1", temperature=0.7)
response = model.invoke("What is the capital of France?")
print(response.content)
```

See [libs/amazon_nova/README.md](libs/amazon_nova/README.md) for full documentation.

## Contributing

As an open-source project in a rapidly developing field, we are extremely open to contributions, whether it be in the form of a new feature, improved infrastructure, or better documentation.

For detailed information on how to contribute, see the [Contributing Guide](https://docs.langchain.com/oss/python/contributing/overview).
