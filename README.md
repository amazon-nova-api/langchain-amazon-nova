# 🦜️🔗 LangChain 🤝 Amazon Nova

This repository contains the official LangChain integration package for Amazon Nova models:

- [langchain-amazon-nova](https://pypi.org/project/langchain-amazon-nova/)

Amazon Nova is a family of state-of-the-art foundation models from AWS that includes text, multimodal, and image generation capabilities. This integration provides seamless access to Nova models through LangChain's standardized chat model interface.

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

## Repository Setup TODO

### Workflow Configuration
- [ ] Populate `.github/workflows/_release.yml` with `on.workflow_dispatch.inputs.working-directory.default`
- [ ] Add secrets as env vars in `.github/workflows/_release.yml`
- [ ] Update `.github/workflows/api_doc_build.yml` with new code location

### GitHub Settings
- [ ] Add integration testing secrets in GitHub
- [ ] Add partner collaborators in GitHub
- [ ] Enable "Allow auto-merge" in General Settings
- [ ] Set to only "Allow squash merging" in General Settings
- [ ] Set up CI build ruleset:
    - name: ci build
    - enforcement: active
    - bypass: write
    - target: default branch
    - rules: restrict deletions, require status checks ("CI Success"), block force pushes
- [ ] Set up PR requirements ruleset:
    - name: require prs
    - enforcement: active
    - bypass: none
    - target: default branch
    - rules: restrict deletions, require a pull request before merging (0 approvals, no boxes), block force pushes

### PyPI Publishing
- [ ] Add repo to test-pypi and pypi trusted publishing

> [!NOTE]
> Tag [@ccurme](https://github.com/ccurme) if you have questions on any step.

## Contributing

As an open-source project in a rapidly developing field, we are extremely open to contributions, whether it be in the form of a new feature, improved infrastructure, or better documentation.

For detailed information on how to contribute, see the [Contributing Guide](https://docs.langchain.com/oss/python/contributing/overview).
