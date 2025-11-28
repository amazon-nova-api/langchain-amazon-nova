# AGENTS.md

This file provides guidance for AI coding assistants when working with code in this repository.

## Project Overview

This is a monorepo containing LangChain and LangGraph integrations for Amazon Nova. The repository contains two main packages:

- `langchain-amazon-nova`: Core model provider for Amazon Nova
- `langchain-amazon-nova-samples`: Cookbooks and samples for using Amazon Nova with LangChain

## Development Commands

### For langchain-aws (libs/nova/)

**Setup:**

```bash
cd libs/amazon_nova
make install_dev
```

**Testing:**

```bash
make tests                    # Run all unit tests
make test TEST_FILE=path      # Run specific test file
make integration_tests        # Run all integration tests
make test_watch              # Interactive test watching
```

**Code Quality:**

```bash
make lint                    # Check code with ruff
make format                  # Format code with ruff
make spell_check            # Check spelling
make check_imports          # Validate imports
```

**Coverage:**

```bash
make coverage_tests                    # Unit test coverage
make coverage_integration_tests        # Integration test coverage
```

## Important Notes

- Both packages use uv for dependency management
- Code style enforced by ruff (formatting and linting)
- MyPy used for type checking with strict configuration
- Import validation ensures proper module organization
- Optional tool dependencies must be explicitly installed for browser/code interpreter functionality
