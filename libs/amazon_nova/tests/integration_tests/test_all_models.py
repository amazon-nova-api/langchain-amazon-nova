"""Integration tests that run against all available Nova models."""

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

from langchain_nova import ChatNova


def pytest_generate_tests(metafunc):  # type: ignore
    """Dynamically parametrize tests with available models."""
    if "model_id" in metafunc.fixturenames:
        # Import here to avoid issues if API not available during collection
        from ..conftest import fetch_available_models

        models = fetch_available_models()
        metafunc.parametrize("model_id", models, ids=models)


def test_invoke_all_models(model_id: str) -> None:
    """Test basic invoke works for each available model."""
    llm = ChatNova(model=model_id, temperature=0)

    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="Say 'Hello' and nothing else."),
    ]

    response = llm.invoke(messages)
    assert response.content
    assert len(response.content) > 0


def test_streaming_all_models(model_id: str) -> None:
    """Test streaming works for each available model."""
    llm = ChatNova(model=model_id, temperature=0)

    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="Count to 3."),
    ]

    chunks = []
    for chunk in llm.stream(messages):
        content = (
            chunk.content if isinstance(chunk.content, str) else str(chunk.content)
        )
        chunks.append(content)

    full_response = "".join(chunks)
    assert len(full_response) > 0
    assert len(chunks) > 0


@pytest.mark.asyncio
async def test_ainvoke_all_models(model_id: str) -> None:
    """Test async invoke works for each available model."""
    llm = ChatNova(model=model_id, temperature=0)

    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="Say 'Hello' and nothing else."),
    ]

    response = await llm.ainvoke(messages)
    assert response.content
    assert len(response.content) > 0


@pytest.mark.asyncio
async def test_astreaming_all_models(model_id: str) -> None:
    """Test async streaming works for each available model."""
    llm = ChatNova(model=model_id, temperature=0)

    messages = [
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content="Count to 3."),
    ]

    chunks = []
    async for chunk in llm.astream(messages):
        content = (
            chunk.content if isinstance(chunk.content, str) else str(chunk.content)
        )
        chunks.append(content)

    full_response = "".join(chunks)
    assert len(full_response) > 0
    assert len(chunks) > 0


def test_model_metadata(single_test_model: str) -> None:
    """Test that models return proper metadata (uses single model for speed)."""
    llm = ChatNova(model=single_test_model, temperature=0)

    messages = [HumanMessage(content="Hi")]
    response = llm.invoke(messages)

    # Check response metadata
    assert hasattr(response, "response_metadata")
    assert "model" in response.response_metadata
    assert "finish_reason" in response.response_metadata

    # Check usage metadata
    assert hasattr(response, "usage_metadata")
    assert "input_tokens" in response.usage_metadata  # type: ignore
    assert "output_tokens" in response.usage_metadata  # type: ignore
    assert "total_tokens" in response.usage_metadata  # type: ignore
