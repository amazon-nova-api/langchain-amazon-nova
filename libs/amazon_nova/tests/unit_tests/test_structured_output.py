"""Unit tests for structured output functionality."""

from typing import List
from unittest.mock import Mock, patch

from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field

from langchain_amazon_nova import ChatAmazonNova


class Person(BaseModel):
    """Simple person schema for testing."""

    name: str = Field(description="The person's name")
    age: int = Field(description="The person's age")


class PersonList(BaseModel):
    """List of people for testing."""

    people: List[Person] = Field(description="List of people")


class TestWithStructuredOutput:
    """Tests for with_structured_output method."""

    def test_with_structured_output_pydantic(self) -> None:
        """Test with_structured_output with Pydantic model."""
        llm = ChatAmazonNova(model="nova-pro-v1", api_key="test-key")
        structured_llm = llm.with_structured_output(Person)

        # Verify it returns a Runnable
        assert hasattr(structured_llm, "invoke")
        assert hasattr(structured_llm, "ainvoke")
        assert hasattr(structured_llm, "stream")

    def test_with_structured_output_json_schema(self) -> None:
        """Test with_structured_output with JSON schema dict."""
        llm = ChatAmazonNova(model="nova-pro-v1", api_key="test-key")

        schema = {
            "title": "Person",
            "type": "object",
            "properties": {
                "name": {"type": "string", "description": "The person's name"},
                "age": {"type": "integer", "description": "The person's age"},
            },
            "required": ["name", "age"],
        }

        structured_llm = llm.with_structured_output(schema)

        # Verify it returns a Runnable
        assert hasattr(structured_llm, "invoke")

    def test_with_structured_output_include_raw(self) -> None:
        """Test with_structured_output with include_raw=True."""
        llm = ChatAmazonNova(model="nova-pro-v1", api_key="test-key")
        structured_llm = llm.with_structured_output(Person, include_raw=True)

        # With include_raw=True, should return a Runnable that outputs dict
        assert hasattr(structured_llm, "invoke")

    @patch("openai.OpenAI")
    def test_with_structured_output_invocation(self, mock_openai: Mock) -> None:
        """Test actual invocation of structured output."""
        # Create properly structured mock for tool call
        function_mock = Mock()
        function_mock.name = "Person"
        function_mock.arguments = '{"name": "John", "age": 30}'

        tool_call_mock = Mock()
        tool_call_mock.id = "call_123"
        tool_call_mock.type = "function"
        tool_call_mock.function = function_mock

        # Mock the OpenAI client response
        mock_response = Mock()
        mock_response.choices = [
            Mock(
                message=Mock(
                    content="",
                    tool_calls=[tool_call_mock],
                ),
                finish_reason="tool_calls",
            )
        ]
        mock_response.model = "nova-pro-v1"
        mock_response.usage = Mock(
            prompt_tokens=10, completion_tokens=20, total_tokens=30
        )

        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        llm = ChatAmazonNova(model="nova-pro-v1", api_key="test-key")
        llm.client = mock_client

        structured_llm = llm.with_structured_output(Person)
        result = structured_llm.invoke("John is 30 years old")

        # Verify the result is a Person instance
        assert isinstance(result, Person)
        assert result.name == "John"
        assert result.age == 30

    @patch("openai.OpenAI")
    def test_with_structured_output_json_schema_invocation(
        self, mock_openai: Mock
    ) -> None:
        """Test actual invocation with JSON schema."""
        # Create properly structured mock for tool call
        function_mock = Mock()
        function_mock.name = "Person"
        function_mock.arguments = '{"name": "Jane", "age": 25}'

        tool_call_mock = Mock()
        tool_call_mock.id = "call_123"
        tool_call_mock.type = "function"
        tool_call_mock.function = function_mock

        # Mock the OpenAI client response
        mock_response = Mock()
        mock_response.choices = [
            Mock(
                message=Mock(
                    content="",
                    tool_calls=[tool_call_mock],
                ),
                finish_reason="tool_calls",
            )
        ]
        mock_response.model = "nova-pro-v1"
        mock_response.usage = Mock(
            prompt_tokens=10, completion_tokens=20, total_tokens=30
        )

        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        llm = ChatAmazonNova(model="nova-pro-v1", api_key="test-key")
        llm.client = mock_client

        schema = {
            "title": "Person",
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "age": {"type": "integer"},
            },
            "required": ["name", "age"],
        }

        structured_llm = llm.with_structured_output(schema)
        result = structured_llm.invoke("Jane is 25 years old")

        # Verify the result is a dict
        assert isinstance(result, dict)
        assert result["name"] == "Jane"
        assert result["age"] == 25

    @patch("openai.OpenAI")
    def test_with_structured_output_include_raw_invocation(
        self, mock_openai: Mock
    ) -> None:
        """Test invocation with include_raw=True."""
        # Create properly structured mock for tool call
        function_mock = Mock()
        function_mock.name = "Person"
        function_mock.arguments = '{"name": "Bob", "age": 35}'

        tool_call_mock = Mock()
        tool_call_mock.id = "call_123"
        tool_call_mock.type = "function"
        tool_call_mock.function = function_mock

        # Mock the OpenAI client response
        mock_response = Mock()
        mock_response.choices = [
            Mock(
                message=Mock(
                    content="",
                    tool_calls=[tool_call_mock],
                ),
                finish_reason="tool_calls",
            )
        ]
        mock_response.model = "nova-pro-v1"
        mock_response.usage = Mock(
            prompt_tokens=10, completion_tokens=20, total_tokens=30
        )

        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        llm = ChatAmazonNova(model="nova-pro-v1", api_key="test-key")
        llm.client = mock_client

        structured_llm = llm.with_structured_output(Person, include_raw=True)
        result = structured_llm.invoke("Bob is 35 years old")

        # Verify the result has both raw and parsed
        assert isinstance(result, dict)
        assert "raw" in result
        assert "parsed" in result

        # Raw should be an AIMessage
        assert isinstance(result["raw"], AIMessage)

        # Parsed should be a Person
        assert isinstance(result["parsed"], Person)
        assert result["parsed"].name == "Bob"
        assert result["parsed"].age == 35

    def test_with_structured_output_nested_schema(self) -> None:
        """Test with_structured_output with nested schema."""
        llm = ChatAmazonNova(model="nova-pro-v1", api_key="test-key")
        structured_llm = llm.with_structured_output(PersonList)

        # Verify it returns a Runnable
        assert hasattr(structured_llm, "invoke")
