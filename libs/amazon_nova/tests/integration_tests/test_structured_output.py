"""Integration tests for structured output functionality."""

from typing import List

import pytest
from pydantic import BaseModel, Field

from langchain_amazon_nova import ChatAmazonNova


class Person(BaseModel):
    """A person with name and age."""

    name: str = Field(description="The person's name")
    age: int = Field(description="The person's age in years")


class People(BaseModel):
    """A list of people."""

    people: List[Person] = Field(description="List of people mentioned")


class Location(BaseModel):
    """A location with city and country."""

    city: str = Field(description="The city name")
    country: str = Field(description="The country name")


@pytest.fixture
def llm() -> ChatAmazonNova:
    """Create a ChatAmazonNova instance for testing."""
    return ChatAmazonNova(model="nova-pro-v1", temperature=0)


class TestStructuredOutputIntegration:
    """Integration tests for with_structured_output."""

    def test_structured_output_simple_pydantic(self, llm: ChatAmazonNova) -> None:
        """Test structured output with simple Pydantic model."""
        structured_llm = llm.with_structured_output(Person)
        result = structured_llm.invoke("John is 30 years old")

        assert isinstance(result, Person)
        assert result.name == "John"
        assert result.age == 30

    def test_structured_output_json_schema(self, llm: ChatAmazonNova) -> None:
        """Test structured output with JSON schema."""
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
        result = structured_llm.invoke("Alice is 25 years old")

        assert isinstance(result, dict)
        assert result["name"] == "Alice"
        assert result["age"] == 25

    def test_structured_output_nested_schema(self, llm: ChatAmazonNova) -> None:
        """Test structured output with nested schema."""
        structured_llm = llm.with_structured_output(People)
        result = structured_llm.invoke("John is 30 years old and Alice is 25 years old")

        assert isinstance(result, People)
        assert len(result.people) == 2
        assert result.people[0].name == "John"
        assert result.people[0].age == 30
        assert result.people[1].name == "Alice"
        assert result.people[1].age == 25

    def test_structured_output_include_raw(self, llm: ChatAmazonNova) -> None:
        """Test structured output with include_raw=True."""
        structured_llm = llm.with_structured_output(Person, include_raw=True)
        result = structured_llm.invoke("Bob is 40 years old")

        assert isinstance(result, dict)
        assert "raw" in result
        assert "parsed" in result

        # Check raw message
        from langchain_core.messages import AIMessage

        assert isinstance(result["raw"], AIMessage)
        assert len(result["raw"].tool_calls) > 0

        # Check parsed output
        assert isinstance(result["parsed"], Person)
        assert result["parsed"].name == "Bob"
        assert result["parsed"].age == 40

    async def test_structured_output_async(self, llm: ChatAmazonNova) -> None:
        """Test async structured output."""
        structured_llm = llm.with_structured_output(Person)
        result = await structured_llm.ainvoke("Sarah is 28 years old")

        assert isinstance(result, Person)
        assert result.name == "Sarah"
        assert result.age == 28

    @pytest.mark.xfail(
        reason=(
            "Streaming structured output not supported - "
            "tool calls only arrive in final chunk"
        )
    )
    def test_structured_output_streaming(self, llm: ChatAmazonNova) -> None:
        """Test structured output with streaming."""
        structured_llm = llm.with_structured_output(Person)

        # Streaming should accumulate and parse at the end
        chunks = list(structured_llm.stream("Mike is 35 years old"))

        # Should have at least one chunk
        assert len(chunks) > 0

        # Final result should be a Person
        final = chunks[-1]
        assert isinstance(final, Person)
        assert final.name == "Mike"
        assert final.age == 35

    def test_structured_output_different_types(self, llm: ChatAmazonNova) -> None:
        """Test structured output with different data types."""
        structured_llm = llm.with_structured_output(Location)
        result = structured_llm.invoke("The capital of France is Paris")

        assert isinstance(result, Location)
        assert result.city == "Paris"
        assert result.country == "France"

    def test_structured_output_with_description(self, llm: ChatAmazonNova) -> None:
        """Test that field descriptions help with extraction."""

        class Product(BaseModel):
            """A product with price."""

            name: str = Field(description="The product name")
            price: float = Field(description="The price in USD")

        structured_llm = llm.with_structured_output(Product)
        result = structured_llm.invoke("The iPhone costs $999")

        assert isinstance(result, Product)
        assert "iPhone" in result.name or "iphone" in result.name.lower()
        assert result.price == 999.0
