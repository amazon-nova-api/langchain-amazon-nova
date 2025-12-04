"""Example of using structured output with ChatAmazonNova.

Structured output uses tool calling to force the model to return data
in a specific format defined by a Pydantic model or JSON schema.
"""

from typing import List

from langchain_amazon_nova import ChatAmazonNova
from pydantic import BaseModel, Field


# Define schemas using Pydantic
class Person(BaseModel):
    """A person with basic information."""

    name: str = Field(description="The person's full name")
    age: int = Field(description="The person's age in years")
    occupation: str = Field(description="The person's job or profession")


class Article(BaseModel):
    """A news article with metadata."""

    title: str = Field(description="The article title")
    author: str = Field(description="The article author")
    summary: str = Field(description="A brief summary of the article")
    topics: List[str] = Field(description="Main topics covered in the article")


class Location(BaseModel):
    """A geographic location."""

    city: str = Field(description="City name")
    country: str = Field(description="Country name")
    population: int = Field(description="Approximate population")
    famous_for: List[str] = Field(description="What the location is famous for")


def example_simple_extraction() -> None:
    """Extract structured data about a person."""
    print("=" * 60)
    print("Example 1: Simple Person Extraction")
    print("=" * 60)

    llm = ChatAmazonNova(model="nova-pro-v1", temperature=0)
    structured_llm = llm.with_structured_output(Person)

    text = """
    John Smith is a 35-year-old software engineer who works at a tech startup.
    He has been coding for over 15 years.
    """

    result = structured_llm.invoke(text)

    print(f"\nInput: {text.strip()}")
    print(f"\nExtracted Person:")
    print(f"  Name: {result.name}")
    print(f"  Age: {result.age}")
    print(f"  Occupation: {result.occupation}")
    print()


def example_json_schema() -> None:
    """Use JSON schema instead of Pydantic model."""
    print("=" * 60)
    print("Example 2: Using JSON Schema")
    print("=" * 60)

    llm = ChatAmazonNova(model="nova-pro-v1", temperature=0)

    # Define schema as a dict
    schema = {
        "title": "Person",
        "type": "object",
        "properties": {
            "name": {"type": "string", "description": "The person's name"},
            "age": {"type": "integer", "description": "The person's age"},
            "occupation": {"type": "string", "description": "Their job"},
        },
        "required": ["name", "age", "occupation"],
    }

    structured_llm = llm.with_structured_output(schema)

    text = "Alice Johnson, 28, is a data scientist specializing in machine learning."

    result = structured_llm.invoke(text)

    print(f"\nInput: {text}")
    print(f"\nExtracted Data (dict):")
    print(f"  Name: {result['name']}")
    print(f"  Age: {result['age']}")
    print(f"  Occupation: {result['occupation']}")
    print()


def example_complex_schema() -> None:
    """Extract complex nested data."""
    print("=" * 60)
    print("Example 3: Complex Schema with Lists")
    print("=" * 60)

    llm = ChatAmazonNova(model="nova-pro-v1", temperature=0)
    structured_llm = llm.with_structured_output(Article)

    text = """
    'The Future of AI' by Dr. Sarah Chen explores how artificial intelligence
    is transforming healthcare, education, and transportation. The article
    discusses recent breakthroughs in machine learning and their practical
    applications across various industries.
    """

    result = structured_llm.invoke(text)

    print(f"\nInput: {text.strip()}")
    print(f"\nExtracted Article:")
    print(f"  Title: {result.title}")
    print(f"  Author: {result.author}")
    print(f"  Summary: {result.summary}")
    print(f"  Topics: {', '.join(result.topics)}")
    print()


def example_include_raw() -> None:
    """Get both raw and parsed output."""
    print("=" * 60)
    print("Example 4: Including Raw Output")
    print("=" * 60)

    llm = ChatAmazonNova(model="nova-pro-v1", temperature=0)
    structured_llm = llm.with_structured_output(Person, include_raw=True)

    text = "Bob Williams, 42, is a professional chef."

    result = structured_llm.invoke(text)

    print(f"\nInput: {text}")
    print(f"\nRaw Message Type: {type(result['raw'])}")
    print(f"Tool Calls: {len(result['raw'].tool_calls)}")
    print(f"\nParsed Person:")
    print(f"  Name: {result['parsed'].name}")
    print(f"  Age: {result['parsed'].age}")
    print(f"  Occupation: {result['parsed'].occupation}")
    print()


def example_location_data() -> None:
    """Extract location information."""
    print("=" * 60)
    print("Example 5: Location Information")
    print("=" * 60)

    llm = ChatAmazonNova(model="nova-pro-v1", temperature=0)
    structured_llm = llm.with_structured_output(Location)

    text = """
    Paris, the capital of France, has a population of about 2.1 million people.
    It's famous for the Eiffel Tower, the Louvre Museum, Notre-Dame Cathedral,
    and its world-class cuisine.
    """

    result = structured_llm.invoke(text)

    print(f"\nInput: {text.strip()}")
    print(f"\nExtracted Location:")
    print(f"  City: {result.city}")
    print(f"  Country: {result.country}")
    print(f"  Population: {result.population:,}")
    print(f"  Famous For:")
    for item in result.famous_for:
        print(f"    - {item}")
    print()


async def example_async() -> None:
    """Demonstrate async structured output."""
    print("=" * 60)
    print("Example 6: Async Structured Output")
    print("=" * 60)

    llm = ChatAmazonNova(model="nova-pro-v1", temperature=0)
    structured_llm = llm.with_structured_output(Person)

    text = "Emma Davis, 31, works as a marketing director."

    result = await structured_llm.ainvoke(text)

    print(f"\nInput: {text}")
    print(f"\nExtracted Person (async):")
    print(f"  Name: {result.name}")
    print(f"  Age: {result.age}")
    print(f"  Occupation: {result.occupation}")
    print()


def main() -> None:
    """Run all examples."""
    example_simple_extraction()
    example_json_schema()
    example_complex_schema()
    example_include_raw()
    example_location_data()

    # Async example
    import asyncio

    asyncio.run(example_async())

    print("=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
