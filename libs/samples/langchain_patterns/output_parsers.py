"""Output parser examples with ChatNova."""

import argparse
from typing import List

from langchain_core.output_parsers import (
    CommaSeparatedListOutputParser,
    JsonOutputParser,
    PydanticOutputParser,
    StrOutputParser,
)
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from langchain_nova import ChatNova


class Person(BaseModel):
    """Person information."""

    name: str = Field(description="Person's name")
    age: int = Field(description="Person's age")
    occupation: str = Field(description="Person's occupation")


class StoryAnalysis(BaseModel):
    """Story analysis."""

    main_characters: List[str] = Field(description="Main characters in the story")
    setting: str = Field(description="Where the story takes place")
    theme: str = Field(description="Primary theme")


def main():
    parser = argparse.ArgumentParser(description="Output parsers with Nova")
    parser.add_argument("--model", type=str, default="nova-pro-v1")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    llm = ChatNova(model=args.model, temperature=0)

    if args.verbose:
        print(f"\n[DEBUG] Using model: {args.model}")
        print("[DEBUG] Demonstrating output parsers\n")

    print("=== 1. String Parser ===\n")

    str_parser = StrOutputParser()
    prompt = ChatPromptTemplate.from_template("What is the capital of {country}?")
    chain = prompt | llm | str_parser

    if args.verbose:
        print("[DEBUG] Using StrOutputParser")

    result = chain.invoke({"country": "Japan"})
    print(f"Result (str): {result}")
    print(f"Type: {type(result)}\n")

    print("=== 2. Comma-Separated List Parser ===\n")

    list_parser = CommaSeparatedListOutputParser()
    prompt = ChatPromptTemplate.from_template(
        "List 5 {category}.\n{format_instructions}"
    )

    if args.verbose:
        print("[DEBUG] Using CommaSeparatedListOutputParser")
        print(f"[DEBUG] Format instructions: {list_parser.get_format_instructions()[:50]}...")

    chain = prompt | llm | list_parser
    result = chain.invoke({
        "category": "programming languages",
        "format_instructions": list_parser.get_format_instructions()
    })
    print(f"Result (list): {result}")
    print(f"Type: {type(result)}\n")

    print("=== 3. JSON Parser ===\n")

    json_parser = JsonOutputParser()
    prompt = ChatPromptTemplate.from_template(
        "Provide information about {topic} in JSON format with keys: name, description, year_created.\n{format_instructions}"
    )

    if args.verbose:
        print("[DEBUG] Using JsonOutputParser")

    chain = prompt | llm | json_parser
    result = chain.invoke({
        "topic": "Python programming language",
        "format_instructions": json_parser.get_format_instructions()
    })
    print(f"Result (dict): {result}")
    print(f"Type: {type(result)}\n")

    print("=== 4. Pydantic Parser ===\n")

    pydantic_parser = PydanticOutputParser(pydantic_object=Person)
    prompt = ChatPromptTemplate.from_template(
        "Generate a fictional person who is a {occupation}.\n{format_instructions}"
    )

    if args.verbose:
        print("[DEBUG] Using PydanticOutputParser with Person model")
        print(f"[DEBUG] Person fields: {list(Person.model_fields.keys())}")

    chain = prompt | llm | pydantic_parser
    result = chain.invoke({
        "occupation": "software engineer",
        "format_instructions": pydantic_parser.get_format_instructions()
    })
    print(f"Result (Person): {result}")
    print(f"  Name: {result.name}")
    print(f"  Age: {result.age}")
    print(f"  Occupation: {result.occupation}")
    print(f"Type: {type(result)}\n")

    print("=== 5. Complex Pydantic Parser ===\n")

    story_parser = PydanticOutputParser(pydantic_object=StoryAnalysis)
    prompt = ChatPromptTemplate.from_template(
        "Analyze this story:\n{story}\n\n{format_instructions}"
    )

    if args.verbose:
        print("[DEBUG] Using PydanticOutputParser with StoryAnalysis model")

    chain = prompt | llm | story_parser
    result = chain.invoke({
        "story": "Alice went to Wonderland and had tea with the Mad Hatter. It was a strange adventure about growing up.",
        "format_instructions": story_parser.get_format_instructions()
    })
    print(f"Result (StoryAnalysis): {result}")
    print(f"  Characters: {result.main_characters}")
    print(f"  Setting: {result.setting}")
    print(f"  Theme: {result.theme}")
    print(f"Type: {type(result)}\n")

    if args.verbose:
        print("[DEBUG] All output parser examples completed")


if __name__ == "__main__":
    main()
