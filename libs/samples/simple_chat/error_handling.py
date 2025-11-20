"""Example demonstrating error handling with Nova exceptions.

This script shows how to catch and handle different Nova error types.
"""

import argparse

from langchain_nova import (
    ChatNova,
    NovaConfigurationError,
    NovaError,
    NovaModelError,
    NovaModelNotFoundError,
    NovaThrottlingError,
    NovaValidationError,
)


def example_catch_specific_errors():
    """Example: Catch specific error types."""
    print("Example 1: Catching specific error types\n")

    llm = ChatNova(model="invalid-model-xyz", temperature=0.7)

    try:
        llm.invoke("Hello!")
    except NovaModelNotFoundError as e:
        print(f"Model not found: {e}")
        print(f"Model name: {e.model_name}")
        print(f"Status code: {e.status_code}")
    except NovaValidationError as e:
        print(f"Validation error: {e}")
        print(f"Status code: {e.status_code}")
    except NovaError as e:
        print(f"Nova error: {e}")


def example_catch_base_error():
    """Example: Catch all Nova errors with base exception."""
    print("\nExample 2: Catching all Nova errors with base exception\n")

    llm = ChatNova(model="invalid-model", temperature=0.7)

    try:
        llm.invoke("Hello!")
    except NovaError as e:
        print(f"Caught Nova error: {e}")
        print(f"Error type: {type(e).__name__}")
        if hasattr(e, "status_code"):
            print(f"Status code: {e.status_code}")
        if hasattr(e, "model_name"):
            print(f"Model: {e.model_name}")


def example_retry_on_throttle():
    """Example: Retry with backoff on throttling."""
    import time

    print("\nExample 3: Handling throttling with retry\n")

    llm = ChatNova(model="nova-pro-v1", temperature=0.7)

    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = llm.invoke("What is Python?")
            print(f"Success: {response.content[:50]}...")
            break
        except NovaThrottlingError as e:
            if attempt < max_retries - 1:
                wait_time = e.retry_after or (2 ** attempt)
                print(f"Throttled. Waiting {wait_time}s before retry...")
                time.sleep(wait_time)
            else:
                print(f"Failed after {max_retries} attempts: {e}")
        except NovaError as e:
            print(f"Non-throttling error: {e}")
            break


def example_validation_fallback():
    """Example: Fallback to different model on validation error."""
    print("\nExample 4: Fallback on validation error\n")

    models_to_try = ["nova-premier-v1", "nova-pro-v1", "nova-lite-v1"]

    for model in models_to_try:
        try:
            llm = ChatNova(model=model, temperature=0.7)
            response = llm.invoke("Hello!")
            print(f"Success with {model}: {response.content[:50]}...")
            break
        except NovaModelNotFoundError:
            print(f"{model} not available, trying next...")
        except NovaValidationError as e:
            print(f"Validation error with {model}: {e}")
        except NovaError as e:
            print(f"Error with {model}: {e}")


def example_graceful_degradation():
    """Example: Graceful degradation on model errors."""
    print("\nExample 5: Graceful degradation\n")

    try:
        llm = ChatNova(model="nova-pro-v1", temperature=0.7, max_tokens=10)
        response = llm.invoke("Write a long essay about AI")
        print(f"Response: {response.content}")

        if response.response_metadata.get("finish_reason") == "length":
            print("\nResponse was truncated due to max_tokens limit")

            llm_unlimited = ChatNova(model="nova-pro-v1", temperature=0.7)
            full_response = llm_unlimited.invoke("Write a long essay about AI")
            print(f"Full response: {full_response.content[:100]}...")
    except NovaModelError as e:
        print(f"Model error occurred: {e}")
        print("Falling back to cached response or alternative")
    except NovaError as e:
        print(f"Nova error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Demonstrate Nova error handling patterns"
    )
    parser.add_argument(
        "--example",
        type=int,
        choices=[1, 2, 3, 4, 5],
        help="Which example to run (1-5)",
    )

    args = parser.parse_args()

    if args.example == 1 or args.example is None:
        example_catch_specific_errors()

    if args.example == 2 or args.example is None:
        example_catch_base_error()

    if args.example == 3 or args.example is None:
        example_retry_on_throttle()

    if args.example == 4 or args.example is None:
        example_validation_fallback()

    if args.example == 5 or args.example is None:
        example_graceful_degradation()


if __name__ == "__main__":
    main()
