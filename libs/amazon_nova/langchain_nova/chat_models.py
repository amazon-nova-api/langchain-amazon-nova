"""Amazon Nova chat models."""

from __future__ import annotations

import json
import os
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Union,
)

import httpx
import openai
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult
from langchain_core.utils import (
    convert_to_secret_str,
    secret_from_env,
)
from pydantic import (
    ConfigDict,
    Field,
    SecretStr,
    model_validator,
)

from langchain_nova.models import get_model_capabilities, validate_tool_calling


def convert_to_nova_tool(tool: Any) -> Dict[str, Any]:
    """Convert a tool to Nova's tool format.

    Nova uses OpenAI-compatible tool format. This function handles conversion
    from LangChain tools, Pydantic models, or raw dicts.

    Args:
        tool: Tool to convert. Can be:
            - LangChain Tool (with .name, .description, .args_schema)
            - Pydantic BaseModel
            - Dict with OpenAI tool format

    Returns:
        Dict in OpenAI/Nova tool format:
        {
            "type": "function",
            "function": {
                "name": str,
                "description": str,
                "parameters": {...}  # JSON Schema
            }
        }
    """
    # If already in dict format, return as-is
    if isinstance(tool, dict):
        return tool

    # Handle LangChain tools
    if hasattr(tool, "name") and hasattr(tool, "description"):
        from langchain_core.utils.function_calling import convert_to_openai_tool

        return convert_to_openai_tool(tool)

    # Handle Pydantic models
    from pydantic import BaseModel

    if isinstance(tool, type) and issubclass(tool, BaseModel):
        from langchain_core.utils.function_calling import convert_to_openai_tool

        return convert_to_openai_tool(tool)

    # Fallback to langchain converter
    from langchain_core.utils.function_calling import convert_to_openai_tool

    return convert_to_openai_tool(tool)


class ChatNova(BaseChatModel):
    """Amazon Nova chat model integration.

    Amazon Nova models are OpenAI-compatible and accessed via the OpenAI SDK
    pointed at Nova's endpoint.

    Setup:
        Install langchain-nova:
            pip install langchain-nova

        Set environment variables:
            export NOVA_API_KEY="your-api-key"
            export NOVA_BASE_URL="https://api.nova.amazon.com/v1"

    Key init args — completion:
        model: str
            Name of Nova model to use.
        temperature: float
            Sampling temperature.
        max_tokens: Optional[int]
            Max number of tokens to generate.

    Key init args — client:
        api_key: Optional[SecretStr]
            Nova API key. If not passed in will be read from env var NOVA_API_KEY.
        base_url: Optional[str]
            Base URL for API requests. Defaults to Nova endpoint from NOVA_BASE_URL.

    Instantiate:
        .. code-block:: python

            from langchain_nova import ChatNova

            llm = ChatNova(
                model="nova-pro-v1",
                temperature=0.7,
                max_tokens=2048,
            )

    Invoke:
        .. code-block:: python

            messages = [
                ("system", "You are a helpful assistant."),
                ("human", "What is the capital of France?"),
            ]
            llm.invoke(messages)

        .. code-block:: python

            AIMessage(content='The capital of France is Paris.', ...)

    Stream:
        .. code-block:: python

            for chunk in llm.stream(messages):
                print(chunk.content, end="", flush=True)

    Async:
        .. code-block:: python

            await llm.ainvoke(messages)

    Tool calling:
        .. code-block:: python

            from pydantic import BaseModel, Field

            class GetWeather(BaseModel):
                '''Get the weather for a location.'''

                location: str = Field(..., description="City name")

            llm_with_tools = llm.bind_tools([GetWeather])
            llm_with_tools.invoke("What's the weather in Paris?")
    """

    model_name: str = Field(default="nova-pro-v1", alias="model")
    """Model name to use."""

    temperature: float = Field(default=0.7, ge=0.0, le=1.0)
    """Sampling temperature."""

    max_tokens: Optional[int] = Field(default=None, ge=1)
    """Maximum number of tokens to generate."""

    api_key: Optional[Union[SecretStr, str]] = Field(
        default_factory=secret_from_env("NOVA_API_KEY", default=None)
    )
    """Nova API key."""

    base_url: str = Field(
        default_factory=lambda: os.getenv(
            "NOVA_BASE_URL",
            "https://api.nova.amazon.com/v1",
        )
    )
    """Base URL for Nova API."""

    timeout: Optional[float] = Field(default=None, ge=0)
    """Timeout for requests."""

    max_retries: int = Field(default=2, ge=0)
    """Maximum number of retries."""

    streaming: bool = False
    """Whether to stream responses."""

    # Private fields
    client: Any = Field(default=None, exclude=True)
    """OpenAI client instance."""

    async_client: Any = Field(default=None, exclude=True)
    """Async OpenAI client instance."""

    model_config = ConfigDict(
        populate_by_name=True,
        arbitrary_types_allowed=True,
    )

    @model_validator(mode="after")
    def validate_environment(self) -> ChatNova:
        """Validate environment and create OpenAI client."""
        if self.client is None:
            if self.api_key:
                api_key_str = convert_to_secret_str(self.api_key).get_secret_value()
            else:
                api_key_str = None

            # Create httpx client with no compression to avoid zstd decompression issues
            http_client = httpx.Client(headers={"Accept-Encoding": "identity"})

            self.client = openai.OpenAI(
                api_key=api_key_str,
                base_url=self.base_url,
                timeout=self.timeout,
                max_retries=self.max_retries,
                http_client=http_client,
            )

        if self.async_client is None:
            if self.api_key:
                api_key_str = convert_to_secret_str(self.api_key).get_secret_value()
            else:
                api_key_str = None

            # Create httpx client with no compression to avoid zstd decompression issues
            http_client = httpx.AsyncClient(
                headers={"Accept-Encoding": "identity"}, timeout=httpx.Timeout(60)
            )

            self.async_client = openai.AsyncOpenAI(
                api_key=api_key_str,
                base_url=self.base_url,
                timeout=self.timeout,
                max_retries=self.max_retries,
                http_client=http_client,
            )

        return self

    @property
    def capabilities(self):
        """Get capabilities for the current model."""
        from langchain_nova.models import get_model_capabilities

        return get_model_capabilities(self.model_name)

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "nova-chat"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Return a dictionary of identifying parameters."""
        return {
            "model_name": self.model_name,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "base_url": self.base_url,
        }

    @property
    def lc_secrets(self) -> Dict[str, str]:
        """Return secrets for serialization."""
        return {"api_key": "NOVA_API_KEY"}

    def bind_tools(
        self,
        tools: List[Any],
        strict: bool = True,
        **kwargs: Any,
    ) -> ChatNova:
        """Bind tools to the model.

        Args:
            tools: List of tools to bind. Can be LangChain tools, Pydantic models, or dicts.
            strict: If True, validate that the model supports tool calling. Default True.
            **kwargs: Additional arguments passed to the model.

        Returns:
            New ChatNova instance with tools bound.

        Raises:
            ValueError: If strict=True and the model doesn't support tool calling.
        """
        # Validate model supports tool calling if strict mode
        if strict:
            validate_tool_calling(self.model_name)

        formatted_tools = [convert_to_nova_tool(tool) for tool in tools]
        return self.bind(tools=formatted_tools, **kwargs)

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Any:
        """Not implemented yet for Nova."""
        raise NotImplementedError(
            "with_structured_output is not yet implemented for ChatNova"
        )

    def _convert_messages_to_nova_format(
        self, messages: List[BaseMessage]
    ) -> List[Dict[str, Any]]:
        """Convert LangChain messages to OpenAI format."""
        openai_messages = []
        for message in messages:
            if hasattr(message, "type"):
                role = {
                    "human": "user",
                    "ai": "assistant",
                    "system": "system",
                    "tool": "tool",
                }.get(message.type, "user")
            else:
                role = "user"

            msg_dict: Dict[str, Any] = {
                "role": role,
            }

            # Only set content if it's not empty
            # Tool calls might have empty content
            if message.content:
                msg_dict["content"] = message.content
            elif not (
                message.type == "ai"
                and hasattr(message, "tool_calls")
                and message.tool_calls
            ):
                # If content is empty and it's not an AI message with tool calls, set empty string
                msg_dict["content"] = ""

            # Handle tool calls in AI messages
            if message.type == "ai" and hasattr(message, "tool_calls"):
                if message.tool_calls:
                    msg_dict["tool_calls"] = [
                        {
                            "id": tc.get("id", ""),
                            "type": "function",
                            "function": {
                                "name": tc["name"],
                                "arguments": json.dumps(tc.get("args", {}))
                                if isinstance(tc.get("args"), dict)
                                else tc.get("args", ""),
                            },
                        }
                        for tc in message.tool_calls
                    ]

            # Handle tool message IDs
            if isinstance(message, ToolMessage):
                msg_dict["tool_call_id"] = message.tool_call_id

            openai_messages.append(msg_dict)

        return openai_messages

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate chat completion."""
        openai_messages = self._convert_messages_to_nova_format(messages)

        params = {
            "model": self.model_name,
            "messages": openai_messages,
            "temperature": self.temperature,
            "stream": False,
            **kwargs,
        }

        if self.max_tokens is not None:
            params["max_tokens"] = self.max_tokens

        if stop is not None:
            params["stop"] = stop

        response = self.client.chat.completions.create(**params)

        choice = response.choices[0]
        message_data: Dict[str, Any] = {
            "content": choice.message.content or "",
            "response_metadata": {
                "model": response.model,
                "finish_reason": choice.finish_reason,
            },
        }

        # Handle tool calls
        if hasattr(choice.message, "tool_calls") and choice.message.tool_calls:
            message_data["tool_calls"] = [
                {
                    "id": tc.id,
                    "name": tc.function.name,
                    "args": json.loads(tc.function.arguments)
                    if tc.function.arguments
                    else {},
                }
                for tc in choice.message.tool_calls
            ]

        # Handle function calls (legacy format)
        if hasattr(choice.message, "function_call") and choice.message.function_call:
            message_data["additional_kwargs"] = {
                "function_call": {
                    "name": choice.message.function_call.name,
                    "arguments": choice.message.function_call.arguments,
                }
            }

        message = AIMessage(**message_data)

        # Add usage metadata if available
        if hasattr(response, "usage") and response.usage:
            message.usage_metadata = {
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Async generate chat completion."""
        openai_messages = self._convert_messages_to_nova_format(messages)

        params = {
            "model": self.model_name,
            "messages": openai_messages,
            "temperature": self.temperature,
            "stream": False,
            **kwargs,
        }

        if self.max_tokens is not None:
            params["max_tokens"] = self.max_tokens

        if stop is not None:
            params["stop"] = stop

        response = await self.async_client.chat.completions.create(**params)

        choice = response.choices[0]
        message_data: Dict[str, Any] = {
            "content": choice.message.content or "",
            "response_metadata": {
                "model": response.model,
                "finish_reason": choice.finish_reason,
            },
        }

        # Handle tool calls
        if hasattr(choice.message, "tool_calls") and choice.message.tool_calls:
            message_data["tool_calls"] = [
                {
                    "id": tc.id,
                    "name": tc.function.name,
                    "args": json.loads(tc.function.arguments)
                    if tc.function.arguments
                    else {},
                }
                for tc in choice.message.tool_calls
            ]

        # Handle function calls (legacy format)
        if hasattr(choice.message, "function_call") and choice.message.function_call:
            message_data["additional_kwargs"] = {
                "function_call": {
                    "name": choice.message.function_call.name,
                    "arguments": choice.message.function_call.arguments,
                }
            }

        message = AIMessage(**message_data)

        # Add usage metadata if available
        if hasattr(response, "usage") and response.usage:
            message.usage_metadata = {
                "input_tokens": response.usage.prompt_tokens,
                "output_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens,
            }

        generation = ChatGeneration(message=message)
        return ChatResult(generations=[generation])

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Any:
        """Stream chat completion."""
        openai_messages = self._convert_messages_to_nova_format(messages)

        params = {
            "model": self.model_name,
            "messages": openai_messages,
            "temperature": self.temperature,
            "stream": True,
            **kwargs,
        }

        if self.max_tokens is not None:
            params["max_tokens"] = self.max_tokens

        if stop is not None:
            params["stop"] = stop

        for chunk in self.client.chat.completions.create(**params):
            if not chunk.choices:
                continue

            choice = chunk.choices[0]
            if choice.delta.content:
                message_chunk = AIMessageChunk(content=choice.delta.content)

                if run_manager:
                    run_manager.on_llm_new_token(
                        choice.delta.content,
                        chunk=ChatGenerationChunk(message=message_chunk),
                    )

                yield ChatGenerationChunk(message=message_chunk)

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Any:
        """Async stream chat completion."""
        openai_messages = self._convert_messages_to_nova_format(messages)

        params = {
            "model": self.model_name,
            "messages": openai_messages,
            "temperature": self.temperature,
            "stream": True,
            **kwargs,
        }

        if self.max_tokens is not None:
            params["max_tokens"] = self.max_tokens

        if stop is not None:
            params["stop"] = stop

        stream = await self.async_client.chat.completions.create(**params)
        async for chunk in stream:
            if not chunk.choices:
                continue

            choice = chunk.choices[0]
            if choice.delta.content:
                message_chunk = AIMessageChunk(content=choice.delta.content)

                if run_manager:
                    await run_manager.on_llm_new_token(
                        choice.delta.content,
                        chunk=ChatGenerationChunk(message=message_chunk),
                    )

                yield ChatGenerationChunk(message=message_chunk)


__all__ = ["ChatNova"]
