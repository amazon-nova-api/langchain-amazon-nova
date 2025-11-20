"""Amazon Nova chat models."""

from __future__ import annotations

import json
import os
from typing import (
    Any,
    Dict,
    List,
    Literal,
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

from langchain_nova._exceptions import map_http_error_to_nova_exception
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
        max_completion_tokens: Optional[int]
            Max tokens in completion (OpenAI compatible param).
        top_p: Optional[float]
            Nucleus sampling threshold.
        reasoning_effort: Optional[Literal["low", "medium", "high"]]
            Reasoning effort level for reasoning models.
        metadata: Optional[Dict[str, Any]]
            Request metadata for tracking.
        stream_options: Optional[Dict[str, bool]]
            Stream options (e.g., include_usage).

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

    max_completion_tokens: Optional[int] = Field(default=None, ge=1)
    """Maximum tokens in completion (OpenAI compatible)."""

    top_p: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    """Nucleus sampling threshold."""

    reasoning_effort: Optional[Literal["low", "medium", "high"]] = Field(default=None)
    """Reasoning effort level for reasoning models."""

    metadata: Optional[Dict[str, Any]] = Field(default=None)
    """Request metadata for tracking."""

    stream_options: Optional[Dict[str, bool]] = Field(default=None)
    """Stream options, e.g., {'include_usage': True}."""

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
            "max_completion_tokens": self.max_completion_tokens,
            "top_p": self.top_p,
            "reasoning_effort": self.reasoning_effort,
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
        """Convert LangChain messages to OpenAI format.

        Supports both text-only and multimodal (text + images) messages.
        """
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

            # Handle content - can be string or list of content blocks
            if message.content:
                # Check if content is already a list (multimodal message)
                if isinstance(message.content, list):
                    # Content is in multi-part format (LangChain content blocks)
                    # Convert LangChain format to OpenAI format
                    content_blocks = []
                    for block in message.content:
                        if isinstance(block, dict):
                            # LangChain uses "type" for dicts, "block_type" for typed objects
                            block_type = block.get("type") or block.get(
                                "block_type", "text"
                            )

                            if block_type == "text":
                                content_blocks.append(
                                    {"type": "text", "text": block.get("text", "")}
                                )
                            elif block_type == "image":
                                # LangChain image block - can have url or base64
                                # url: direct image URL
                                # base64: base64 encoded image data
                                url = block.get("url")
                                base64_data = block.get("base64")

                                if base64_data:
                                    # Convert base64 to data URL
                                    # Format: data:image/jpeg;base64,{base64_data}
                                    mime_type = block.get("mime_type", "image/jpeg")
                                    image_url = f"data:{mime_type};base64,{base64_data}"
                                elif url:
                                    image_url = url
                                else:
                                    # Skip block if no image data
                                    continue

                                content_blocks.append({
                                    "type": "image_url",
                                    "image_url": {"url": image_url}
                                })
                            elif block_type == "image_url":
                                # OpenAI format image_url block
                                # {"type": "image_url", "image_url": {"url": "..."}}
                                # or {"type": "image_url", "image_url": "..."}
                                image_url = block.get("image_url", {})
                                if isinstance(image_url, dict):
                                    url = image_url.get("url", "")
                                    content_blocks.append({
                                        "type": "image_url",
                                        "image_url": {"url": url}
                                    })
                                else:
                                    # image_url is directly a string
                                    content_blocks.append({
                                        "type": "image_url",
                                        "image_url": {"url": str(image_url)}
                                    })
                        elif hasattr(block, "block_type"):
                            # LangChain content block object (not dict)
                            # Skip for now - needs proper serialization
                            continue
                        elif isinstance(block, str):
                            # String in list - treat as text
                            content_blocks.append({"type": "text", "text": block})

                    msg_dict["content"] = content_blocks
                else:
                    # Simple string content
                    msg_dict["content"] = message.content
            elif not (
                message.type == "ai"
                and hasattr(message, "tool_calls")
                and message.tool_calls
            ):
                # For non-AI messages or AI without tool calls, set empty content
                msg_dict["content"] = ""

            # Handle tool message IDs
            if isinstance(message, ToolMessage):
                msg_dict["tool_call_id"] = message.tool_call_id

            openai_messages.append(msg_dict)

        return openai_messages

    def _merge_params(self, base_kwargs: Dict[str, Any]) -> Dict[str, Any]:
        """Merge model-level params with invoke-level kwargs.

        Invoke-level kwargs take precedence over model-level defaults.

        Args:
            base_kwargs: Kwargs passed to invoke/stream methods

        Returns:
            Merged parameters dict
        """
        params = {}

        # max_completion_tokens takes precedence over max_tokens if both provided
        max_completion = base_kwargs.get(
            "max_completion_tokens", self.max_completion_tokens
        )
        max_tok = base_kwargs.get("max_tokens", self.max_tokens)

        if max_completion is not None:
            params["max_completion_tokens"] = max_completion
        elif max_tok is not None:
            params["max_tokens"] = max_tok

        # Add other optional parameters if they exist
        if (top_p := base_kwargs.get("top_p", self.top_p)) is not None:
            params["top_p"] = top_p
        if (
            reasoning := base_kwargs.get("reasoning_effort", self.reasoning_effort)
        ) is not None:
            params["reasoning_effort"] = reasoning
        if (metadata := base_kwargs.get("metadata", self.metadata)) is not None:
            params["metadata"] = metadata
        if (
            stream_opts := base_kwargs.get("stream_options", self.stream_options)
        ) is not None:
            params["stream_options"] = stream_opts

        return params

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate chat completion."""
        openai_messages = self._convert_messages_to_nova_format(messages)

        # Merge model-level and invoke-level params
        merged_params = self._merge_params(kwargs)

        params = {
            "model": self.model_name,
            "messages": openai_messages,
            "temperature": self.temperature,
            "stream": False,
            **merged_params,
        }

        if stop is not None:
            params["stop"] = stop

        try:
            response = self.client.chat.completions.create(**params)
        except Exception as e:
            # Map OpenAI SDK errors to Nova exceptions
            nova_exception = map_http_error_to_nova_exception(
                e, model_name=self.model_name
            )
            raise nova_exception from e

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

        # Merge model-level and invoke-level params
        merged_params = self._merge_params(kwargs)

        params = {
            "model": self.model_name,
            "messages": openai_messages,
            "temperature": self.temperature,
            "stream": False,
            **merged_params,
        }

        if stop is not None:
            params["stop"] = stop

        try:
            response = await self.async_client.chat.completions.create(**params)
        except Exception as e:
            # Map OpenAI SDK errors to Nova exceptions
            nova_exception = map_http_error_to_nova_exception(
                e, model_name=self.model_name
            )
            raise nova_exception from e

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

        # Merge model-level and invoke-level params
        merged_params = self._merge_params(kwargs)

        params = {
            "model": self.model_name,
            "messages": openai_messages,
            "temperature": self.temperature,
            "stream": True,
            **merged_params,
        }

        if stop is not None:
            params["stop"] = stop

        try:
            stream = self.client.chat.completions.create(**params)
        except Exception as e:
            # Map OpenAI SDK errors to Nova exceptions
            nova_exception = map_http_error_to_nova_exception(
                e, model_name=self.model_name
            )
            raise nova_exception from e

        for chunk in stream:
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

        # Merge model-level and invoke-level params
        merged_params = self._merge_params(kwargs)

        params = {
            "model": self.model_name,
            "messages": openai_messages,
            "temperature": self.temperature,
            "stream": True,
            **merged_params,
        }

        if stop is not None:
            params["stop"] = stop

        try:
            stream = await self.async_client.chat.completions.create(**params)
        except Exception as e:
            # Map OpenAI SDK errors to Nova exceptions
            nova_exception = map_http_error_to_nova_exception(
                e, model_name=self.model_name
            )
            raise nova_exception from e

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
