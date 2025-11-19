"""Amazon Nova chat models."""

from __future__ import annotations

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
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage
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
    def validate_environment(self) -> "ChatNova":
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
            http_client = httpx.AsyncClient(timeout=httpx.Timeout(60))

            self.async_client = openai.AsyncOpenAI(
                api_key=api_key_str,
                base_url=self.base_url,
                timeout=self.timeout,
                max_retries=self.max_retries,
                http_client=http_client,
            )

        return self

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

    def _convert_messages_to_openai_format(
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

            openai_messages.append(
                {
                    "role": role,
                    "content": message.content,
                }
            )

        return openai_messages

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """Generate chat completion."""
        openai_messages = self._convert_messages_to_openai_format(messages)

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

        message = AIMessage(
            content=response.choices[0].message.content or "",
            response_metadata={
                "model": response.model,
                "finish_reason": response.choices[0].finish_reason,
            },
        )

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
        openai_messages = self._convert_messages_to_openai_format(messages)

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

        message = AIMessage(
            content=response.choices[0].message.content or "",
            response_metadata={
                "model": response.model,
                "finish_reason": response.choices[0].finish_reason,
            },
        )

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
        openai_messages = self._convert_messages_to_openai_format(messages)

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
        openai_messages = self._convert_messages_to_openai_format(messages)

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
