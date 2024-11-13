import typing as typ
from collections.abc import AsyncGenerator, Callable
from functools import wraps

import fastapi
import httpx
from openai.types.chat import (
    ChatCompletion,
    ChatCompletionChunk,
    ChatCompletionMessageToolCall,
)
from openai.types.chat.chat_completion import Choice
from openai.types.chat.chat_completion_message_tool_call import Function

from throughster.azure.models import OpenAIChatRequest
from throughster.base import ModelInterface
from throughster.core.errors import AzureContentFilterError, CompletionError, RateLimitError
from throughster.core.models import BaseResponse


def validate_completion_response(func: Callable) -> Callable:
    """Validate httpx response from azure openai api."""

    @wraps(func)
    def wrapper(self: "OpenAiInterface", response: httpx.Response) -> httpx.Response:
        try:
            response.raise_for_status()
            return func(self, response)
        except (ValueError, AttributeError, IndexError) as e:
            msg = "OpenAI API returned an unexpected completion"
            raise CompletionError(msg) from e
        except httpx.HTTPStatusError as e:
            if e.response.status_code == fastapi.status.HTTP_429_TOO_MANY_REQUESTS:
                msg = f"Azure OpenAI API rate limit exceeded for deployment: `{e.request.url}`."
                raise RateLimitError(message="", request=e.request, response=e.response) from e
            if "content_filter_result" in e.response.text.lower():
                msg = "Content was filtered due to Azure or OpenAI's content management policy."
                raise AzureContentFilterError(message=msg, request=e.request, response=e.response) from e
            msg = f"Azure OpenAI API returned an unexpected status code: {response.status_code}."
            raise httpx.HTTPStatusError(message=msg, response=e.response, request=e.request) from e

    return wrapper


class OpenAiInterface(ModelInterface):
    """Generic interface for querying a causal language model hosted by OpenAI."""

    def list_models(self) -> list[str]:
        """Return the available models."""
        raise NotImplementedError("Azure OpenAI API does not support listing models.")

    @property
    def params(self) -> dict[str, str]:
        return {"api-version": self.api_version}  # type: ignore

    @property
    def headers(self) -> dict[str, str]:
        return {"api-key": self.api_key}  # type: ignore

    @property
    def api_base(self) -> str:
        if self.model_name is None:
            return self._api_base
        else:
            return self._api_base + "openai/deployments/" + self.model_name

    def validate_request(self, data: dict[str, typ.Any]) -> dict[str, typ.Any]:
        """Format the request data."""
        m = OpenAIChatRequest(**data)
        return m.model_dump(exclude_none=True)

    def unpack_tool_call(self, choice: Choice) -> str:
        """Unpack the openai tool call message."""
        tool_calls = typ.cast(list[ChatCompletionMessageToolCall], choice.message.tool_calls)
        tool_call = ChatCompletionMessageToolCall.model_validate(tool_calls[0])
        function = Function.model_validate(tool_call.function)
        return function.arguments

    @validate_completion_response
    def unpack_call(self, response: httpx.Response) -> BaseResponse:
        """Unpack the openai message.
        NOTE: implement functionality to allow for n_samples > 1.
        """
        completion = ChatCompletion.model_validate_json(response.text)
        choice = completion.choices[0]
        if choice.message.tool_calls:
            choice.message.content = self.unpack_tool_call(choice)
        # if choice.finish_reason == "content_filter":
        #     msg = "Content was filtered due to Azure or OpenAI's content management policy."
        #     raise AzureContentFilterError(msg, request=response.request, response=response)
        # if choice.message.content is None:
        #     msg = f"OpenAI API returned an unexpected completion: {completion}"
        #     raise ValueError(msg)
        return BaseResponse(
            id=completion.id,
            object="structured.completion",
            created=completion.created,
            model=completion.model,
            choices=[choice.model_dump()],  # type: ignore
            usage=completion.usage.model_dump() if completion.usage else None,  # type: ignore
        )

    @validate_completion_response
    async def unpack_stream(self, response: httpx.Response) -> AsyncGenerator[str, None]:
        """Unpack the openai chunk."""
        async for chunk in response.aiter_lines():
            if "chat.completion.chunk" not in chunk:
                continue
            modified_chunk = chunk.replace("data: ", "").strip()
            chunk_completion = ChatCompletionChunk.model_validate_json(modified_chunk)
            choice_chunk = chunk_completion.choices[0]
            if choice_chunk.finish_reason == "content_filter":
                msg = "Content was filtered due to Azure or OpenAI's content management policy."
                raise AzureContentFilterError(msg, request=response.request, response=response)
            if choice_chunk.delta.content is None:
                continue
            yield choice_chunk.delta.content
