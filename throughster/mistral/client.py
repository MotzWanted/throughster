import typing as typ
from collections.abc import Callable
from functools import wraps

import httpx

from throughster.base import ModelInterface
from throughster.core.errors import CompletionError
from throughster.core.models import BaseResponse
from throughster.mistral import models


def validate_completion_response(func: Callable) -> Callable:
    """Validate httpx response from vllm api."""

    @wraps(func)
    def wrapper(self: "MistralInterface", response: httpx.Response) -> httpx.Response:
        try:
            response.raise_for_status()
            return func(self, response)
        except (ValueError, AttributeError, IndexError) as e:
            msg = "Mistral API returned an unexpected completion."
            raise CompletionError(msg) from e
        except httpx.HTTPStatusError as e:
            msg = "Mistral API returned an unexpected status code."
            raise httpx.HTTPStatusError(message=msg, response=e.response, request=e.request) from e

    return wrapper


class MistralInterface(ModelInterface):
    """Class for querying a llm like the openai api hosted with `vllm.entrypoints.openai.api_server`."""

    def list_models(self) -> str:
        """Fetch and cache the available models."""
        client = httpx.Client(base_url=self.api_base[0], headers=self.headers, timeout=self.timeout)
        resp = client.get("/models")
        return str(resp.content)

    def validate_request(self, data: dict[str, typ.Any]) -> dict[str, typ.Any]:
        """Format the request data."""
        m = models.MistralChatRequest(**data)
        return m.model_dump(exclude_none=True)

    def unpack_tool_call(self, choice: models.ChatCompletionChoice) -> str:
        """Unpack the openai tool call message."""
        tool_calls = typ.cast(list[models.ToolCall], choice.message.tool_calls)
        tool_call = models.ToolCall.model_validate(tool_calls[0])
        function = models.Function.model_validate(tool_call.function)
        return function.arguments

    @validate_completion_response
    def unpack_call(self, response: httpx.Response) -> BaseResponse:
        """Unpack the openai message."""
        completion = models.ChatCompletionResponse.model_validate_json(response.text)
        choice = completion.choices[0]
        if choice.message.tool_calls:
            return BaseResponse(
                source=f"api.{completion.model}",
                content=self.unpack_tool_call(choice),
                finish_reason=choice.finish_reason,
                usage=completion.usage,  # type: ignore
            )
        if choice.message.content is None:
            msg = f"Mistral API returned an unexpected completion: {completion}"
            raise ValueError(msg)
        content = choice.message.content
        return BaseResponse(
            source=f"model.{completion.model}",
            content=content,
            finish_reason=choice.finish_reason,
            logprobs=None,
            usage=completion.usage,  # type: ignore
        )

    @property
    def headers(self) -> dict[str, str]:
        return {"Authorization": ("Bearer " + self.api_key)}  # type: ignore

    @property
    def params(self) -> dict[str, str] | None:
        return None

    @validate_completion_response
    async def unpack_stream(self, response: httpx.Response) -> None:
        """Unpack the mistral chunk."""
        raise NotImplementedError
