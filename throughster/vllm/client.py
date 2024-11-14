import typing as typ
from collections.abc import AsyncGenerator, Callable
from functools import wraps

import httpx

from throughster.base import ModelInterface
from throughster.core.errors import CompletionError
from throughster.core.models import BaseResponse, ModelCard
from throughster.vllm import models


def validate_completion_response(func: Callable) -> Callable:
    """Validate httpx response from vllm api."""

    @wraps(func)
    def wrapper(self: "VllmOpenAiInterface", response: httpx.Response) -> httpx.Response:
        try:
            response.raise_for_status()
            return func(self, response)
        except (ValueError, AttributeError, IndexError) as e:
            msg = "vLLM returned an unexpected completion."
            raise CompletionError(msg) from e
        except httpx.HTTPStatusError as e:
            msg = "vLLM API returned an unexpected status code."
            raise httpx.HTTPStatusError(message=msg, response=e.response, request=e.request) from e

    return wrapper


class VllmOpenAiInterface(ModelInterface):
    """Class for querying a llm like the openai api hosted with `vllm.entrypoints.openai.api_server`."""

    def list_models(self) -> list[ModelCard]:
        """Fetch and cache the available models."""
        client = httpx.Client(base_url=self.api_base, headers=self.headers, timeout=self.timeout)
        resp = client.get("/models")
        available_models = models.AvailableModels.model_validate_json(resp.content)
        return available_models.data

    @property
    def headers(self) -> dict[str, str] | None:
        if self.api_key is None or self.api_key == "":
            return None
        else:
            return {"Authorization": f"Bearer {self.api_key}"}

    @property
    def params(self) -> dict[str, str] | None:
        return None

    def validate_request(self, data: dict[str, typ.Any]) -> dict[str, typ.Any]:
        """Format the request data."""
        if "model" not in data:
            data["model"] = self.model_name
        m = models.VllmRequest(**data)
        return m.model_dump(exclude_none=True)

    @validate_completion_response
    def unpack_call(self, response: httpx.Response) -> BaseResponse:
        """Unpack the openai message."""
        return BaseResponse.model_validate_json(response.text)

    @validate_completion_response
    async def unpack_stream(self, response: httpx.Response) -> AsyncGenerator[str, None]:
        """Unpack the openai chunk."""
        async for chunk in response.aiter_lines():
            if "chat.completion.chunk" not in chunk:
                continue
            modified_chunk = chunk.replace("data: ", "").strip()
            chunk_completion = models.ChatCompletionStreamResponse.model_validate_json(modified_chunk)
            choice_chunk = chunk_completion.choices[0]
            if choice_chunk.delta.content is None:
                continue
            yield choice_chunk.delta.content
