import typing as typ
from collections.abc import AsyncGenerator, Callable
from functools import wraps

import httpx

from throughster.base import ModelInterface
from throughster.core.decorators import _nlg_evaluation
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
        completion = models.ChatCompletionResponse.model_validate_json(response.text)
        choice = completion.choices[0]
        logprobs = typ.cast(models.LogProbs, choice.logprobs)
        return BaseResponse(
            source=f"model.{completion.model}",
            content=choice.message,
            finish_reason=choice.finish_reason,
            logprobs=logprobs.top_logprobs if logprobs else None,
            usage=completion.usage,  # type: ignore
        )

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

    async def nlg_evaluation(
        self, request: dict[str, typ.Any], protocol: str | list[dict[str, str]], scores: list[int]
    ) -> tuple[BaseResponse, float]:
        """Evaluates the quality of a natural language generation (NLG) response using a specified protocol and scoring
        range, normalizing the scores with the probabilities of output tokens from the LLM.

        This method calculates the probability of each predefined score and computes a weighted summation to derive a
        fine-grained, continuous final score that better reflects the quality and diversity of the generated texts.

        Args:
            `request (dict[str, typ.Any])`: A dictionary containing the parameters required for the API call.
            `protocol (str | list[dict[str, str]])`: A protocol string or a list of chat messages defining the evaluation criteria
            and instructions. The protocol provides context and guidelines for assessing the quality of the generated text.
            `scores (list[int])`: A list of integers representing the possible scores that can be assigned during evaluation.
            This list defines the scoring range for assessing the quality of the text (e.g., [1, 2, 3, 4, 5]).

        Returns:
            tuple[BaseResponse, float]: A tuple containing:
                - `BaseResponse`: The response object from the model's inference.
                - `float`: The evaluated score for the quality of the text, computed as a weighted sum.

        Calculation:
            The final score is calculated using the formula:
                `score = sum(p(si) * si for si in S)`
            where `p(si)` is the probability of score `si` being assigned by the LLM.

        For details, see: https://arxiv.org/pdf/2303.16634
        """  # noqa: E501
        wrapped_call = _nlg_evaluation(self.call, protocol, scores)
        return await wrapped_call(request)
