import asyncio
import functools
import hashlib
import inspect
import json
import typing as typ
from collections.abc import Callable

import anyio
import numpy as np
import pydantic
from loguru import logger

from throughster.core.errors import StructuredResponseError
from throughster.core.models import BaseResponse
from aiocache import BaseCache

from throughster.core.retry import _retry_stat_decorator


def _adjust_temperature(request: dict[str, typ.Any], max_attempts: int, attempt: int, adjust_temp_factor: float):
    """Custom activation function for temperature adjustment when validating the response fails."""
    temp = request.get("temperature", 0.1)
    # Exaggerate the adjustment for temperatures below 0.5, diminish adjustment for temperatures above 0.5
    delta_temp = adjust_temp_factor * ((1 - temp) ** 2) * temp if temp > 0 else adjust_temp_factor * 0.1

    new_temp = temp + delta_temp
    # Clamp the new temperature to [0, 1.5]
    new_temp = max(0.0, min(new_temp, 1.5))
    logger.info(
        f"[{attempt}/{max_attempts}] Couldn't validate response.",
        "Retrying with increased temperature: {new_temperature}",
    )
    return request


def _structured_pydantic_call(
    endpoint_func: typ.Callable, schema: type[pydantic.BaseModel], max_attempts: int, adjust_temp_factor: float = 1.5
) -> Callable:
    """Endpoint wrapper to validate the llm response against a Pydantic schema."""

    @functools.wraps(endpoint_func)
    async def wrapper(request: dict[str, typ.Any], retry_fn_constructor: Callable) -> BaseResponse:
        request["schema"] = schema
        for attempt in range(max_attempts):
            resp: BaseResponse = await endpoint_func(request, retry_fn_constructor)

            try:
                resp.validated_schema = schema.model_validate_json(resp.content)
                return resp
            except pydantic.ValidationError:
                request = _adjust_temperature(request, max_attempts, attempt, adjust_temp_factor)
                continue

        raise StructuredResponseError("No response was validated against the provided schema.")

    return wrapper


def _pydantic_tools_call(
    endpoint_func: typ.Callable, tools: list[pydantic.BaseModel], max_attempts: int, adjust_temp_factor: float
) -> Callable:
    """Endpoint wrapper to validate the llm response against a list of Pydantic schemas.

    See for usecase: https://arxiv.org/abs/2210.03629
    NOTE: This is currently only supported for OpenAI's Chat API.
    """

    @functools.wraps(endpoint_func)
    async def wrapper(request: dict[str, typ.Any]) -> pydantic.BaseModel:
        request["tools"] = tools
        for attempt in range(max_attempts):
            resp = await endpoint_func(request)
            valid_response = False

            for tool in tools:
                try:
                    resp.validated_schema = tool.model_validate_json(resp.content)
                    valid_response = True
                    break
                except pydantic.ValidationError:
                    continue

            if valid_response:
                return resp

            request = _adjust_temperature(request, max_attempts, attempt, adjust_temp_factor)

        raise StructuredResponseError("No response was validated against the provided list of tools.")

    return wrapper


def _nlg_evaluation(endpoint_func: typ.Callable, protocol: str | list[dict[str, str]], scores: list[int]) -> Callable:
    """Endpoint wrapper to validate the llm response against a list of Pydantic schemas.

    See for implementation details: https://arxiv.org/pdf/2303.16634.pdf
    NOTE: This is currently only supported for vLLM instances.
    """

    @functools.wraps(endpoint_func)
    async def wrapper(request: dict[str, typ.Any]) -> tuple[BaseResponse, float]:
        request["guided_choice"] = scores
        request["max_tokens"] = 1
        request["logprobs"] = True
        request["top_logprobs"] = len(scores)
        request["messages"] = protocol if isinstance(protocol, list) else [{"role": "user", "content": protocol}]

        resp: BaseResponse = await endpoint_func(request)

        if resp.logprobs is None:
            raise ValueError(f"Invalid logprobs: {resp}")

        score = np.sum([np.exp(v) * int(k) for k, v in resp.logprobs[0].items()])
        return resp, np.round(score, decimals=4)

    return wrapper


# workaround for the issue with the asyncio.run() function
# https://github.com/microsoftgraph/msgraph-sdk-python/issues/366
def get_loop():
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop


def _sync_call(async_method: typ.Callable[..., typ.Coroutine[typ.Any, typ.Any, typ.Any]]):
    @functools.wraps(async_method)
    def wrapper(self, *args, **kwargs) -> typ.Any:
        # Ensure the method is a coroutine
        coro = async_method(self, *args, **kwargs)
        if not isinstance(coro, typ.Coroutine):
            raise TypeError(f"Expected coroutine, got {type(coro).__name__}")
        return get_loop().run_until_complete(coro)

    return wrapper


def _batch_decorator(
    async_method: typ.Callable[..., typ.Coroutine[typ.Any, typ.Any, BaseResponse]],
) -> typ.Callable[..., typ.Coroutine[typ.Any, typ.Any, list[BaseResponse]]]:
    async def wrapper(requests: list[dict[str, typ.Any]], retry_fn_constructor) -> list[BaseResponse]:
        responses = [None] * len(requests)

        async def task_wrapper(index: int, request: dict[str, typ.Any]) -> None:
            retry_fn = _retry_stat_decorator(retry_fn_constructor, async_method)
            response: BaseResponse = await retry_fn(request)
            responses[index] = response

        # avoids potential race conditions where one coroutine could close the client before others finish
        async with anyio.create_task_group() as tg:
            for idx, request in enumerate(requests):
                tg.start_soon(task_wrapper, idx, request)

        return typ.cast(list[BaseResponse], responses)

    return wrapper


def build_cache_key(ignore_args: list[str], func, *args, **kwargs) -> str:
    """
    Generate a cache key based on the function's signature and arguments.
    """
    # Get the function's signature and bound arguments
    signature = inspect.signature(func)
    bound_arguments = signature.bind(*args, **kwargs)
    bound_arguments.apply_defaults()

    # Filter out ignored arguments
    filtered_args = {key: value for key, value in bound_arguments.arguments.items() if key not in ignore_args}

    # Convert bound arguments to a string representation
    key_str = json.dumps(filtered_args, sort_keys=True, default=str)

    # Generate a hash of the key to keep it short and unique
    return hashlib.md5(key_str.encode("utf-8")).hexdigest()


def _handle_cached_function(
    func: typ.Callable, cache: BaseCache, cache_condition: typ.Callable, ignore_args: list[str] = []
) -> typ.Callable:
    """Endpoint wrapper to conditionally cache the LLM response."""

    @functools.wraps(func)
    async def wrapper(*args, **kwargs) -> BaseResponse:
        # Generate a cache key based on the arguments (implement your key generation logic here)
        cache_key = build_cache_key(ignore_args, func, *args, **kwargs)
        # Try to get the cached result
        cached_result = await cache.get(cache_key)

        if cached_result is not None:
            return cached_result

        # Execute the function normally if no cached result
        result = await func(*args, **kwargs)

        # Cache the result if the condition is met, e.g., the response returns status code 200
        if cache_condition(result):
            await cache.set(cache_key, result)

        return result

    return wrapper
