import typing as typ
from abc import ABC, abstractmethod
from collections.abc import AsyncGenerator

import httpx
import pydantic
import tenacity

from throughster.core.errors import RateLimitError
from throughster.core import decorators
from throughster.core.models import BaseResponse, ModelCard
from aiocache import BaseCache

RetryingFn = typ.Callable[[typ.Callable], typ.Callable]
RetryingConstructor = typ.Callable[[], RetryingFn]


def get_default_retry() -> RetryingFn:
    return tenacity.retry(
        stop=tenacity.stop_after_attempt(10),
        wait=tenacity.wait_exponential(multiplier=1, min=3, max=30),
        retry=tenacity.retry_if_exception_type(RateLimitError)
        | tenacity.retry_if_exception_type(httpx.TimeoutException)
        | tenacity.retry_if_exception_type(httpx.ProtocolError)
        | tenacity.retry_if_exception_type(httpx.NetworkError),
        reraise=True,
    )


async def _call_client(
    client: httpx.AsyncClient, endpoint: str, data: dict, set_cache_value: httpx.Response | None = None
) -> httpx.Response:
    if set_cache_value is not None:
        return set_cache_value

    return await client.post(endpoint, json=data)


def create_httpx_client(
    base_url: str,
    limits: httpx.Limits,
    timeout: httpx.Timeout,
    params: dict[str, str] | None = None,
    headers: dict[str, str] | None = None,
) -> httpx.AsyncClient:
    """Create an httpx client."""
    default_headers = {"Content-Type": "application/json"}
    if isinstance(headers, dict):
        default_headers.update(headers)
    return httpx.AsyncClient(
        base_url=base_url,
        params=params,
        headers=default_headers,
        limits=limits,
        timeout=timeout,
    )


class ModelInterface(ABC):
    """Abstract class for querying a causal language model with HTTP requests."""

    def __init__(
        self,
        api_base: str,
        endpoint: str,
        limits: httpx.Limits,
        timeout: httpx.Timeout,
        api_key: str | None = None,
        api_version: str | None = None,
        model_name: str | None = None,
        cache: BaseCache | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._api_base = api_base
        self.endpoint = endpoint
        self.limits = limits
        self.timeout = timeout
        self._client = client
        self.model_name = model_name
        self.api_key = api_key
        self.api_version = api_version
        self._available_models: None | list[ModelCard] = None
        self.cache = cache
        self._no_fingerprint = ["_client", "api_key", "api_version", "timeout", "limits", "cache"]

        self._call_client_function = _call_client
        if self.cache is not None:
            # Do not cache the cached function because it does not have a reliable fingerprint
            self._no_fingerprint.append("_call_client_function")
            self._call_client_function = decorators._handle_cached_function(
                _call_client,  # Pass the function to be wrapped
                cache=self.cache,  # Inject the cache instance
                ignore_args=["client", "set_cache_value"],  # Ignore these args when creating the cache key
                cache_condition=lambda x: x.status_code == 200,  # Condition for caching
            )

    async def __aenter__(self):
        """Async context manager entry point to safely share ModelInterface across coroutines.
        I.e., the httpx client is created once and shared across coroutines; client won't close during coroutines.
        See: https://github.com/encode/httpx/discussions/2275
        """
        if self._client is None:
            self._client = create_httpx_client(
                base_url=self.api_base,
                headers=self.headers,
                params=self.params,
                limits=self.limits,
                timeout=self.timeout,
            )
        return self

    async def __aexit__(self, exc_type, exc_value, traceback):
        """Async context manager exit point."""
        if self._client is not None:
            await self._client.aclose()

    @property
    def client(self) -> httpx.AsyncClient:
        """Return the client."""
        if self._client is None:
            self._client = create_httpx_client(
                base_url=self.api_base,
                headers=self.headers,
                params=self.params,
                limits=self.limits,
                timeout=self.timeout,
            )
        return self._client

    @property
    def available_models(self) -> list[ModelCard]:
        """Return the available models."""
        if self._available_models is None:
            self._available_models = self.list_models()
        return self._available_models

    @property
    def api_base(self) -> str:
        return self._api_base

    def __getstate__(self) -> object:
        """Return the state."""
        state = self.__dict__.copy()
        state["_client"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        """Set the state."""
        self.__dict__.update(state)

    async def close(self) -> None:
        """Close the client."""
        if self._client is not None:
            await self._client.aclose()

    @abstractmethod
    def list_models(self) -> list[ModelCard]:
        """Return the available models."""
        raise NotImplementedError

    @abstractmethod
    def validate_request(self, data: dict[str, typ.Any]) -> dict[str, typ.Any]:
        """Format the call data."""

    @abstractmethod
    def unpack_call(self, response: httpx.Response) -> BaseResponse:
        """Unpack the response."""

    @abstractmethod
    def unpack_stream(self, chunk: httpx.Response) -> AsyncGenerator[str, None]:
        """Unpack the response."""

    @property
    @abstractmethod
    def headers(self) -> dict[str, str]:
        """Return the headers."""

    @property
    @abstractmethod
    def params(self) -> dict[str, str]:
        """Return the parameters."""

    async def _call(self, request: dict[str, typ.Any]) -> BaseResponse:
        """NOTE: client is passed as an argument to share the same AsyncClient across  multiple coroutines."""
        request["stream"] = False
        data = self.validate_request(request)
        resp = await self._call_client_function(self.client, self.endpoint, data)
        return self.unpack_call(resp)

    async def call(
        self,
        request: dict[str, typ.Any],
        retry_fn_constructor: RetryingConstructor = get_default_retry,
    ) -> BaseResponse:
        """Makes a single post call to the specified model endpoint with the provided request data.

        Args:
            `request (dict[str, typ.Any])`: A dictionary containing the parameters required for the API call.
            This typically includes the prompt/chat messages and sampling params.
            `retry_fn (RetryDecorator, optional)`: A decorator function to retry the call in case of failure.

        Returns:
            `BaseResponse`: A response object containing the result of the model's inference.
        """
        return await retry_fn_constructor()(self._call)(request)

    async def _stream(self, request: dict) -> AsyncGenerator[str, None]:
        request["stream"] = True
        data = self.validate_request(request)
        async with self.client.stream("POST", self.endpoint, json=data) as resp:
            async for chunk in self.unpack_stream(resp):
                yield chunk

    async def stream(
        self, request: dict, retry_fn_constructor: RetryingConstructor = get_default_retry
    ) -> AsyncGenerator[str, None]:
        """Streams the response from the model endpoint using the provided request data.

        Args:
            `request (dict[str, typ.Any])`: A dictionary containing the parameters required for the API call.
            This typically includes the prompt/chat messages and sampling params.
            `retry_fn (StreamRetryDecorator, optional)`: A decorator function to retry the stream call in case of failure.

        Yields:
            `str`: Chunks of the model's response as they are received.
        """  # noqa: E501
        return await retry_fn_constructor()(self._stream)(request)

    async def structured_call(
        self,
        request: dict[str, typ.Any],
        schema: type[pydantic.BaseModel],
        max_attempts: int = 2,
        retry_fn_constructor: RetryingConstructor = get_default_retry,
    ) -> BaseResponse:
        """Makes an API call to the specified model endpoint and validates the response against a Pydantic schema.

        Args:
            `request (dict[str, typ.Any])`: A dictionary containing the parameters required for the API call.
            This typically includes the prompt/chat messages and sampling params.
            `schema (type[pydantic.BaseModel])`: A Pydantic model class that defines the expected schema of the response.
            `max_attempts (int, optional)`: The maximum number of attempts to retry the call in case of failure. Defaults to 2.

        Returns:
            `BaseResponse`: A response object containing the result of the model's inference, validated against the schema.

        Raises:
            `StructuredResponseError`: If the response does not conform to the provided Pydantic schema.
        """  # noqa: E501
        wrapped_call = decorators._structured_pydantic_call(self.call, schema, max_attempts)
        return await wrapped_call(request, retry_fn_constructor)

    async def batch_call(
        self, requests: list[dict[str, typ.Any]], retry_fn_constructor: RetryingConstructor = get_default_retry
    ) -> list[BaseResponse]:
        """Makes multiple post calls to the specified model endpoint with the provided request data."""
        return await decorators._batch_decorator(self.call)(requests, retry_fn_constructor)

    async def structured_batch_call(
        self,
        requests: list[dict[str, typ.Any]],
        schema: type[pydantic.BaseModel],
        max_attempts: int = 2,
        retry_fn_constructor: RetryingConstructor = get_default_retry,
    ) -> list[BaseResponse]:
        wrapped_call = decorators._structured_pydantic_call(
            endpoint_func=self.call, schema=schema, max_attempts=max_attempts
        )
        return await decorators._batch_decorator(wrapped_call)(requests, retry_fn_constructor)

    # Sync dectorators
    def sync_call(self, request: dict[str, typ.Any]) -> BaseResponse:
        return decorators._sync_call(self.call)(request)

    def sync_structured_call(
        self,
        request: dict[str, typ.Any],
        schema: type[pydantic.BaseModel],
        max_attempts: int = 2,
    ) -> BaseResponse:
        return decorators._sync_call(self.structured_call)(request, schema, max_attempts)

    def sync_batch_call(self, requests: list[dict[str, typ.Any]]) -> list[BaseResponse]:
        return decorators._sync_call(self.batch_call)(requests)

    def sync_structured_batch_call(
        self,
        requests: list[dict[str, typ.Any]],
        schema: type[pydantic.BaseModel],
        max_attempts: int = 2,
    ) -> list[BaseResponse]:
        return decorators._sync_call(self.structured_batch_call)(requests, schema, max_attempts)
