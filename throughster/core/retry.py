import typing as typ

import httpx
import tenacity

from throughster.core.errors import RateLimitError
from throughster.core.models import BaseResponse, TenacityRetryStats, RetryStats
from collections import Counter
import functools

RetryingFn = tenacity.AsyncRetrying
RetryingConstructor = typ.Callable[[], RetryingFn]


class RetryCollector:
    def __init__(self) -> None:
        self._retries_count = Counter()

    def count_error(self, retry_state: tenacity.RetryCallState) -> None:
        """Custom before sleep function to collect retry stats."""
        status_code = retry_state.outcome._exception.response.status_code
        exception_name = retry_state.outcome._exception.__class__.__name__
        self._retries_count[f"{status_code}_{exception_name}"] += 1


def _retry_stat_decorator(retry_fn_constructur: RetryingConstructor, endpoint_func) -> BaseResponse:
    """Decorator to add retry stats to the response."""
    retry_collector = RetryCollector()
    retry_fn = retry_fn_constructur()(endpoint_func)
    if retry_fn.retry.before_sleep is None:
        retry_fn.retry.before_sleep = retry_collector.count_error

    @functools.wraps(endpoint_func)
    async def wrapped(request: dict[str, typ.Any]) -> BaseResponse:
        response: BaseResponse = await retry_fn(request)
        response.retry_stats = RetryStats(
            tenacity=TenacityRetryStats(**retry_fn.statistics),
            error_count=retry_collector._retries_count,
        )
        return response

    return wrapped


def get_default_retry() -> RetryingFn:
    retry_collector = RetryCollector()
    retry_fn = tenacity.retry(
        stop=tenacity.stop_after_attempt(10),
        wait=tenacity.wait_exponential(multiplier=1, min=3, max=30),
        retry=tenacity.retry_if_exception_type(RateLimitError)
        | tenacity.retry_if_exception_type(httpx.TimeoutException)
        | tenacity.retry_if_exception_type(httpx.ProtocolError)
        | tenacity.retry_if_exception_type(httpx.NetworkError),
        reraise=True,
        before_sleep=retry_collector.count_error,
    )
    retry_fn.retry_collector = retry_collector
    return retry_fn
