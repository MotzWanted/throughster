import os
import shutil
from tests.utils import HELLO_WORLD_MESSAGES, HELLO_WORLD_MAX_TOKENS, TestGuidedGeneration
from pathlib import Path
import pytest
import json
import httpx
from throughster.core.models import BaseResponse
from throughster.core.errors import StructuredResponseError, AzureContentFilterError, RateLimitError
from throughster.factory import create_interface
import tenacity

BASE_PATH = Path("tests/azure/responses")
CACHE_DIR = ".llm-client-test-cache"
TARGET_VERSIONS = [dir.name for dir in BASE_PATH.iterdir() if dir.is_dir()]


class RetrySimulator:
    def __init__(self, responses: list[httpx.Response]):
        self.i = 0
        self.responses = responses

    def __call__(self, request):
        response = self.responses[self.i]
        self.i += 1
        return response


@pytest.fixture
def model() -> str:
    return "dummy_model"


@pytest.fixture
def api_key() -> str:
    return "dummy_api_key"


@pytest.fixture
def endpoint() -> str:
    return "https://example.com"


@pytest.mark.parametrize("api_version", TARGET_VERSIONS)
@pytest.mark.asyncio
async def test_chat_completion(api_version, endpoint, model, api_key) -> None:
    with (BASE_PATH / api_version / "chat_completion.json").open("r") as f:
        mock_json_response = json.load(f)

    test_client = httpx.AsyncClient(
        base_url=endpoint, transport=httpx.MockTransport(lambda request: httpx.Response(200, json=mock_json_response))
    )
    openai_client = create_interface(
        "azure",
        api_base=endpoint,
        api_version=api_version,
        client=test_client,
        model_name=model,
        api_key=api_key,
        cache_dir=CACHE_DIR,
    )
    if os.path.exists(CACHE_DIR):
        shutil.rmtree(CACHE_DIR)
    chat_completion = await openai_client.call(
        {
            "messages": HELLO_WORLD_MESSAGES,
            "max_tokens": HELLO_WORLD_MAX_TOKENS,
        }
    )

    assert isinstance(chat_completion, BaseResponse)
    assert isinstance(chat_completion.content, str)
    assert len(chat_completion.content) > 0


@pytest.mark.parametrize("api_version", TARGET_VERSIONS)
@pytest.mark.asyncio
async def test_batch_chat_completion(api_version, endpoint, model, api_key) -> None:
    with (BASE_PATH / api_version / "chat_completion.json").open("r") as f:
        mock_json_response = json.load(f)

    BATCH_SIZE = 5

    test_client = httpx.AsyncClient(
        base_url=endpoint,
        transport=httpx.MockTransport(lambda request: httpx.Response(200, json=mock_json_response)),
    )
    openai_client = create_interface(
        "azure",
        api_base=endpoint,
        api_version=api_version,
        client=test_client,
        model_name=model,
        api_key=api_key,
        cache_dir=CACHE_DIR,
    )
    if os.path.exists(CACHE_DIR):
        shutil.rmtree(CACHE_DIR)

    chat_completions = await openai_client.batch_call(
        requests=[
            {
                "messages": HELLO_WORLD_MESSAGES,
                "max_tokens": HELLO_WORLD_MAX_TOKENS,
            }
        ]
        * BATCH_SIZE
    )

    assert isinstance(chat_completions, list)
    assert isinstance(chat_completions[0], BaseResponse)
    assert isinstance(chat_completions[0].content, str)
    assert len(chat_completions) == BATCH_SIZE


@pytest.mark.parametrize("api_version", TARGET_VERSIONS)
def test_sync_chat_completion(api_version, endpoint, model, api_key) -> None:
    with (BASE_PATH / api_version / "chat_completion.json").open("r") as f:
        mock_json_response = json.load(f)

    test_client = httpx.AsyncClient(
        base_url=endpoint, transport=httpx.MockTransport(lambda request: httpx.Response(200, json=mock_json_response))
    )
    openai_client = create_interface(
        "azure",
        api_base=endpoint,
        api_version=api_version,
        client=test_client,
        model_name=model,
        api_key=api_key,
        cache_dir=CACHE_DIR,
    )
    if os.path.exists(CACHE_DIR):
        shutil.rmtree(CACHE_DIR)

    chat_completion = openai_client.sync_call(
        {
            "messages": HELLO_WORLD_MESSAGES,
            "max_tokens": HELLO_WORLD_MAX_TOKENS,
        }
    )

    assert isinstance(chat_completion, BaseResponse)
    assert isinstance(chat_completion.content, str)
    assert len(chat_completion.content) > 0


@pytest.mark.parametrize("api_version", TARGET_VERSIONS)
@pytest.mark.asyncio
async def test_chat_tools_completion_success(api_version, endpoint, model, api_key) -> None:
    with (BASE_PATH / api_version / "chat_tools_completion.json").open("r") as f:
        mock_json_response = json.load(f)

    test_client = httpx.AsyncClient(
        base_url=endpoint, transport=httpx.MockTransport(lambda request: httpx.Response(200, json=mock_json_response))
    )
    openai_client = create_interface(
        "azure",
        api_base=endpoint,
        api_version=api_version,
        api_key=api_key,
        client=test_client,
        cache_dir=".llm-client-test-cache",
    )
    if os.path.exists(CACHE_DIR):
        shutil.rmtree(CACHE_DIR)
    request = {
        "messages": HELLO_WORLD_MESSAGES,
        "max_tokens": HELLO_WORLD_MAX_TOKENS,
    }
    chat_tools_completion = await openai_client.structured_call(request, schema=TestGuidedGeneration)

    assert isinstance(chat_tools_completion.validated_schema, TestGuidedGeneration)


@pytest.mark.parametrize("api_version", TARGET_VERSIONS)
@pytest.mark.asyncio
async def test_chat_tools_completion_fail(api_version, endpoint, model, api_key) -> None:
    with (BASE_PATH / api_version / "chat_tools_completion_fail.json").open("r") as f:
        mock_json_response = json.load(f)

    test_client = httpx.AsyncClient(
        base_url=endpoint, transport=httpx.MockTransport(lambda request: httpx.Response(200, json=mock_json_response))
    )
    openai_client = create_interface(
        "azure",
        api_base=endpoint,
        api_version=api_version,
        api_key=api_key,
        client=test_client,
        cache_dir=".llm-client-test-cache",
    )
    if os.path.exists(CACHE_DIR):
        shutil.rmtree(CACHE_DIR)
    request = {
        "messages": HELLO_WORLD_MESSAGES,
        "max_tokens": HELLO_WORLD_MAX_TOKENS,
    }
    with pytest.raises(StructuredResponseError):
        await openai_client.structured_call(request, schema=TestGuidedGeneration, max_attempts=0)


@pytest.mark.parametrize("api_version", TARGET_VERSIONS)
@pytest.mark.asyncio
async def test_chat_completion_content_filter(api_version, endpoint, model, api_key) -> None:
    with (BASE_PATH / api_version / "chat_completion_content_filter.json").open("r") as f:
        mock_json_response = json.load(f)

    test_client = httpx.AsyncClient(
        base_url=endpoint, transport=httpx.MockTransport(lambda request: httpx.Response(200, json=mock_json_response))
    )
    openai_client = create_interface(
        "azure",
        api_base=endpoint,
        api_version=api_version,
        client=test_client,
        model_name=model,
        api_key=api_key,
        cache_dir=".llm-client-test-cache",
    )
    if os.path.exists(CACHE_DIR):
        shutil.rmtree(CACHE_DIR)

    def retry_constructor():
        return tenacity.retry(stop=tenacity.stop_after_attempt(1), wait=tenacity.wait_fixed(0.1), reraise=True)

    with pytest.raises(AzureContentFilterError):
        await openai_client.call(
            {
                "messages": HELLO_WORLD_MESSAGES,
                "max_tokens": HELLO_WORLD_MAX_TOKENS,
            },
            retry_fn_constructor=retry_constructor,
        )


@pytest.mark.parametrize("api_version", TARGET_VERSIONS)
@pytest.mark.asyncio
async def test_ratelimit_error_failed(api_version, endpoint, model, api_key) -> None:
    with (BASE_PATH / api_version / "chat_tools_completion.json").open("r") as f:
        mock_json_response = json.load(f)

    responses = [httpx.Response(429, json={"error": "rate limit exceeded"})] * 2 + [
        httpx.Response(200, json=mock_json_response)
    ]
    test_client = httpx.AsyncClient(base_url=endpoint, transport=httpx.MockTransport(RetrySimulator(responses)))
    openai_client = create_interface(
        "azure",
        api_base=endpoint,
        api_version=api_version,
        client=test_client,
        model_name=model,
        api_key=api_key,
        cache_dir=".llm-client-test-cache",
    )
    if os.path.exists(CACHE_DIR):
        shutil.rmtree(CACHE_DIR)

    def retry_constructor():
        return tenacity.retry(stop=tenacity.stop_after_attempt(2), wait=tenacity.wait_fixed(0.1), reraise=True)

    with pytest.raises(RateLimitError):
        await openai_client.call(
            {
                "messages": HELLO_WORLD_MESSAGES,
                "max_tokens": HELLO_WORLD_MAX_TOKENS,
            },
            retry_fn_constructor=retry_constructor,
        )


@pytest.mark.parametrize("api_version", TARGET_VERSIONS)
@pytest.mark.asyncio
async def test_ratelimit_error_success(api_version, endpoint, model, api_key) -> None:
    with (BASE_PATH / api_version / "chat_tools_completion.json").open("r") as f:
        mock_json_response = json.load(f)

    responses = [httpx.Response(429, json={"error": "rate limit exceeded"})] * 2 + [
        httpx.Response(200, json=mock_json_response)
    ]
    test_client = httpx.AsyncClient(base_url=endpoint, transport=httpx.MockTransport(RetrySimulator(responses)))
    openai_client = create_interface(
        "azure",
        api_base=endpoint,
        api_version=api_version,
        client=test_client,
        model_name=model,
        api_key=api_key,
    )

    def retry_constructor():
        return tenacity.retry(stop=tenacity.stop_after_attempt(3), wait=tenacity.wait_fixed(0.1), reraise=True)

    await openai_client.call(
        {
            "messages": HELLO_WORLD_MESSAGES,
            "max_tokens": HELLO_WORLD_MAX_TOKENS,
        },
        retry_fn_constructor=retry_constructor,
    )
