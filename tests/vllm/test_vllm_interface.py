import json
from pathlib import Path


import httpx
import pytest

from throughster.core.models import BaseResponse
from tests.utils import HELLO_WORLD_MESSAGES, HELLO_WORLD_MAX_TOKENS, TestGuidedGeneration
from throughster.core.errors import StructuredResponseError
from throughster.factory import create_interface

BASE_PATH = Path("tests/vllm/responses")
TARGET_VERSIONS = [dir.name for dir in BASE_PATH.iterdir() if dir.is_dir()]


@pytest.fixture
def model() -> str:
    return "dummy_model"


@pytest.fixture
def endpoint() -> str:
    return "https://example.com"


@pytest.mark.parametrize("target_version", TARGET_VERSIONS)
@pytest.mark.asyncio
async def test_chat_completion(target_version, endpoint, model) -> None:
    with (BASE_PATH / target_version / "chat_completion.json").open("r") as f:
        mock_json_response = json.load(f)

    test_client = httpx.AsyncClient(
        base_url=endpoint, transport=httpx.MockTransport(lambda request: httpx.Response(200, json=mock_json_response))
    )
    vllm = create_interface(
        "vllm",
        api_base=endpoint,
        client=test_client,
        model_name=model,
        cache_dir=".llm-client-test-cache",
        use_cache=True,
    )
    await vllm.cache.clear()

    chat_completion = await vllm.call(
        {
            "messages": HELLO_WORLD_MESSAGES,
            "max_tokens": HELLO_WORLD_MAX_TOKENS,
        }
    )

    assert isinstance(chat_completion, BaseResponse)
    assert isinstance(chat_completion.content, str)
    assert len(chat_completion.content) > 0


@pytest.mark.parametrize("target_version", TARGET_VERSIONS)
@pytest.mark.asyncio
async def test_chat_completion_max_tokens(target_version, endpoint, model) -> None:
    with (BASE_PATH / target_version / "chat_completion_max_tokens.json").open("r") as f:
        mock_json_response = json.load(f)

    test_client = httpx.AsyncClient(
        base_url=endpoint, transport=httpx.MockTransport(lambda request: httpx.Response(200, json=mock_json_response))
    )
    vllm = create_interface(
        "vllm",
        api_base=endpoint,
        client=test_client,
        model_name=model,
        cache_dir=".llm-client-test-cache",
        use_cache=True,
    )
    await vllm.cache.clear()

    chat_completion = await vllm.call(
        {
            "messages": HELLO_WORLD_MESSAGES,
            "max_tokens": 1,
        }
    )

    assert isinstance(chat_completion, BaseResponse)
    assert isinstance(chat_completion.content, str)
    assert len(chat_completion.content) > 0


@pytest.mark.parametrize("target_version", TARGET_VERSIONS)
@pytest.mark.asyncio
async def test_chat_completion_stop_token(target_version, endpoint, model) -> None:
    with (BASE_PATH / target_version / "chat_completion_stop_reason.json").open("r") as f:
        mock_json_response = json.load(f)

    test_client = httpx.AsyncClient(
        base_url=endpoint, transport=httpx.MockTransport(lambda request: httpx.Response(200, json=mock_json_response))
    )
    vllm = create_interface(
        "vllm",
        api_base=endpoint,
        client=test_client,
        model_name=model,
        cache_dir=".llm-client-test-cache",
        use_cache=True,
    )
    await vllm.cache.clear()

    chat_completion = await vllm.call(
        {
            "messages": HELLO_WORLD_MESSAGES,
            "max_tokens": HELLO_WORLD_MAX_TOKENS,
            "stop": ["hello"],
        }
    )

    assert isinstance(chat_completion, BaseResponse)
    assert isinstance(chat_completion.content, str)
    assert len(chat_completion.content) > 0


@pytest.mark.parametrize("target_version", TARGET_VERSIONS)
@pytest.mark.asyncio
async def test_chat_completion_guided(target_version, endpoint, model) -> None:
    with (BASE_PATH / target_version / "chat_guided_generation_success.json").open("r") as f:
        mock_json_response = json.load(f)

    test_client = httpx.AsyncClient(
        base_url=endpoint, transport=httpx.MockTransport(lambda request: httpx.Response(200, json=mock_json_response))
    )
    vllm = create_interface(
        "vllm",
        api_base=endpoint,
        model_name=model,
        client=test_client,
        cache_dir=".llm-client-test-cache",
        use_cache=True,
    )
    await vllm.cache.clear()
    request = {
        "messages": HELLO_WORLD_MESSAGES,
        "max_tokens": HELLO_WORLD_MAX_TOKENS,
    }
    chat_completion = await vllm.structured_call(request, schema=TestGuidedGeneration)

    assert isinstance(chat_completion.validated_schema, TestGuidedGeneration)


@pytest.mark.parametrize("target_version", TARGET_VERSIONS)
@pytest.mark.asyncio
async def test_chat_completion_guided_failed(target_version, endpoint, model) -> None:
    with (BASE_PATH / target_version / "chat_guided_generation_failed.json").open("r") as f:
        mock_json_response = json.load(f)

    test_client = httpx.AsyncClient(
        base_url=endpoint, transport=httpx.MockTransport(lambda request: httpx.Response(200, json=mock_json_response))
    )
    vllm = create_interface(
        "vllm",
        api_base=endpoint,
        model_name=model,
        client=test_client,
        cache_dir=".llm-client-test-cache",
        use_cache=True,
    )
    await vllm.cache.clear()
    request = {
        "messages": HELLO_WORLD_MESSAGES,
        "max_tokens": HELLO_WORLD_MAX_TOKENS,
    }
    with pytest.raises(StructuredResponseError):
        await vllm.structured_call(request, schema=TestGuidedGeneration, max_attempts=0)
