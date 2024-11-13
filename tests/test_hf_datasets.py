# test_hf_datasets.py
import asyncio
from functools import partial
from pathlib import Path
import httpx
import pytest
from datasets import Dataset
import typing as typ

from throughster.base import ModelInterface
from throughster.factory import create_interface
from throughster.hf_datasets import transform, HfOperation
from throughster.core.models import BaseResponse
from tests.utils import HELLO_WORLD_PROMPT
import json

BASE_PATH = Path("tests/vllm/responses")
TARGET_VERSIONS = [dir.name for dir in BASE_PATH.iterdir() if dir.is_dir()]
EXPECTED_TRANSFORMATION_OUTPUT = "Hello World!"


class AddOneOperation(HfOperation):
    def __init__(self):
        pass

    def __call__(self, batch: dict[str, list[typ.Any]], idx: None | list[int] = None) -> dict[str, list[typ.Any]]:
        return {"value": [x + 1 for x in batch["value"]]}


class OperationWithClient(HfOperation):
    _no_fingerprint: None | list[str] = ["init_client_fn"]

    def __init__(self, init_client_fn: typ.Callable[..., ModelInterface], model: str, **kwargs: typ.Any):
        self.init_client_fn = init_client_fn
        self.model = model
        self.messages = [{"role": "user", "content": HELLO_WORLD_PROMPT}]

    def __call__(self, batch: dict[str, list[typ.Any]], idx: None | list[int] = None) -> dict[str, list[typ.Any]]:
        requests = [{"model": self.model, "messages": self.messages} for _ in batch["value"]]
        results = asyncio.run(self.fn(requests))

        return {"value": batch["value"], "response": [result.content for result in results]}

    async def fn(self, requests: list[dict[str, typ.Any]]) -> list[BaseResponse]:
        coroutines = [self.client.call(request=request) for request in requests]
        results = await asyncio.gather(*coroutines)
        return results


@pytest.fixture
def endpoint() -> str:
    return "https://example.com"


@pytest.fixture
def mock_json_response() -> dict[str, typ.Any]:
    resp = json.load(open(BASE_PATH / max(TARGET_VERSIONS) / "chat_completion.json"))
    resp["choices"][0]["message"]["content"] = EXPECTED_TRANSFORMATION_OUTPUT
    return resp


@pytest.fixture
def dataset() -> Dataset:
    return Dataset.from_dict({"value": [1, 2, 3, 4, 5]})


@pytest.mark.parametrize(
    "operation,expected",
    [
        (AddOneOperation(), Dataset.from_dict({"value": [2, 3, 4, 5, 6]})),
    ],
)
def test_transform_map(dataset, operation, expected):
    initial_fingerprint = dataset._fingerprint
    result = transform(dataset, operation, operation="map", batched=True)
    new_fingerprint = result._fingerprint
    assert result["value"] == expected["value"]
    assert initial_fingerprint != new_fingerprint

    cached_result = transform(dataset, operation, operation="map", batched=True)
    same_fingerprint = cached_result._fingerprint
    assert new_fingerprint == same_fingerprint


@pytest.mark.parametrize(
    "expected_transformation",
    [
        Dataset.from_dict({"value": [2, 3, 4, 5, 6], "response": [EXPECTED_TRANSFORMATION_OUTPUT] * 5}),
    ],
)
def test_transform_with_client(dataset, endpoint, mock_json_response, expected_transformation):
    test_client = httpx.AsyncClient(
        base_url=endpoint, transport=httpx.MockTransport(lambda request: httpx.Response(200, json=mock_json_response))
    )
    init_vthroughster = partial(
        create_interface, provider="vllm", api_base=endpoint, client=test_client, use_cache=False
    )

    initial_fingerprint = dataset._fingerprint
    operation = OperationWithClient(init_client_fn=init_vthroughster, model="dummy_model")
    actual = transform(dataset, operation, operation="map", batched=True)
    new_fingerprint = actual._fingerprint

    assert actual["response"] == expected_transformation["response"]
    assert initial_fingerprint != new_fingerprint

    cached_result = transform(dataset, operation, operation="map", batched=True)
    same_fingerprint = cached_result._fingerprint
    assert new_fingerprint == same_fingerprint

    new_operation = OperationWithClient(init_client_fn=init_vthroughster, model="dummy_model_2")
    new_result = transform(dataset, new_operation, operation="map", batched=True)
    newer_fingerprint = new_result._fingerprint
    assert new_fingerprint != newer_fingerprint
