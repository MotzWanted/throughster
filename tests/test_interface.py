import os
import pytest
import typing as typ

from throughster import ModelInterface, OpenAiInterface, MistralInterface, VllmOpenAiInterface
from throughster.factory import create_interface

MSG_UNKNOWN_PROVIDER = "Unknown provider: {provider}"
MSG_API_ENDPOINT_PLACEHOLDER = "API_ENDPOINT contains a placeholder, but no deployments were provided."
MSG_API_DEPLOYMENTS = "API_ENDPOINT must contain a placeholder `{deployment}` if deployments are provided."

AZURE_INIT_KWARGS = {"api_base": "http://example.com", "api_key": "dummykey", "api_version": "dummyversion"}
VLLM_INIT_KWARGS = {"api_base": "localhost:8080"}
MISTRAL_INIT_KWARGS = {"api_base": "http://example.com", "model_name": "model-1"}


@pytest.fixture(scope="session", autouse=True)
def set_env_variables():
    os.environ["AIOCACHE_DISABLE"] = "1"
    os.environ["AZURE_OPENAI_API_KEY"] = "dummykey"
    os.environ["AZURE_OPENAI_API_VERSION"] = "dummyversion"
    os.environ["AZURE_OPENAI_API_BASE"] = "https://someresource.openai.azure.com/openai/deployments/{deployment}"
    os.environ["MISTRAL_API_KEY"] = "dummykey"
    os.environ["MISTRAL_API_BASE"] = "https://some-serverless-mistral.location.inference.ai.azure.com/v1/"
    os.environ["VLLM_API_BASE"] = "http://localhost:8080"


@pytest.mark.parametrize(
    ("test_provider", "test_init_kwargs", "expected_type", "expected_exception", "expected_message"),
    [
        ("azure", {}, OpenAiInterface, None, None),
        ("azure", AZURE_INIT_KWARGS, OpenAiInterface, None, None),
        ("vllm", {}, VllmOpenAiInterface, None, None),
        ("vllm", VLLM_INIT_KWARGS, VllmOpenAiInterface, None, None),
        ("mistral", {}, MistralInterface, None, None),
        ("mistral", MISTRAL_INIT_KWARGS, MistralInterface, None, None),
        ("test", {}, None, ValueError, MSG_UNKNOWN_PROVIDER.format(provider="test")),
    ],
)
def test_client(
    test_provider: str,
    test_init_kwargs: dict[str, typ.Any],
    expected_type: type[ModelInterface],
    expected_exception: type[Exception],
    expected_message: str,
) -> None:
    if expected_type:
        client = create_interface(provider=test_provider, **test_init_kwargs)
        assert isinstance(client, expected_type), "Client is not an instance of `{expected_type}`"

    else:
        with pytest.raises(expected_exception) as exc_info:
            create_interface(provider=test_provider, **test_init_kwargs)
        assert str(exc_info.value) == expected_message, f"Expected message: {expected_message}"
