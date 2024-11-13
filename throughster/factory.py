import httpx

from throughster.core import config
from throughster import ModelInterface, VllmOpenAiInterface, MistralInterface, OpenAiInterface
from throughster.core.constants import DEFAULT_LIMITS, DEFAULT_TIMEOUT
from aiofilecache import FileCache
from aiocache.serializers import PickleSerializer


def create_interface(
    provider: str,
    api_base: str | None = None,
    endpoint: str = "chat/completions",
    limits: httpx.Limits = DEFAULT_LIMITS,
    timeout: httpx.Timeout = DEFAULT_TIMEOUT,
    api_key: str | None = None,
    api_version: str | None = None,
    model_name: str | None = None,
    client: httpx.AsyncClient | None = None,
    cache_dir: str | None = None,
    use_cache: bool | None = None,
) -> ModelInterface:
    """Create the model interface based on the provider."""

    provider_map = {
        "azure": ("AZURE_OPENAI_", config.AzureClientSettings, OpenAiInterface),
        "vllm": ("VLLM_", config.VllmClientSettings, VllmOpenAiInterface),
        "mistral": ("MISTRAL_", config.MistralClientSettings, MistralInterface),
    }

    if provider not in provider_map:
        raise ValueError(f"Unknown provider: {provider}")

    env_prefix, settings_class, interface_class = provider_map[provider]
    settings = settings_class(
        API_BASE=api_base,
        API_KEY=api_key,
        API_VERSION=api_version,
        MODEL_NAME=model_name,
        _env_prefix=env_prefix,
        CACHE_DIR=cache_dir,
        USE_CACHE=use_cache,
    )  # type: ignore

    cache = None
    if settings.USE_CACHE:
        cache = FileCache(serializer=PickleSerializer(), basedir=settings.CACHE_DIR)

    return interface_class(
        api_base=settings.API_BASE,
        endpoint=endpoint,
        api_key=settings.API_KEY,
        api_version=settings.API_VERSION if hasattr(settings, "API_VERSION") else api_version,
        model_name=settings.MODEL_NAME,
        limits=limits,
        timeout=timeout,
        client=client,
        cache=cache,
    )
