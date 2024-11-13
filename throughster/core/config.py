import pydantic
import pydantic_settings
from pathlib import Path


class ClientSettings(pydantic_settings.BaseSettings):
    """Base settings for client configurations."""

    model_config = pydantic_settings.SettingsConfigDict(extra="ignore", protected_namespaces=("settings_",))

    API_BASE: str = pydantic.Field(..., description="API host, e.g., `http://localhost:8080`")
    API_KEY: str = pydantic.Field(..., description="API key")
    MODEL_NAME: str | None = pydantic.Field(None, description="The name or identifier of the deployed model.")
    CACHE_DIR: Path = pydantic.Field(Path("~/.cache/llm-client").expanduser(), description="Cache directory")
    USE_CACHE: bool = pydantic.Field(False, description="Whether to use cache")

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[pydantic_settings.BaseSettings],
        init_settings: pydantic_settings.InitSettingsSource,
        env_settings: pydantic_settings.EnvSettingsSource,
        dotenv_settings: pydantic_settings.DotEnvSettingsSource,
        file_secret_settings: pydantic_settings.SecretsSettingsSource,
    ) -> tuple[pydantic_settings.PydanticBaseSettingsSource, ...]:
        """See: https://docs.pydantic.dev/latest/concepts/pydantic_settings/#customise-settings-sources"""
        _init_kwargs = {k: v for k, v in init_settings.init_kwargs.items() if v is not None}
        if _init_kwargs:
            init_settings.init_kwargs = _init_kwargs
            return init_settings, env_settings, dotenv_settings, file_secret_settings
        return env_settings, dotenv_settings, file_secret_settings


class AzureClientSettings(ClientSettings):
    """Settings for Azure OpenAI API."""

    API_VERSION: str = pydantic.Field(..., description="API version")


class MistralClientSettings(ClientSettings):
    """Settings for Mistral API."""


class VllmClientSettings(ClientSettings):
    """Settings for VLLM API."""

    API_KEY: str = pydantic.Field(default="", description="API key")
