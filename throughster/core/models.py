import time
import typing as typ
import uuid

import httpx
import pydantic


class UsageInfo(pydantic.BaseModel):
    """Usage information."""

    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: int | None = 0


class TenacityRetryStats(pydantic.BaseModel):
    start_time: float
    attempt_number: int
    idle_for: float
    delay_since_first_attempt: float = None


class RetryStats(pydantic.BaseModel):
    """Retry stats."""

    tenacity: TenacityRetryStats | None = None
    error_count: dict[str, int] | None = None


class BaseResponse(pydantic.BaseModel):
    """Base response."""

    source: str
    content: str
    finish_reason: str
    validated_schema: pydantic.BaseModel | None = None
    logprobs: list[dict[str, float]] | None = None
    usage: UsageInfo | None = None
    retry_stats: RetryStats | None = None

    @pydantic.field_validator("usage", mode="before")
    @classmethod
    def validate_usage(cls: type["BaseResponse"], v: UsageInfo | dict | None) -> dict[str, typ.Any] | None:
        """Validate the usage."""
        if isinstance(v, pydantic.BaseModel):
            return v.model_dump()
        return v


class ClientSettings(pydantic.BaseModel):
    """Client settings."""

    model_config = pydantic.ConfigDict(arbitrary_types_allowed=True)

    api_base: list[str]
    headers: dict[str, str] | None = None
    params: dict[str, str] | None = None
    limits: httpx.Limits
    timeout: httpx.Timeout


class ModelPermission(pydantic.BaseModel):
    """Model permission."""

    id: str = pydantic.Field(default_factory=lambda: f"modelperm-{str(uuid.uuid4().hex)}")
    object: typ.Literal["model_permission"]
    created: int = pydantic.Field(default_factory=lambda: int(time.time()))
    allow_create_engine: bool = False
    allow_sampling: bool = True
    allow_logprobs: bool = True
    allow_search_indices: bool = False
    allow_view: bool = True
    allow_fine_tuning: bool = False
    organization: str = "*"
    group: str | None = None
    is_blocking: bool = False


class ModelCard(pydantic.BaseModel):
    """Model card."""

    model_config = pydantic.ConfigDict(extra="forbid")
    id: str
    object: typ.Literal["model"]
    created: int = pydantic.Field(default_factory=lambda: int(time.time()))
    owned_by: str = "vllm"
    root: str | None = None
    parent: str | None = None
    permission: list[ModelPermission] = pydantic.Field(default_factory=list)
