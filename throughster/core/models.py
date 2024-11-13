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


class LogProbs(pydantic.BaseModel):
    text_offset: list[int] = pydantic.Field(default_factory=list)
    token_logprobs: list[float] = pydantic.Field(default_factory=list)
    tokens: list[str] = pydantic.Field(default_factory=list)
    top_logprobs: list[dict[str, float]] = pydantic.Field(default_factory=list)


class ResponseChoice(pydantic.BaseModel):
    """Base response."""

    index: int
    content: str = pydantic.Field(..., validation_alias=pydantic.AliasChoices("message", "text"))
    finish_reason: str
    validated_schema: pydantic.BaseModel | None = None
    logprobs: LogProbs | None = None

    @pydantic.field_validator("content", mode="before")
    @classmethod
    def validate_message(cls: type["ResponseChoice"], v: dict[str, str] | str) -> str:
        """Validate the message."""
        if isinstance(v, dict):
            return v["content"]
        return v

    @pydantic.field_validator("logprobs", mode="before")
    @classmethod
    def validate_logprobs(cls: type["ResponseChoice"], v: dict[str, typ.Any] | list[dict] | None) -> dict | None:
        """Validate the logprobs."""
        if v is None:
            return None
        if isinstance(v, list):
            return {"top_logprobs": v}
        return v


class BaseResponse(pydantic.BaseModel):
    """Text completion model."""

    id: str
    object: typ.Literal["chat.completion", "text_completion", "structured.completion"]
    created: int
    model: str
    choices: list[ResponseChoice]
    usage: UsageInfo | None = None


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
