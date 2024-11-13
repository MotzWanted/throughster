import typing as typ
from collections.abc import Callable

import pydantic

from throughster.core.models import ModelCard
from throughster.vllm.utils import generate_schema_from_pydantic


class AvailableModels(pydantic.BaseModel):
    """Available models."""

    object: typ.Literal["list"]
    data: list[ModelCard]


class UsageInfo(pydantic.BaseModel):
    """Usage information."""

    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: int | None = 0


class VllmRequest(pydantic.BaseModel):
    """Sampling parameters for vLLM.

    See https://github.com/vllm-project/vllm/blob/30e754390c2a8a7198f472386d35ee1ec9443e4a/vllm/entrypoints/openai/protocol.py#L104.
    """

    model: str
    messages: list[dict[str, str]] | None = None
    prompt: str | None = None
    n: int | None = 1
    best_of: int | None = None
    presence_penalty: float | None = 0.0
    frequency_penalty: float | None = 0.0
    repetition_penalty: float | None = 1.0
    temperature: float | None = 0.7
    top_p: float | None = 1.0
    top_k: int | None = -1
    min_p: float | None = 0.0
    use_beam_search: bool | None = False
    length_penalty: float | None = 1.0
    early_stopping: bool | None = False
    stop: str | list[str] | None = None
    stop_token_ids: list[int] | None = None
    include_stop_str_in_output: bool | None = False
    ignore_eos: bool | None = False
    max_tokens: int | None = 512
    logprobs: bool | None = False
    top_logprobs: int | None = None
    skip_special_tokens: bool | None = True
    spaces_between_special_tokens: bool | None = True
    logits_processor: list[Callable] | None = None
    guided_json: dict[str, typ.Any] | None = pydantic.Field(None, alias="schema")
    guided_choice: list[str] | None = None
    guided_regex: str | None = None
    guided_grammar: str | None = None
    guided_decoding_backend: str | None = None
    guided_whitespace_pattern: str | None = None

    @pydantic.field_validator("guided_json", mode="before")
    @classmethod
    def validate_schema(
        cls: type["VllmRequest"],
        v: list | dict[str, typ.Any] | type[pydantic.BaseModel] | None,
    ) -> dict[str, typ.Any] | None:
        """Validate the schema."""
        if v is None or isinstance(v, dict):
            return v
        if isinstance(v, list):
            if len(v) != 1:
                msg = "Only one tool is supported."
                raise ValueError(msg)
            return generate_schema_from_pydantic(v[0])
        return generate_schema_from_pydantic(v)

    @pydantic.model_validator(mode="after")
    def validate_prompt_or_messages(self) -> typ.Self:
        """Validate the prompt or messages."""
        if self.prompt is None and self.messages is None:
            raise ValueError("Either `prompt` or `messages` must be provided.")
        return self


class Function(pydantic.BaseModel):
    """A function."""

    arguments: str
    name: str


class ChatMessage(pydantic.BaseModel):
    """A chat message."""

    role: str
    content: str | None = None


class LogProbs(pydantic.BaseModel):
    """Log probabilities."""

    text_offset: list[int] = pydantic.Field(default_factory=list)
    token_logprobs: list[float] | None = pydantic.Field(default_factory=list)
    tokens: list[str] | None = pydantic.Field(default_factory=list)
    top_logprobs: list[dict[str, float]] | None = None


class ChatCompletionResponseChoice(pydantic.BaseModel):
    """A chat completion response choice."""

    index: int
    message: str = pydantic.Field(..., validation_alias=pydantic.AliasChoices("message", "text"))
    logprobs: LogProbs | None = None
    finish_reason: typ.Literal["stop", "length", "tool_calls"]

    @pydantic.field_validator("message", mode="before")
    @classmethod
    def validate_message(cls, v: dict[str, str] | str) -> str:
        """Validate the message."""
        if isinstance(v, str):
            return v
        if isinstance(v, dict) and "content" in v:
            return v["content"]
        raise ValueError("Unexpected message format.")


class ChatCompletionResponse(pydantic.BaseModel):
    """Text completion model."""

    id: str
    object: typ.Literal["chat.completion", "text_completion"]
    created: int
    model: str
    choices: list[ChatCompletionResponseChoice]
    usage: UsageInfo | None = None


class DeltaMessage(pydantic.BaseModel):
    """A message delta."""

    role: str | None = None
    content: str | None = None


class ChatCompletionResponseStreamChoice(pydantic.BaseModel):
    """A chat completion response stream choice."""

    index: int
    delta: DeltaMessage
    finish_reason: typ.Literal["stop", "length"]


class ChatCompletionStreamResponse(pydantic.BaseModel):
    """A stream of chat completion responses."""

    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChatCompletionResponseStreamChoice]
    usage: UsageInfo | None = None
