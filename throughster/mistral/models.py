import pydantic
import typing

from throughster.mistral.utils import get_openai_schema

Roles = typing.Literal["system", "user", "assistant", "tool"]


class ResponseFormat(pydantic.BaseModel):
    """Format of the model output."""

    type: str | pydantic.Json = "json_object"


class Tool(pydantic.BaseModel):
    """Tool called by model."""

    type: typing.Literal["function"] = "function"
    function: dict[str, typing.Any]


class Function(pydantic.BaseModel):
    """Function."""

    name: str
    arguments: str


class ToolCall(pydantic.BaseModel):
    """Tool call."""

    id: str | None = None
    type: typing.Literal["function"] = "function"
    function: Function
    call_id: str | None = None


class MistralMessage(pydantic.BaseModel):
    """Chat message."""

    role: Roles
    content: str
    tool_calls: list[ToolCall] | None = None


class ChatCompletionChoice(pydantic.BaseModel):
    """Chat completion response choice."""

    index: int
    message: MistralMessage
    finish_reason: typing.Literal["stop", "length", "tool_calls"]


class UsageInfo(pydantic.BaseModel):
    """Usage information."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatCompletionResponse(pydantic.BaseModel):
    """Chat completion."""

    id: str
    object: typing.Literal["chat.completion"]
    created: int
    model: str
    choices: list[ChatCompletionChoice]
    usage: UsageInfo | None = None
    logprobs: dict[str, list[str]] | None = None


class MistralChatRequest(pydantic.BaseModel):
    messages: list[MistralMessage]
    max_tokens: int | None = None
    stream: bool | None = False
    random_seed: int | None = None
    stop: str | list[str] | None = None
    temperature: float | None = 0.7
    top_p: float | None = 1
    response_format: ResponseFormat | None = None
    tools: list[Tool] | None = pydantic.Field(default=None, validation_alias=pydantic.AliasChoices("schema", "tools"))
    tool_choice: typing.Literal["none", "any", "auto"] | None = None
    safe_prompt: bool | None = False

    @pydantic.field_validator("tools", mode="before")
    @classmethod
    def validate_tool(
        cls: type["MistralChatRequest"], v: list[type[pydantic.BaseModel]] | None
    ) -> list[dict[str, typing.Any]] | None:
        """Validate tool."""
        if v is None:
            return v
        if isinstance(v, type) and issubclass(v, pydantic.BaseModel):
            return [{"type": "function", "function": get_openai_schema(v)}]
        return [{"type": "function", "function": get_openai_schema(m)} for m in v]

    @pydantic.model_validator(mode="after")
    @classmethod
    def validate_tool_calls(cls: type["MistralChatRequest"], data: typing.Any) -> typing.Any:
        """Validate tool_calls."""
        if data.tools:
            data.tool_choice = "any" if len(data.tools) == 1 else "auto"
        return data
