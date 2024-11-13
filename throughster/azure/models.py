from typing import Any, Literal

import pydantic

from throughster.azure.utils import get_openai_schema

Roles = Literal["system", "user", "assistant", "tool"]


class ResponseFormat(pydantic.BaseModel):
    """Format of the model output."""

    type: str | pydantic.Json = "json_object"


class Tool(pydantic.BaseModel):
    """Tool called by model."""

    type: Literal["function"] = "function"
    function: dict[str, Any]


class ToolChoice(pydantic.BaseModel):
    """Tool choice."""

    type: Literal["function"] | None = "function"
    function: dict[Literal["name"], str] = pydantic.Field(
        ..., description='Forces the model to output {"name": "my_function"}.'
    )


class OpenAIMessage(pydantic.BaseModel):
    """Chat message."""

    role: Roles
    content: str | None = ""
    name: str | None = None
    tool_calls: list[dict] | None = None
    tool_call_id: str | None = None


class OpenAIChatRequest(pydantic.BaseModel):
    """OpenAI chat request."""

    messages: list[OpenAIMessage]
    frequency_penalty: float | None = 0
    logit_bias: dict[int, float] | None = None
    max_tokens: int | None = None
    n: int | None = 1
    presence_penalty: None | float = 0
    stream: bool | None = False
    seed: int | None = None
    stop: str | list[str] | None = None
    temperature: float | None = 0.7
    top_p: float | None = 1
    user: str | None = None
    response_format: ResponseFormat | None = None
    tools: list[Tool] | None = pydantic.Field(default=None, validation_alias=pydantic.AliasChoices("schema", "tools"))
    tool_choice: Literal["none", "auto"] | ToolChoice | None = None

    @pydantic.field_validator("tools", mode="before")
    @classmethod
    def validate_tool(
        cls: type["OpenAIChatRequest"], v: type[pydantic.BaseModel] | list[type[pydantic.BaseModel]] | None
    ) -> list[dict[str, Any]] | None:
        """Validate tool."""
        if v is None:
            return v
        if isinstance(v, type) and issubclass(v, pydantic.BaseModel):
            return [{"type": "function", "function": get_openai_schema(v)}]
        return [{"type": "function", "function": get_openai_schema(m)} for m in v]

    @pydantic.model_validator(mode="after")  # pyright: ignore reportArgumentType
    @classmethod
    def validate_tool_calls(cls: type["OpenAIChatRequest"], data: Any) -> Any:
        """Validate tool_calls."""
        if data.tools:
            data.tool_choice = (
                ToolChoice(function={"name": data.tools[0].function["name"]}) if len(data.tools) == 1 else "auto"
            )
        return data
