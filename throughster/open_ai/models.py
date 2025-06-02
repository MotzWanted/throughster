from typing import Any, Dict, Literal, Optional, Type

import docstring_parser
import pydantic
from openai.lib._pydantic import to_strict_json_schema


Roles = Literal["system", "user", "assistant", "tool"]


class ResponseFormat(pydantic.BaseModel):
    """Format of the model output."""

    type: str | pydantic.Json = "json_object"


class JSONSchema(pydantic.BaseModel):
    """OpenAI JSON Schema."""

    name: str
    description: Optional[str] = None
    schema: Optional[Dict[str, object]] = None
    strict: Optional[bool] = None


class ResponseFormatJSONSchema(pydantic.BaseModel):
    """OpenAI response format as JSON schema."""

    json_schema: JSONSchema
    type: Literal["json_schema"]


class Tool(pydantic.BaseModel):
    """Tool called by model."""

    type: Literal["function"] = "function"
    function: dict[str, Any]
    strict: bool = True


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
    model: str = pydantic.Field(
        ...,
        description="The name or identifier of the deployed model.",
        validation_alias=pydantic.AliasChoices("model", "model_name"),
    )
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
    response_format: ResponseFormat | ResponseFormatJSONSchema | None = pydantic.Field(
        default=None,
        validation_alias=pydantic.AliasChoices("schema", "response_format"),
    )
    tools: list[Tool] | None = None
    tool_choice: Literal["none", "auto", "required"] | ToolChoice | None = None

    @pydantic.field_validator("response_format", mode="before")
    @classmethod
    def validate_response_format(
        cls: type["OpenAIChatRequest"],
        v: str | pydantic.Json | Type[pydantic.BaseModel] | None,
    ) -> ResponseFormat | ResponseFormatJSONSchema | None:
        """Validate the response format."""
        if v is None:
            return v
        if issubclass(v, pydantic.BaseModel):
            schema = to_strict_json_schema(v)
            description = (
                docstring_parser.parse(v.__doc__).description if v.__doc__ else None
            )
            name = v.__name__.encode("ascii", errors="ignore").decode()
            return ResponseFormatJSONSchema(
                type="json_schema",
                json_schema=JSONSchema(
                    name=name,
                    description=description,
                    strict=True,
                    schema=schema,
                ),
            )
        return ResponseFormat(type=v)
