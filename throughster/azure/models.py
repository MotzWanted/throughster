from typing import Any, Dict, Literal, Optional, Type

import docstring_parser
import pydantic
from openai.lib._pydantic import to_strict_json_schema
from pydantic.fields import Field

from throughster.azure.utils import get_openai_schema

Roles = Literal["system", "user", "assistant", "tool"]


class ResponseFormat(pydantic.BaseModel):
    """Format of the model output."""

    type: str | pydantic.Json = "json_object"


class JSONSchema(pydantic.BaseModel):
    """OpenAI JSON Schema."""

    name: str
    description: Optional[str] = None
    schema_: Optional[Dict[str, object]] = Field(alias="schema", default=None)
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
    max_completion_tokens: int | None = None
    reasoning_effort: str | None = "medium"
    n: int | None = 1
    presence_penalty: None | float = 0
    stream: bool | None = False
    seed: int | None = None
    stop: str | list[str] | None = None
    top_p: float | None = 1
    user: str | None = None
    response_format: ResponseFormat | ResponseFormatJSONSchema | None = pydantic.Field(
        default=None, validation_alias="schema"
    )
    tools: list[Tool] | None = None
    tool_choice: Literal["none", "auto", "required"] | ToolChoice | None = None

    @pydantic.field_validator("response_format", mode="before")
    @classmethod
    def validate_response_format(
        cls: type["OpenAIChatRequest"],
        v: str | pydantic.Json | Type[pydantic.BaseModel] | None,
    ) -> str | pydantic.Json | Type[pydantic.BaseModel] | None:
        """Validate the response format."""
        if v is None:
            return v
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
        return v

    @pydantic.field_validator("tools", mode="before")
    @classmethod
    def validate_tool(
        cls: type["OpenAIChatRequest"],
        v: (
            type[pydantic.BaseModel]
            | list[type[pydantic.BaseModel]]
            | list[dict[str, Any]]
            | None
        ),
    ) -> list[dict[str, Any]] | None:
        """Validate tool."""
        if v is None:
            return v
        if isinstance(v, type) and issubclass(v, pydantic.BaseModel):
            return [{"type": "function", "function": get_openai_schema(v)}]

        validated_tools = []
        for tool in v:
            if (
                isinstance(tool, dict)
                and "type" in tool
                and "function" in tool
                and tool["type"] == "function"
                and "parameters" in tool["function"]
                and isinstance(tool["function"]["parameters"], dict)
            ):
                validated_tools.append(tool)
            if isinstance(tool, type) and issubclass(tool, pydantic.BaseModel):
                validated_tools.append(
                    {"type": "function", "function": get_openai_schema(tool)}
                )
        return validated_tools

    @pydantic.model_validator(mode="after")  # pyright: ignore reportArgumentType
    @classmethod
    def validate_tool_calls(cls: type["OpenAIChatRequest"], data: Any) -> Any:
        """Validate tool_calls."""
        if data.tools:
            if isinstance(data.tools, type) and issubclass(
                data.tools, pydantic.BaseModel
            ):
                schema = get_openai_schema(data.tools)
                data.tool_choice = (
                    ToolChoice(function={"name": schema["name"]})
                    if len(data.tools) == 1
                    else "auto"
                )
                return data
            if all(
                isinstance(tool, type) and issubclass(tool, pydantic.BaseModel)
                for tool in data.tools
            ):
                data.tool_choice = (
                    ToolChoice(function={"name": data.tools[0].function["name"]})
                    if len(data.tools) == 1
                    else "auto"
                )
                return data
            if data.tool_choice:
                if data.tool_choice == "auto":
                    return data
                if data.tool_choice == "required":
                    return data
                if (
                    isinstance(data.tool_choice, dict)
                    and "type" in data.tool_choice
                    and "function" in data.tool_choice
                    and data.tool_choice["type"] == "function"
                ):
                    return data
                if issubclass(data.tool_choice, pydantic.BaseModel):
                    if data.tools:
                        data.tool_choice = (
                            ToolChoice(
                                function={"name": data.tools[0].function["name"]}
                            )
                            if len(data.tools) == 1
                            else "auto"
                        )
                        return data
        if not data.tools:
            if data.tool_choice == "required":
                raise ValueError("Tool choice is required but no tools were provided.")
            if (
                isinstance(data.tool_choice, dict)
                and "type" in data.tool_choice
                and "function" in data.tool_choice
                and data.tool_choice["type"] == "function"
            ):
                raise ValueError("Tool choice is required but no tools were provided.")
        return data
