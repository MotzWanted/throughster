from typing import Any

import pydantic
from docstring_parser import parse


def generate_schema_from_pydantic(tool: type[pydantic.BaseModel]) -> dict[str, Any]:
    """Return the schema in the format of OpenAI's schema as json schema.

    Note:
    ----
        Its important to add a docstring to describe how to best use this class,
        it will be included in the description attribute and be part of the prompt.

    Returns:
    -------
        model_json_schema (dict): A dictionary in the format of OpenAI's schema as json schema

    """
    schema = tool.model_json_schema()
    docstring = parse(tool.__doc__ or "")
    parameters = {k: v for k, v in schema.items() if k not in ("title", "description")}
    for param in docstring.params:
        if (name := param.arg_name) in parameters["properties"] and (
            description := param.description and "description" not in parameters["properties"][name]
        ):
            parameters["properties"][name]["description"] = description

    parameters["required"] = sorted(k for k, v in parameters["properties"].items() if "default" not in v)

    if "description" not in schema:
        if docstring.short_description:
            schema["description"] = docstring.short_description
        else:
            schema["description"] = (
                f"Correctly extracted `{tool.__name__}` with all the required parameters with correct types"
            )

    return schema
