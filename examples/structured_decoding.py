"""Example of using the structured decoding interface with the llm-client.

NOTE: Requires you to define sufficient environment variables for the provider you want to use.
"""

import pydantic
from pydantic_settings import BaseSettings, SettingsConfigDict
import rich

from throughster import create_interface


class QuestionAnswer(pydantic.BaseModel):
    question: str
    answer: str


QUESTION = "What is the meaning of life?"
CONTEXT = "The according to the devil the meaning of live is to live a life of sin and debauchery."


def parser(response: str) -> QuestionAnswer:
    """Parse the response from the LLM into a QuestionAnswer object."""
    try:
        parsed_response = QuestionAnswer.model_validate_json(response)
        return parsed_response
    except pydantic.ValidationError as e:
        raise ValueError(f"Failed to parse response: {e}") from e


class Arguments(BaseSettings):
    provider: str = "vllm"
    api_base: str = "http://localhost:6538/v1"
    deployment: str = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B"

    model_config = SettingsConfigDict(cli_parse_args=True, frozen=True)


def run(args: Arguments):
    client = create_interface(provider=args.provider, api_base=args.api_base, model_name=args.deployment)
    sampling_params = {"temperature": 0.5}

    request = {
        "messages": [
            {
                "role": "system",
                "content": "You are a system that answers questions based on the context. answer exactly what the question asks using the context.",  # noqa: E501
            },
            {
                "role": "user",
                "content": f"using the context: {CONTEXT}\n\nAnswer the following question: {QUESTION}",
            },
        ],
        "schema": QuestionAnswer,
        **sampling_params,
    }

    response = client.sync_structured_call(request, parser=parser, max_attempts=3)

    rich.print(response)


if __name__ == "__main__":
    args = Arguments()
    run(args)
