"""Example of using the structured decoding interface with the llm-client.

NOTE: Requires you to define sufficient environment variables for the provider you want to use.
"""

import argparse
import asyncio
import pydantic
import rich

from throughster import create_interface


class QuestionAnswer(pydantic.BaseModel):
    question: str
    answer: str


QUESTION = "What is the meaning of life?"
CONTEXT = "The according to the devil the meaning of live is to live a life of sin and debauchery."


async def main(args):
    client = create_interface("azure")
    sampling_params = {"temperature": 0.5, "max_tokens": 100}

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
        **sampling_params,
    }

    response = await client.structured_call(request, schema=QuestionAnswer, max_attempts=3)

    rich.print(response)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Example of using async requests with the llm-client.")
    parser.add_argument(
        "--provider", type=str, choices=["vllm", "azure", "mistral"], default="vllm", help="The provider to use."
    )
    args = parser.parse_args()
    asyncio.run(main(args))
