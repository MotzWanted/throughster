import asyncio
import pydantic
import os
import re

import typing as typ

from throughster.factory import create_interface
from throughster.core.models import BaseResponse
from throughster.prompt import Prompt

os.environ["VLLM_HOST"] = "http://localhost:{deployment}/v1"
os.environ["VLLM_PORTS"] = "6536"


class AlignmentModel(pydantic.BaseModel):
    """Model for alignment."""

    chain_of_thought: str = pydantic.Field(
        ..., description="Think step-by-step about what options best describe the sentence."
    )
    labels: list[int] = pydantic.Field(..., description="List of options that best describe the sentence.")


MAX_ATTEMPTS = 3
SAMPLING_PARAMS = {"temperature": 0.05, "tools": [AlignmentModel]}

ALIGMENT_TEMPLATE = [
    {
        "role": "user",
        "content": """You are a medical expert asserting medical documentation sentence by sentence.
Your goal is figure out what options provide sufficient information to support the given sentence.

Instruction guidelines:
1.  Select the option(s) that best describes the sentence.
2.  If none of the options are relevant, select the option [0] "None of the following options is relevant."
3.  If the sentence is relevant to an option, but contradict it,
    select the option and add a minus sign (-) before the option number.
    For example, if the sentence contradicts option 2, select [-2].
4.  Output all relevant options as a comma-separated list. E.g., [1,2,3,-4]

Medical documentation: "{{summary}}"

Options:
[0] None of the following options is relevant.
{% for option in text %}
[{{ loop.index }}] "{{option}}".
{% endfor %}
""",
    },
]


def split_transcript(text):
    """Split the transcript into segments based on punctiuation."""
    pattern = r"[^?.!]+[?.!]?"
    segments = re.findall(pattern, text)
    return [segment.strip() for segment in segments]


async def run_async_funcs(coroutines: list[asyncio.Future]):
    """Run the async coroutines and return the results."""
    results = await asyncio.gather(*coroutines)
    return results


if __name__ == "__main__":
    AlignmentPrompt = Prompt()

    transcript = (
        "What seems to be the problem today? "
        "I hurt my leg while playing soccer yesterday. "
        "Can you describe the pain and show me where it hurts? "
        "It's a sharp, stabbing pain in my left calf when I try to walk."
        "I suggest you try and get some rest"
    )

    list_of_generated_facts = [
        "Hurt leg while playing football",
        "Sharp and stabbing pain, left calf",
        "Should rest leg",
        "Keep moving around",
    ]

    # Create the vLLM client
    vthroughster = create_interface("vllm")
    model = vthroughster.list_models()[0].id

    # Split the transcript into segments
    transcript_segmented = split_transcript(transcript)
    requests = [
        {
            "messages": AlignmentPrompt(ALIGMENT_TEMPLATE, {"summary": s, "text": transcript_segmented}),
            **SAMPLING_PARAMS,
        }
        for s in list_of_generated_facts
    ]

    async def generate(requests: list[dict[str, typ.Any]]) -> list[BaseResponse]:
        coroutines = [
            vthroughster.structured_call(r, schema=AlignmentModel, max_attempts=MAX_ATTEMPTS) for r in requests
        ]
        results = await asyncio.gather(*coroutines)
        return results

    # Prepare and run the async LLM calls
    results = asyncio.run(generate(requests))
    # Print results of the alignment
    for i, result in enumerate(results):
        alignment_data: AlignmentModel = result.validated_schema  # type: ignore
        print("=" * 10, "Alignment result", "=" * 10, "\n")
        print("Generated fact:", f'"{list_of_generated_facts[i]}"')
        print("Relevant segments:")

        [print(f"\t{[label]}", transcript_segmented[label - 1]) for label in alignment_data.labels]
        print("CoT Reason:\n\t", alignment_data.chain_of_thought, "\n")
