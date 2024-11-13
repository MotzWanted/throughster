"""Example is an adaptation from `https://github.com/nlpyang/geval` on how to evaluate the quality of texts assessed by a llm."""  # noqa: E501

import asyncio

import rich

from throughster import create_interface, Prompt
from throughster.vllm.client import VllmOpenAiInterface

PROTOCOL_TEMPLATE = """You will be given a chunk of text coming from a transcription of a conversation between a doctor and a patient.

Your task is to rate the chunk of text on one metric.

Please make sure you read and understand these instructions carefully. Please keep this document open while reviewing, and refer to it as needed.

Evaluation Criteria:
Informativeness (1-5) - the degree to which the chunk of text provides useful and relevant information in the context of a medical conversation.
This includes clarity, comprehensiveness, and the presence of significant details that contribute to understanding the patient's condition and the doctor's assessment or advice.

Evaluation Steps:
1. Read the chunk of text carefully and identify the key information being communicated.
2. Determine how much useful and relevant information is provided in the chunk, considering the context of a medical conversation.
3. Assess the clarity and comprehensiveness of the information.
4. Assign a score for informativeness on a scale of 1 to 5, where 1 is the lowest and 5 is the highest based on the Evaluation Criteria.


Example:

Source Text:

"{text}"

Evaluation Form (scores ONLY):

Informativeness (1-5):"""  # noqa: E501

SCORING_RANGE = [1, 2, 3, 4, 5]

TEXT = """"Patient: I've been experiencing a sharp pain in my lower back for the past two weeks.
It gets worse when I try to lift anything heavy or when I'm sitting for a long time.
I haven't taken any medication for it yet, but I tried using a heating pad, and it helped a little.
Doctor: It sounds like you might be dealing with a muscle strain. It's good that the heating pad is providing some relief.
I recommend avoiding heavy lifting and trying some over-the-counter pain medication like ibuprofen.
If the pain persists or worsens, we should consider scheduling an MRI to rule out any more serious issues."""  # noqa: E501

PromptTemplate = Prompt()


async def main():
    vthroughster: VllmOpenAiInterface = create_interface("vllm")  # type: ignore
    protocol = PromptTemplate(PROTOCOL_TEMPLATE, {"text": TEXT})
    sampling_params = {"temperature": 0.0}

    response = await vthroughster.nlg_evaluation(sampling_params, protocol=protocol, scores=SCORING_RANGE)

    rich.print(f"Quality of the text assessed by the llm: {response}")


if __name__ == "__main__":
    asyncio.run(main())
