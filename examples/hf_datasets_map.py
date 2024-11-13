"""Example of using the `llm-client` for offline processing with Hugging Face `datasets.map()` or `datasets.filter()`."""  # noqa: E501

import asyncio
from functools import partial
import datasets
import typing as typ

from pydantic_settings import BaseSettings, SettingsConfigDict
import rich

from throughster.base import ModelInterface
from throughster.factory import create_interface
from throughster.hf_datasets import HfOperation, transform
from throughster.core.models import BaseResponse
from throughster.prompt import Prompt

SYSTEM_PROMPT = [{"role": "system", "content": "You are a helpful translator."}]
USER_PROMPT = [
    {
        "role": "user",
        "content": """Please translate the following text from {{ source }} to {{ target }}:
"{{ text }}""",
    }
]

SUPPORTED_LANGUAGES = {
    "en": "English",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "it": "Italian",
}

LanguageCode = typ.Literal["en", "de", "fr", "es", "it"]

BATCH_SIZE = 5
NUM_WORKERS = 2


class Arguments(BaseSettings):
    provider: str = "vllm"
    api_base: str = "http://localhost:6538/v1"
    deployment: str = "meta-llama/Meta-Llama-3.1-70B-instruct"
    source: LanguageCode = "en"
    target: LanguageCode = "fr"

    model_config = SettingsConfigDict(cli_parse_args=True, frozen=True)


class TranslateOp(HfOperation):
    """Make translation tasks."""

    def __init__(
        self,
        init_client_fn: typ.Callable[..., ModelInterface],
        translate_from: LanguageCode,
        translate_to: LanguageCode,
        prompt: Prompt,
        user_prompt: list[dict[str, str]],
        sampling_params: dict[str, typ.Any],
        text_key: str = "text",
    ) -> None:
        self._client = None
        self.translate_from = SUPPORTED_LANGUAGES[translate_from]
        self.translate_to = SUPPORTED_LANGUAGES[translate_to]
        self.prompt = prompt
        self.user_prompt = user_prompt
        self.text_key = text_key
        self.sampling_params = sampling_params
        super().__init__(init_client_fn)

    def __call__(self, batch: dict[str, list[typ.Any]], idx: None | list[int] = None) -> dict[str, list[typ.Any]]:
        """Translate the input batch."""
        requests = self.create_requests(batch)
        results = asyncio.run(self.translate(requests=requests))

        return {"text": batch[self.text_key], "translation": [r.content for r in results]}

    def create_requests(self, batch: dict[str, list[typ.Any]]) -> list[dict[str, typ.Any]]:
        """Create translation requests."""
        return [
            {
                "messages": self.prompt(
                    prompt=self.user_prompt,
                    prompt_variables={"text": text, "source": self.translate_from, "target": self.translate_to},
                ),
                **self.sampling_params,
            }
            for text in batch[self.text_key]
        ]

    async def translate(self, requests: list[dict[str, typ.Any]]) -> list[BaseResponse]:
        """Async wrapper for the translation operation."""
        return await self.client.batch_call(requests)


def run(args: Arguments):
    data = datasets.load_dataset("EleutherAI/lambada_openai", args.source, split="test[:100]")

    if not isinstance(data, (datasets.Dataset, datasets.DatasetDict)):
        raise ValueError("The dataset must be a `datasets.Dataset` or `datasets.DatasetDict`.")

    sampling_params = {"temperature": 0.5, "top_p": 0.95, "max_tokens": 512}
    init_client_fn = partial(
        create_interface, provider=args.provider, api_base=args.api_base, model_name=args.deployment
    )
    translate_op = TranslateOp(
        init_client_fn=init_client_fn,
        translate_from=args.source,
        translate_to=args.target,
        prompt=Prompt(system_prompt=SYSTEM_PROMPT),
        user_prompt=USER_PROMPT,
        sampling_params=sampling_params,
    )

    rich.print(f"Original data: {data}", f"{data[0]}", sep="\n\n")
    data = transform(
        data,
        translate_op,
        operation="map",
        # HuggingFace specific parameters
        desc="Translating text...",
        batched=True,
        batch_size=BATCH_SIZE,
        num_proc=NUM_WORKERS,
        load_from_cache_file=False,
    )
    rich.print(f"Translated data: {data}", f"{data[0]}", sep="\n\n")


if __name__ == "__main__":
    args = Arguments()
    run(args)
