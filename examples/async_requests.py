"""Example of using multiple async requests with the llm-client.

NOTE: Requires you to define sufficient environment variables for the provider you want to use.
"""

import asyncio

from pydantic_settings import BaseSettings, SettingsConfigDict
import rich
from throughster import create_interface, Prompt

SYSTEM_PROMPT = [{"role": "system", "content": "You are a world class linguist."}]
USER_PROMPT = [{"role": "user", "content": """Translate this "{{ text }}" to {{ language }}."""}]
LANGUAGES = ["French", "Spanish", "Italian"]


class Arguments(BaseSettings):
    provider: str = "vllm"
    api_base: str = "http://localhost:6538/v1"
    deployment: str = "meta-llama/Meta-Llama-3.1-70B-instruct"

    model_config = SettingsConfigDict(cli_parse_args=True, frozen=True)


async def main(args):
    throughster = create_interface(provider=args.provider, api_base=args.api_base, model_name=args.deployment)
    TranslatePrompt = Prompt(system_prompt=SYSTEM_PROMPT)
    sampling_params = {"temperature": 0.5}

    text_to_translate = "Happy birthday!"
    requests = [
        {
            "messages": TranslatePrompt(USER_PROMPT, {"text": text_to_translate, "language": language}),
            **sampling_params,
        }
        for language in LANGUAGES
    ]

    results = await throughster.batch_call(requests)

    for idx, result in enumerate(results):
        rich.print(f"Translation to {LANGUAGES[idx]}:")
        rich.print(result.model_dump())


if __name__ == "__main__":
    args = Arguments()
    asyncio.run(main(args))
