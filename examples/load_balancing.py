"""Example on how to implement a hacky load balancing mechanism for the llm-client.

NOTE: Requires you to define sufficient environment variables for the provider you want to use.
"""

import argparse
import asyncio

from throughster import create_interface, Prompt
from throughster.random_load_balancer import RandomLoadBalancer

SYSTEM_PROMPT = [{"role": "system", "content": "You are a helpful assistant."}]
USER_PROMPT = [{"role": "user", "content": "Who won the world series in {{ year }}?"}]

PredictionTemplate = Prompt(system_prompt=SYSTEM_PROMPT)


async def main(
    provider: str,
    deployments: list[str],
):
    throughsters = [create_interface(provider=provider, model_name=d) for d in deployments]
    llm_balancer = RandomLoadBalancer(throughsters)

    sampling_params = {"temperature": 0.5, "top_p": 0.95, "max_tokens": 24}

    requests = [
        {
            "messages": PredictionTemplate(USER_PROMPT, {"year": year}),
            **sampling_params,
        }
        for year in range(2010, 2022)
    ]

    # Each request is randomly send to one of the deployments.
    results = await llm_balancer.batch_call(requests)

    for idx, result in enumerate(results):
        print(f"Prediction for the year {2010 + idx}:")
        print(result.model_dump())


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Example of using async requests with the llm-client.")
    parser.add_argument(
        "--provider", type=str, choices=["vllm", "azure", "mistral"], default="azure", help="The provider to use."
    )
    parser.add_argument("--deployments", type=str, nargs="+", help="The deployments to use.")
    args = parser.parse_args()
    asyncio.run(main(provider=args.provider, deployments=args.deployments))
