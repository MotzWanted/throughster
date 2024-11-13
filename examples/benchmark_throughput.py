"""Measures the throughtput of a vLLM deployment."""

import asyncio
from datetime import date
import multiprocessing as mp
import pathlib
import time
import json

import typing as typ
from pydantic_settings import BaseSettings, SettingsConfigDict
import rich
import tqdm

from throughster.core.models import BaseResponse
from examples.counter import CounterClient, CounterServer

from throughster import create_interface, ModelInterface

THROUGHPUT_FOLDER = pathlib.Path.cwd() / "throughput"


class Arguments(BaseSettings):
    """Args for the script."""

    provider: str = "vllm"
    api_base: str = "http://localhost:6539/v1"
    prompt: str = "Write a 2-page story."
    deployment: str = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    num_workers: str = "1:16:32:64:128:256"
    n_samples: int = 300
    max_tokens: int = 256
    n: int = 1
    instances: int = 1

    model_config = SettingsConfigDict(cli_parse_args=True)


class WithCounterServer:
    """A Context manager to spawn a CounterServer and log the results to a file."""

    def __init__(self, *, path: pathlib.Path, n_instances: int, stats_: dict = {}) -> None:
        self.start_time: None | float = None
        self.path = path
        self.n_instances = n_instances
        self.stats_ = stats_

    def __enter__(self) -> CounterClient:
        self.server = CounterServer()
        self.server.__enter__()
        self.start_time = time.perf_counter()
        return self.server.get_client()

    def __exit__(self, *args: typ.Any) -> None:
        if self.start_time is None:
            raise ValueError("You must enter the context manager first")
        elapsed_time = time.perf_counter() - self.start_time
        counts = self.server.get()
        if len(counts):
            stats = {
                "raw_elapsed": elapsed_time,
                "raw_counts": counts,
                "throughput_per_instance": {k: v / elapsed_time / self.n_instances for k, v in counts.items()},
            }
            self.stats_.update(stats)
            rich.print({self.path.name: stats})
            with open(self.path, "w") as f:
                json.dump(stats, f, indent=2)
        self.server.__exit__(*args)


class ChatCompleteFn:
    """Complete prompt."""

    _client: None | ModelInterface = None

    def __init__(
        self,
        provider: str,
        api_base: str,
        deployment: str,
        max_tokens: int,
        temperature: float,
        n: int,
        counter: CounterClient,
    ) -> None:
        """Initialize."""
        self.provider = provider
        self.api_base = api_base
        self.deployment = deployment
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.n = n
        self.counter = counter

    @property
    def client(self) -> ModelInterface:
        """Get the client."""
        if self._client:
            return self._client
        return create_interface(provider=self.provider, api_base=self.api_base, model_name=self.deployment)

    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_client"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        self.__dict__.update(state)

    def __call__(self, prompt: str) -> int:
        """Call the generate function."""
        request = {
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        response: BaseResponse = asyncio.run(self.complete(request))
        self.counter.increment(response.usage.model_dump() if response.usage is not None else {})
        return response.usage.__getattribute__("total_tokens")

    async def complete(self, request: dict[str, typ.Any]) -> BaseResponse:
        """Complete the prompt."""
        return await self.client.call(request)


def benchmark_throughput(path: pathlib.Path, num_workers: int, args: Arguments) -> dict[str, float]:
    experiment = {
        "prompt": args.prompt,
        "n_instances": args.instances,
        "n_samples": args.n_samples,
        "max_tokens": args.max_tokens,
    }
    experiment_file_name = f"num_workers_{num_workers}_n_samples_{args.n_samples}_max_tokens_{args.max_tokens}.json"
    stats_output_path = path / experiment_file_name
    with WithCounterServer(path=stats_output_path, n_instances=args.instances, stats_=experiment) as counter:
        fn = ChatCompleteFn(
            provider=args.provider,
            api_base=args.api_base,
            deployment=args.deployment,
            max_tokens=args.max_tokens,
            temperature=0.5,
            n=args.n,
            counter=counter,
        )
        t = time.perf_counter()
        ntokens = 0
        pbar = tqdm.tqdm(total=args.n_samples)
        with mp.Pool(num_workers) as pool:
            for m in pool.imap_unordered(fn, [args.prompt] * args.n_samples):
                ntokens += m
                pbar.update(1)
                pbar.set_description(
                    f"{ntokens/(time.perf_counter() - t):.2f} tokens/s (n={args.n}, num_workers={num_workers})"
                )

        return counter.get()


def run(args: Arguments) -> None:
    """Run the script."""
    rich.print(args)

    stats_folder_path = THROUGHPUT_FOLDER / f"{args.deployment}" / str(date.today())
    stats_folder_path.mkdir(exist_ok=True, parents=True)

    num_workers_ = [int(x) for x in args.num_workers.split(":")] if ":" in args.num_workers else [int(args.num_workers)]
    for num_workers in num_workers_:
        benchmark_throughput(stats_folder_path, num_workers, args)


if __name__ == "__main__":
    settings = Arguments()
    run(settings)
