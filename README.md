<div align="center">
<br/>
<p align="center">
<a href="https://www.python.org/downloads/"><img alt="Python" src="https://img.shields.io/badge/-Python 3.12-blue?style=for-the-badge&logo=python&logoColor=white"></a>
<a href="https://www.python.org/downloads/"><img alt="Python" src="https://img.shields.io/badge/-Python 3.11-blue?style=for-the-badge&logo=python&logoColor=white"></a>
<br/>
<a href="https://azure.microsoft.com/en-us/"><img alt="Azure" src="https://img.shields.io/badge/-Azure-0089D6?style=for-the-badge&logo=microsoft-azure&logoColor=white"></a><a href="https://openai.com/"><img alt="OpenAI" src="https://img.shields.io/badge/-OpenAI-412991?style=for-the-badge&logo=openai&logoColor=white"></a>
<a href="https://mistral.ai/"><img alt="Mistral" src="https://img.shields.io/badge/-Mistral-7E57C2?style=for-the-badge&logo=mistral&logoColor=white"></a>
<a href="https://docs.vllm.ai/en/stable/"><img alt="vLLM" src="https://img.shields.io/badge/-vLLM-006400?style=for-the-badge&logo=vl&logoColor=white"></a>
</div>
<br/>

# <auto-title>THROUGHSTER</auto-title>

`Throughster` is designed to provide a *unified interface* for API interactions with various large language model (LLM).

| Supported APIs | Streaming | Function Calling |
|----------------|-----------|------------------|
|  `Azure`       |    ✅     |        ✅          |
|  `OpenAI`       |    ✅     |        ✅          |
| `Mistral`      |     ⚠️   |          ✅        |
|  `vLLM`        |     ✅      |       ✅           |

This package leverages `httpx` for efficient handling of *multiple asynchronous requests*, making it highly suitable for integration with applications that require parallel processing of LLM queries.

> 📌 **Note:** The `httpx.Client` instance uses HTTP connection pooling. This means that when you make several requests to the same host, the `llm-client` will reuse the underlying TCP connection instead of recreating one for every single request.

Additionally, the `llm-client` is designed to be serializable, ensuring compatibility with the HuggingFace `datasets.map` function, providing multiprocessing and caching capabilities out-of-the-box, which is useful in offline settings. See for [here an example](https://github.com/corticph/ml-llm-client/blob/main/examples/hf_datasets_map.py)

## Throughput Benchmark ⏱️
This benchmark measures the throughput of a `vLLM` deployment by running model inference and tracking the tokens processed per second. It evaluates performance under different loads by varying the number of workers.

| Model | GPUs | Workers | Prompt tokens/s | Completion tokens/s | Total tokens/s |
| ---- | ---- | ---- | ---- |---- |---- |
 mistralai/Mixtral-8x7B-Instruct-v0.1 | 2xA100 | 1 | 4.25 | 63.97 | **419.95** |
| mistralai/Mixtral-8x7B-Instruct-v0.1 | 2xA100 | 16 | 27.89 | 419.95 | **447.84** |
| mistralai/Mixtral-8x7B-Instruct-v0.1 | 2xA100 | 32 | 45.25 | 681.34 | **726.58** |
| mistralai/Mixtral-8x7B-Instruct-v0.1 | 2xA100 | 64 | 67.22 | 1012.22 | **1079.43** |
| mistralai/Mixtral-8x7B-Instruct-v0.1 | 2xA100 | 128 | 80.97 | 1219.25 | **1300.22** |
| mistralai/Mixtral-8x7B-Instruct-v0.1 | 2xA100 | 256 | 87.07 | 1311.11 | **1398.17** |

[Here is the script run the benchmark](https://github.com/corticph/ml-llm-client/blob/main/examples/benchmark_throughput.py).

## Installation 📦

<details>
<summary>Production</summary>

```shell
poetry install --without examples
```
</details>

<details>
<summary>Development</summary>

```shell
make install
```
</details>

## Quickstart 💨
This guide shows how to use `llm-client` to:
- Set up a client interface with a supported API provider.
- Stream responses from the API provider.
- Validate prompt templates to confirm all required variables are included before making requests.

> 📌 **Note:** By default, `llm-client` leverages `pydantic-settings` and will, thus, attempt to determine the values of any fields not passed as keyword arguments by reading from the environment. Please make sure to export your environment variables with the correct prefix: `AZURE_OPENAI_`, `MISTRAL_` or `VLLM_`. For example, Azure-OpenAI API requires you to define the following variables:
```bash
export AZURE_OPENAI_API_BASE=<endpoint>
export AZURE_OPENAI_API_KEY=<api-key>
export AZURE_OPENAI_API_VERSION=<api-version>
```
See the [client settings](https://github.com/corticph/ml-llm-client/blob/main/src/throughster/config.py) for more details.

### ⚡️ Initialize `llm-interface`
An `llm-client` instance can be initialized in two ways:

#### **1. 🏭 Factory Method**
If you defined the necessary variables in your environment, you can simply initialize an instance with:
```python
from throughster import create_interface

request = {
    "messages": [{"role": "user", "content": "Say this is a test!"}],
    "temperature": 0.7
}

azure_client = create_interface("azure")
response = await azure_client.call(request)
```
> 📌 **Note:** By default, `create_interface` expects the necessary environment variables to be defined. However, you can also initialize it by passing the variables directly in the code. `create_interface` allows you to input `api_base`, `api_key`, `api_version`, and `model_name` if you prefer this method.


#### **2. 👉 Direct Initialization**
Direct initialization gives you more control over the client configuration. Here's how to set it up:
```python
import httpx
from throughster import OpenAiInterface

request = {
    "messages": [{"role": "user", "content": "Say this is a test!"}],
    "temperature": 0.7
}

API_KEY = "supersecretkey"
API_BASE = "https://<host>.openai.azure.com/openai/deployments/<deployment>/"
API_VERSION = "<some-version>"

azure_client = OpenAiInterface(
    api_base=API_BASE,
    limits=httpx.Limits(),
    timeout=httpx.Timeout(),
    headers={"api-key": API_KEY},
    params={"api-version": self.API_VERSION}
    )

response = await azure_client.call(request)
```

### Streaming Responses
To stream responses from the API provider, use the following approach:
```
from throughster import create_interface

request = {
    "messages": [{"role": "user", "content": "Say this is a test!"}],
    "temperature": 0.7
}

azure_client = create_interface("azure")
response = await azure_client.stream(request)
```

### Structured Responses
To have responses follow a certain schema, use the following approach:

```python
import pydantic

class QuestionAnswer(pydantic.BaseModel):
    question: str
    answer: str


QUESTION = "What is the meaning of life?"
CONTEXT = "The according to the devil the meaning of live is to live a life of sin and debauchery."

client = create_interface("azure")
sampling_params = {"temperature": 0.5, "max_tokens": 100}

request = {
    "messages": [
        {
            "role": "system",
            "content": "You are a system that answers questions based on the context. answer exactly what the question asks using the context.",
        },
        {
            "role": "user",
            "content": f"using the context: {CONTEXT}\n\nAnswer the following question: {QUESTION}",
        },
    ],
    **sampling_params,
}

response = await client.structured_call(request, schema=QuestionAnswer, max_attempts=3)
```

#### Batch calling
To process requests in batches, the `llm-client` provides batch calling for both regular and structured calls; `client.batch_call()` and `client.structured_batch_call()`.

#### Sync calling
If you prefer to run the `llm-client` in a non-async environment, this can be done with the sync implement of above functions; `client.sync_call()`, `client.sync_structured_call()`, `client.sync_batch_call()`, and `client.sync_structured_batch_call()`.

## Prompting
The `Prompt` class allows you to create and render prompts using Jinja2 templates. This is particularly useful for dynamically generating prompts based on different contexts and variables.

```python
from throughster import Prompt

SYSTEM_PROMPT = [
    {
        "role": "system",
        "content": "You are a world class linguist."
    }
]

TranslatePrompt = Prompt(system_prompt=SYSTEM_PROMPT)

USER_PROMPT = [
    {
        "role": "user"
        "content": "Translate this "{{ text }}" to {{ language }}.
    }
]
text_to_translate = "Happy birthday!"
requests = [
    {
        "messages": TranslatePrompt(prompt_template, {"text": text_to_translate, "language": language}),
        **sampling_params,
    }
    for language in ["French", "Spanish", "Italian"]
]
```
> 📌 **Note:** The `Prompt` class provides template validation. Ensures all necessary variables are provided, preventing runtime errors due to missing data.

## Examples 🐙

⚙️ [Structured decoding with `Pydantic`](https://github.com/corticph/ml-llm-client/blob/main/examples/structured_decoding.py)

👨‍🏫 [Evaluate text quality with LLMs](https://github.com/corticph/ml-llm-client/blob/main/examples/nlg_evaluation.py)

🛰️ [Async requests](https://github.com/corticph/ml-llm-client/blob/main/examples/async_requests.py)

🪄 [Hacky local load balancing](https://github.com/corticph/ml-llm-client/blob/main/examples/load_balancing.py)

🤗 [HuggingFace datasets .map integration](https://github.com/corticph/ml-llm-client/blob/main/examples/hf_datasets_map.py)

# Testing

We are relying on generating output of specific version of vllm and azure-openai to have some static data to test against.

### Updating the azure api version:
Run the makefile and set the correct version and credentials
```
export AZURE_DEPLOYMENT=<DEPLOYMENT_NAME>
export AZURE_OPENAI_ENDPOINT=<ENDPOINT>
export AZURE_OPENAI_API_KEY=<KEY>
export TARGET_VERSION=2024-02-01
make generate-test-data-azure
```

### Updating the vllm api version
Remember to have a vllm instance running with the correct version.
```
export VLLM_ENDPOINT=https://localhost/6538
export VLLM_MODEL_NAME=meta-llama/Meta-Llama-3-70B-Instruct
export TARGET_VERSION=v0.4.3
make generate-test-data-vllm
```
