# Send Queries to the NVIDIA Triton Server for NeMo LLMs

After starting the service with the scripts supplied in the TensorRT-LLM, vLLM, and In-Framework sections, the service will be in standby mode, ready to receive incoming requests. There are multiple methods available for sending queries to this service.

* Use the Query Script or Classes: Execute the query script or classes within the currently running container.
* PyTriton: Utilize PyTriton to send requests directly.
* HTTP Requests: Make HTTP requests using various tools or libraries.


## Send a Query using the Script

Choose the appropriate query script based on your deployment type. Each deployment method has its own specialized query script with relevant parameters.


### General TensorRT-LLM Models

For the models deployed with TensorRT-LLM using the [deployment script described here](../nemo_models/optimized/tensorrt-llm.md):

```shell
python /opt/Export-Deploy/scripts/deploy/nlp/query.py --url "http://localhost:8000" --model_name llama --prompt "What is the capital of United States?"
```

**Additional parameters:**
- `--prompt_file`: Read prompt from file instead of command line
- `--max_output_len`: Max output token length (default: 128)
- `--top_k`: Top-k sampling (default: 1)
- `--top_p`: Top-p sampling (default: 0.0)
- `--temperature`: Sampling temperature (default: 1.0)
- `--lora_task_uids`: LoRA task UIDs for LoRA-enabled models
- `--stop_words_list`: List of stop words
- `--bad_words_list`: List of words to avoid
- `--no_repeat_ngram_size`: N-gram size for repetition penalty

### In-Framework PyTorch NeMo Models

For NeMo models deployed with PyTorch in-framework using the [deployment script described here](../nemo_models/in-framework.md):

```shell
python /opt/Export-Deploy/scripts/deploy/nlp/query_inframework.py --url "http://localhost:8000" --model_name llama --prompt "What is the capital of United States?"
```

**Specific parameters:**
- `--compute_logprob`: Return log probabilities


### In-Framework HuggingFace Models

For HuggingFace models deployed with in-framework backend using the [deployment script described here](../automodel/automodel-in-framework.md):

```shell
python /opt/Export-Deploy/scripts/deploy/nlp/query_inframework_hf.py --url "http://localhost:8000" --model_name llama --prompt "What is the capital of United States?"
```

**Additional parameters:**
- `--output_logits`: Return raw logits from the model output
- `--output_scores`: Return token probability scores from the model output


### vLLM Deployments

For models deployed with vLLM using the [deployment script described here](../nemo_models/optimized/vllm.md):

```shell
python /opt/Export-Deploy/scripts/deploy/nlp/query_vllm.py --url "http://localhost:8000" --model_name llama --prompt "What is the capital of United States?"
```

**vLLM-specific parameters:**
- `--max_tokens`: Maximum tokens to generate (default: 16)
- `--min_tokens`: Minimum tokens to generate (default: 0)
- `--n_log_probs`: Number of log probabilities per output token
- `--n_prompt_log_probs`: Number of log probabilities per prompt token
- `--seed`: Random seed for generation

**Note:** The `--max_output_len` parameter is not available in the `query_vllm.py` script. Instead, use `--max_tokens` to control the maximum number of output tokens.


### TensorRT-LLM API Deployments

For models deployed using TensorRT-LLM API using the [deployment script described here](../nemo_models/optimized/tensorrt-llm.md):

```shell
python /opt/Export-Deploy/scripts/deploy/nlp/query_trtllm_api.py --url "http://localhost:8000" --model_name llama --prompt "What is the capital of United States?"
```

**TensorRT-LLM API parameters:**
- `--max_length`: Maximum length of generated sequence (default: 256)

   

## Send a Query using the NeMo APIs

The NeMo Framework provides multiple query APIs to send requests to the Triton server for different deployment types. These APIs are only accessible from the NeMo Framework container. Choose the appropriate query class based on your deployment method:

### NemoQueryLLM  (TensorRT-LLM Models)

For deployed TensorRT-LLM models with comprehensive parameter support:

1. To run the request example using the general NeMo API, run the following command:

   ```python
   from nemo_deploy.nlp import NemoQueryLLM

   nq = NemoQueryLLM(url="localhost:8000", model_name="llama")
   output = nq.query_llm(prompts=["What is the capital of United States?"], max_output_len=10, top_k=1, top_p=0.0, temperature=1.0)
   print(output)
   ```

2. If there is a LoRA model, run the following command to send a query:

   ```python
   output = nq.query_llm(prompts=["What is the capital of United States?"], max_output_len=10, top_k=1, top_p=0.0, temperature=1.0, lora_uids=["0"])
   ```

### NemoQueryLLMPyTorch (PyTorch-based Models)

For PyTorch-based LLM deployments with extended parameter support:

```python
from nemo_deploy.nlp import NemoQueryLLMPyTorch

nq = NemoQueryLLMPyTorch(url="localhost:8000", model_name="llama")
output = nq.query_llm(
    prompts=["What is the capital of United States?"],
    max_length=100,
    top_k=1,
    top_p=0.0,
    temperature=1.0,
    use_greedy=True,
    repetition_penalty=1.0
)
print(output)
```

### NemoQueryLLMHF (HuggingFace Models)

For HuggingFace model deployments:

```python
from nemo_deploy.nlp import NemoQueryLLMHF

nq = NemoQueryLLMHF(url="localhost:8000", model_name="llama")
output = nq.query_llm(
    prompts=["What is the capital of United States?"],
    max_length=100,
    top_k=1,
    top_p=0.0,
    temperature=1.0
)
print(output)
```

### NemoQueryTRTLLMAPI (TensorRT-LLM API)

For TensorRT-LLM API deployments:

```python
from nemo_deploy.nlp import NemoQueryTRTLLMAPI

nq = NemoQueryTRTLLMAPI(url="localhost:8000", model_name="llama")
output = nq.query_llm(
    prompts=["What is the capital of United States?"],
    max_length=100,
    top_k=1,
    top_p=0.8,
    temperature=1.0
)
print(output)
```

### NemoQueryvLLM (vLLM Deployments)

For vLLM deployments with OpenAI-compatible responses:

```python
from nemo_deploy.nlp import NemoQueryvLLM

nq = NemoQueryvLLM(url="localhost:8000", model_name="llama")
output = nq.query_llm(
    prompts=["What is the capital of United States?"],
    max_tokens=100,
    top_k=1,
    top_p=0.8,
    temperature=1.0,
    seed=42
)
print(output)
```

## Query Class Selection Guide

Choose the appropriate query class based on your deployment type:

- **NemoQueryLLM**: TensorRT-LLM model deployments using TensorRT-LLM engine
- **NemoQueryTRTLLMAPI**: TensorRT-LLM API deployments with simplified parameter set. This is specific to TensorRT-LLM's new API to export models to TensorRT-LLM
- **NemoQueryLLMPyTorch**: PyTorch-based model deployments
- **NemoQueryLLMHF**: HuggingFace model deployments 
- **NemoQueryvLLM**: vLLM deployments that return OpenAI-compatible responses


