# Deploy AutoModel LLMs with Triton Inference Server

This section demonstrates how to deploy NeMo AutoModel LLMs with the NVIDIA Triton Inference Server. NeMo AutoModel workflows generate Hugging Face-compatible checkpoints that provide a simplified interface for working with pre-trained language models. These checkpoints maintain high performance during inference, while offering enhanced configurability through the Hugging Face ecosystem.


## Quick Example

1. If you need access to the Llama-3.2-1B model, visit the [Llama 3.2 Hugging Face page](https://huggingface.co/meta-llama/Llama-3.2-1B) to request access.

2. Pull and run the Docker container image. Replace ``:vr`` with your desired version:

   ```shell
   docker pull nvcr.io/nvidia/nemo:vr

   docker run --gpus all -it --rm \
       --shm-size=4g \
       -p 8000:8000 \
       -v ${PWD}/:/opt/checkpoints/ \
       -w /opt/NeMo \
       --name nemo-fw \
       nvcr.io/nvidia/nemo:vr
   ``` 

3. Log in to Hugging Face with your access token:

   ```shell
   huggingface-cli login
   ```

4. Deploy the model to Triton:

   ```python
   python /opt/Export-Deploy/scripts/deploy/nlp/deploy_inframework_hf_triton.py \
      --hf_model_id_path meta-llama/Llama-3.2-1B \
      --triton_model_name llama
   ```
   
   **Note:** If you encounter shared memory errors, increase ``--shm-size`` gradually by 50%.

5. In a separate terminal, access the running container as follows:

   ```shell
   docker exec -it nemo-fw bash
   ```

6. Test the deployed model:

   ```shell
   python /opt/Export-Deploy/scripts/deploy/nlp/query_inframework_hf.py \
      -mn llama \
      -p "What is the color of a banana?" \
      -mol 15
   ```

## Detailed Deployment Guide

### Deploy a NeMo AutoModel LLM Model

Follow these steps to deploy your model on the Triton Inference Server:

1. Start the container as shown in the **Quick Example** section.

2. Deploy your model:

   ```shell
   python /opt/Export-Deploy/scripts/deploy/nlp/deploy_inframework_hf_triton.py \
      --hf_model_id_path meta-llama/Llama-3.2-1B \
      --triton_model_name llama
   ```

   Available Parameters:
   
   - ``--hf_model_id_path``: Path to a local Hugging Face model directory or model ID from the Hugging Face Hub.
   - ``--task``: Task type for the Hugging Face model (currently only 'text-generation' is supported).
   - ``--device_map``: Device mapping strategy for model placement (e.g., 'auto', 'sequential', etc.).
   - ``--tp_plan``: Tensor parallelism plan for distributed inference. 'auto' is the only option supported.
   - ``--trust_remote_code``: Allow loading remote code from the Hugging Face Hub.
   - ``--triton_model_name``: Name for the model in Triton.
   - ``--triton_model_version``: Version of the model. Default is 1.
   - ``--triton_port``: Port for the Triton server to listen for requests. Default is 8000.
   - ``--triton_http_address``: HTTP address for the Triton server. Default is 0.0.0.0.
   - ``--max_batch_size``: Maximum inference batch size. Default is 8.
   - ``--debug_mode``: Enables additional debug logging messages from the script.

3. To use a different model, modify the ``--hf_model_id_path`` parameter. You can specify either a local path or a Hugging Face model ID.

4. For models requiring authentication (e.g., StarCoder1, StarCoder2, LLama3):

   Option 1 - Log in via CLI:
   
   ```shell
   huggingface-cli login
   ```

   Option 2 - Set the environment variable:

   ```shell
   export HF_TOKEN=your_token_here
   ```

### Deploy on Multiple GPUs

For multi-GPU inference:

1. Use ``--tp_plan`` instead of ``--device_map`` (they are mutually exclusive).
2. For distributed inference across GPUs, use ``torchrun``. The following example shows 2 GPUs:

   ```shell
   torchrun --standalone --nnodes=1 --nproc_per_node=2 \
      /opt/Export-Deploy/scripts/deploy/nlp/deploy_inframework_hf_triton.py \
      --hf_model_id_path meta-llama/Llama-3.2-1B \
      --triton_model_name llama \
      --tp_plan auto
   ```

For more information:
   - Device mapping: [Hugging Face Loading Big Models docs](https://huggingface.co/docs/accelerate/main/concept_guides/big_model_inference).
   - Tensor parallelism: [Hugging Face Multi-GPU Inference docs](https://huggingface.co/docs/transformers/v4.47.0/en/perf_infer_gpu_multi).


## How To Send a Query

### Send a Query using the Script

```shell
python /opt/Export-Deploy/scripts/deploy/nlp/query_inframework_hf.py --model_name llama --prompt "What is the capital of United States?"
```

**Parameters:**
- `--model_name`: Name of the Triton model to query (required)
- `--prompt`: Prompt text to send to the model (required, mutually exclusive with --prompt_file)
- `--prompt_file`: Path to a file containing the prompt (mutually exclusive with --prompt)
- `--url`: URL for the Triton server (default: 0.0.0.0)
- `--max_output_len`: Maximum number of output tokens to generate (default: 128)
- `--top_k`: Top-k sampling (default: 1)
- `--top_p`: Top-p (nucleus) sampling (default: 0.0)
- `--temperature`: Sampling temperature (default: 1.0)
- `--output_logits`: Return raw logits from the model output (flag)
- `--output_scores`: Return token probability scores from the model output (flag)

For HuggingFace models deployed with in-framework backend using the [deployment script described here](../automodel/automodel-in-framework.md):


### Send a Query using the Export-Deploy APIs

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