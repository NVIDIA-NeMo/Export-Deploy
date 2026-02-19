# Deploy AutoModel LLMs with Triton Inference Server

This section demonstrates how to deploy [NeMo AutoModel](https://docs.nvidia.com/nemo/automodel/latest/index.html) LLMs with the NVIDIA Triton Inference Server. NeMo AutoModel workflows generate Hugging Face-compatible checkpoints that provide a simplified interface for working with pre-trained language models. These checkpoints maintain high performance during inference, while offering enhanced configurability through the Hugging Face ecosystem.


## Quick Example

1. If you need access to the Llama-3.2-1B model, visit the [Llama 3.2 Hugging Face page](https://huggingface.co/meta-llama/Llama-3.2-1B) to request access.

2. Pull and run the Docker container image. Replace ``:vr`` with your desired version:

   ```shell
   docker pull nvcr.io/nvidia/nemo:vr

   docker run --gpus all -it --rm \
       --shm-size=4g \
       -p 8000:8000 \
       -v ${PWD}/:/opt/checkpoints/ \
       -w /opt/Export-Deploy \
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

### Deploy a NeMo AutoModel LLM

Follow these steps to deploy your model on the Triton Inference Server:

1. Start the container as shown in the **Quick Example** section.

2. Deploy your model:

   ```shell
   python /opt/Export-Deploy/scripts/deploy/nlp/deploy_inframework_hf_triton.py \
      --hf_model_id_path meta-llama/Llama-3.2-1B \
      --triton_model_name llama
   ```

   Available Parameters:
   
   - ``--hf_model_id_path``: Path to a local Hugging Face model directory or model ID from the Hugging Face Hub. (Required)
   - ``--task``: Task type for the Hugging Face model (currently only 'text-generation' is supported). (Optional)
   - ``--device_map``: Device mapping strategy for model placement (e.g., 'auto', 'sequential', etc.). (Optional, mutually exclusive with --tp_plan)
   - ``--tp_plan``: Tensor parallelism plan for distributed inference. 'auto' is the only option supported. (Optional, mutually exclusive with --device_map)
   - ``--trust_remote_code``: Allow loading remote code from the Hugging Face Hub. (Flag; set to enable)
   - ``--triton_model_name``: Name for the model in Triton. (Required)
   - ``--triton_model_version``: Version of the model in Triton. Default: 1
   - ``--triton_port``: Port for the Triton server to listen for requests. Default: 8000
   - ``--triton_http_address``: HTTP address for the Triton server. Default: 0.0.0.0
   - ``--max_batch_size``: Maximum inference batch size. Default: 8
   - ``--debug_mode``: Enables additional debug logging messages from the script. (Flag; set to enable)

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
- `-mn`, `--model_name`: Name of the model as deployed on Triton server (required)
- `-p`, `--prompt`: Text prompt to send to the model (mutually exclusive with --prompt_file; required if --prompt_file not given)
- `-pf`, `--prompt_file`: Path to file containing the prompt text (mutually exclusive with --prompt; required if --prompt not given)
- `-u`, `--url`: URL of the Triton Inference Server (default: 0.0.0.0)
- `-mol`, `--max_output_len`: Maximum number of tokens to generate in the response (default: 128)
- `-tk`, `--top_k`: Number of highest probability tokens to consider for sampling (default: 1)
- `-tpp`, `--top_p`: Cumulative probability threshold for token sampling (default: 0.0)
- `-t`, `--temperature`: Temperature for controlling randomness in sampling (default: 1.0)
- `-ol`, `--output_logits`: Return raw logits from model output (flag; set to enable)
- `-os`, `--output_scores`: Return token probability scores from model output (flag; set to enable)

For HuggingFace models deployed with in-framework backend using the [deployment script described here](../automodel/automodel-in-framework.md):


### Send a Query using the Export-Deploy APIs

For HuggingFace model deployments:

```python
from nemo_deploy.llm import NemoQueryLLMHF

nq = NemoQueryLLMHF(url="localhost:8000", model_name="llama")
output = nq.query_llm(
    prompts=["What is the capital of United States? "],
    max_length=100,
    top_k=1,
    top_p=0.0,
    temperature=1.0
)
print(output)
```