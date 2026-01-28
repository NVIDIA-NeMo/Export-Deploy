# Deploy Megatron-Bridge LLMs with vLLM and Triton Inference Server

This section shows how to use scripts and APIs to export a Megatron-Bridge LLM to vLLM and deploy it with the NVIDIA Triton Inference Server.

## Quick Example

1. Follow the steps in the [Generate a Megatron-Bridge Checkpoint](../gen_mbridge_ckpt.md) to generate a Megatron-Bridge Llama checkpoint.

2. In a terminal, go to the folder where the ``hf_llama31_8B_mbridge`` checkpoint is located. Pull down and run the Docker container image using the command shown below. Change the ``:vr`` tag to the version of the container you want to use:

   ```shell
   docker pull nvcr.io/nvidia/nemo:vr

   docker run --gpus all -it --rm --shm-size=4g -p 8000:8000 \
       -v ${PWD}/hf_llama31_8B_mbridge:/opt/checkpoints/hf_llama31_8B_mbridge/ \
       -w /opt/Export-Deploy \
       --name nemo-fw \
       nvcr.io/nvidia/nemo:vr
   ```

3. Install vLLM by executing the following command inside the container if it is not available in the container:

   ```shell
   cd /opt/Export-Deploy
   uv sync --inexact --link-mode symlink --locked --extra vllm $(cat /opt/uv_args.txt)

   ```

4. Run the following deployment script to verify that everything is working correctly. The script exports the Llama Megatron-Bridge checkpoint to vLLM and subsequently serves it on the Triton server:

   ```shell
   python /opt/Export-Deploy/scripts/deploy/nlp/deploy_vllm_triton.py \
       --model_path_id /opt/checkpoints/hf_llama31_8B_mbridge/iter_0000000/  \
       --model_format megatron_bridge \
       --triton_model_name llama \
       --tensor_parallelism_size 1
   ```

5. If the test yields a shared memory-related error, increase the shared memory size using ``--shm-size`` (gradually by 50%, for example).

6. In a separate terminal, access the running container as follows:

   ```shell
   docker exec -it nemo-fw bash
   ```

7. To send a query to the Triton server, run the following script:

   ```shell
   python /opt/Export-Deploy/scripts/deploy/nlp/query_vllm.py -mn llama -p "The capital of Canada is" -mat 50
   ```

## Use a Script to Deploy Megatron-Bridge LLMs on a Triton Server

You can deploy a LLM from a Megatron-Bridge checkpoint on Triton using the provided script.

### Export and Deploy a Megatron-Bridge LLM

After executing the script, it will export the model to vLLM and then initiate the service on Triton.

1. Start the container using the steps described in the **Quick Example** section.

2. To begin serving the downloaded model, run the following script:

   ```shell
   python /opt/Export-Deploy/scripts/deploy/nlp/deploy_vllm_triton.py \
       --model_path_id /opt/checkpoints/hf_llama31_8B_mbridge \
       --triton_model_name llama \
       --tensor_parallelism_size 1
   ```

   The following parameters are defined in the ``deploy_vllm_triton.py`` script:

   - ``--model_path_id``: Path of a Megatron-Bridge checkpoint, or Hugging Face model ID or path. (Required)
   - ``--tokenizer``: Tokenizer file if it is not provided in the checkpoint. (Optional)
   - ``--lora_ckpt``: List of LoRA checkpoints in HF format. (Optional, can specify multiple)
   - ``--tensor_parallelism_size``: Number of GPUs to use for tensor parallelism. Default is 1.
   - ``--dtype``: Data type for the model in vLLM. Choices: "auto", "bfloat16", "float16", "float32". Default is "auto".
   - ``--quantization``: Quantization method for vLLM. Choices: "awq", "gptq", "fp8". Default is None.
   - ``--seed``: Random seed for reproducibility. Default is 0.
   - ``--gpu_memory_utilization``: GPU memory utilization percentage for vLLM. Default is 0.9.
   - ``--swap_space``: Size (GiB) of CPU memory per GPU to use as swap space. Default is 4.
   - ``--cpu_offload_gb``: Size (GiB) of CPU memory to use for offloading model weights. Default is 0.
   - ``--enforce_eager``: Whether to enforce eager execution. Default is False.
   - ``--max_seq_len_to_capture``: Maximum sequence length covered by CUDA graphs. Default is 8192.
   - ``--triton_model_name``: Name for the service/model on Triton. (Required)
   - ``--triton_model_version``: Version for the service/model. Default is 1.
   - ``--triton_port``: Port for the Triton server to listen for requests. Default is 8000.
   - ``--triton_http_address``: HTTP address for the Triton server. Default is 0.0.0.0.
   - ``--max_batch_size``: Maximum batch size of the model. Default is 8.
   - ``--debug_mode``: Enable debug/verbose output. Default is False.
   
3. Access the models with a Hugging Face token.

   If you want to run inference using the StarCoder1, StarCoder2, or LLama3 models, you'll need to generate a Hugging Face token that has access to these models. Visit `Hugging Face <https://huggingface.co/>`__ for more information. After you have the token, perform one of the following steps.

   - Log in to Hugging Face:

   ```shell
   huggingface-cli login
   ```

   - Or, set the HF_TOKEN environment variable:

   ```shell
   export HF_TOKEN=your_token_here
   ```

## Supported LLMs

Megatron-Bridge models are supported for export and deployment if they are listed as compatible in the [vLLM supported models list](https://docs.vllm.ai/en/v0.9.2/models/supported_models.html).


## Use NeMo Export and Deploy APIs to Export

Up until now, we have used scripts for exporting and deploying LLM models. However, NeMo's deploy and export modules offer straightforward APIs for deploying models to Triton and exporting Megatron-Bridge checkpoints to vLLM.


### Export Megatron-Bridge LLMs

You can use the APIs in the export module to export a Megatron-Bridge checkpoint to vLLM. The following code example assumes the ``hf_llama31_8B_mbridge`` checkpoint has already been downloaded and generated at the ``/opt/checkpoints/`` path.

```python
from nemo_export.vllm_exporter import vLLMExporter

checkpoint_file = "/opt/checkpoints/hf_llama31_8B_mbridge"

exporter = vLLMExporter()
exporter.export(
    model_path_id=checkpoint_file,
    tensor_parallel_size=1,
)

# The correct argument for output length is 'max_tokens', not 'max_output_len'
output = exporter.forward(
    ["What is the best city in the world?"],
    max_tokens=50,
    top_k=1,
    top_p=0.0,
    temperature=1.0,
)
print("output: ", output)
```

Be sure to check the vLLMExporter class docstrings for details.


## How To Send a Query

### Send a Query using the Script

You can send queries to your deployed Megatron-Bridge LLM using the provided query script. This script allows you to interact with the model via HTTP requests, sending prompts and receiving generated responses directly from the Triton server.

The example below demonstrates how to use the query script to send a prompt to your deployed model. You can customize the request with various parameters to control generation behavior, such as output length, sampling strategy, and more. For a full list of supported parameters, see below.

```shell
python /opt/Export-Deploy/scripts/deploy/nlp/query_vllm.py --url "http://localhost:8000" --model_name llama --prompt "What is the capital of United States?"
```

**Additional parameters:**
- `--prompt_file`: Read prompt from a file instead of the command line
- `--max_tokens`: Maximum number of tokens to generate (default: 16)
- `--min_tokens`: Minimum number of tokens to generate (default: 0)
- `--n_log_probs`: Number of log probabilities to return per output token
- `--n_prompt_log_probs`: Number of log probabilities to return per prompt token
- `--seed`: Random seed for generation
- `--top_k`: Top-k sampling (default: 1)
- `--top_p`: Top-p sampling (default: 0.1)
- `--temperature`: Sampling temperature (default: 1.0)
- `--lora_task_uids`: List of LoRA task UIDs for LoRA-enabled models (use -1 to disable)
- `--init_timeout`: Init timeout for the Triton server in seconds (default: 60.0)


### Send a Query using the NeMo APIs

Please see the below if you would like to use APIs to send a query.

```python
from nemo_deploy.nlp import NemoQueryvLLM

nq = NemoQueryvLLM(url="localhost:8000", model_name="llama")
output = nq.query_llm(
    prompts=["What is the capital of United States? "],
    max_tokens=100,
    top_k=1,
    top_p=0.8,
    temperature=1.0,
)
print("output: ", output)
```
