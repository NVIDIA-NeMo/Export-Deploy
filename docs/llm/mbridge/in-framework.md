# Deploy Megatron-Bridge LLMs with Triton Inference Server

This section explains how to deploy [Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge) LLMs with the NVIDIA Triton Inference Server.

## Quick Example

1. Follow the steps on the [Generate A Megatron-Bridge Checkpoint page](gen_mbridge_ckpt.md) to generate a Megatron-Bridge Llama checkpoint.

2. In a terminal, go to the folder where the ``hf_llama31_8B_mbridge`` checkpoint is located. Pull and run the Docker container image using the command shown below. Change the ``:vr`` tag to the version of the container you want to use:

   ```shell
   docker pull nvcr.io/nvidia/nemo:vr

   docker run --gpus all -it --rm --shm-size=4g -p 8000:8000 \
       -v ${PWD}/:/opt/checkpoints/ \
       -w /opt/Export-Deploy \
       --name nemo-fw \
       nvcr.io/nvidia/nemo:vr
   ```

3. Using a Megatron-Bridge model, run the following deployment script to verify that everything is working correctly. The script directly serves the Megatron-Bridge model on the Triton server:

   ```shell
   python /opt/Export-Deploy/scripts/deploy/nlp/deploy_inframework_triton.py --megatron_checkpoint /opt/checkpoints/hf_llama31_8B_mbridge --triton_model_name llama
   ```

4. If the test yields a shared memory-related error, increase the shared memory size using ``--shm-size`` (for example, gradually by 50%).

5. In a separate terminal, access the running container as follows:

   ```shell
   docker exec -it nemo-fw bash
   ```

6. To send a query to the Triton server, run the following script:

   ```shell
   python /opt/Export-Deploy/scripts/deploy/nlp/query_inframework.py -mn llama -p "What is the color of a banana?" -mol 5
   ```

## Use a Script to Deploy Megatron-Bridge LLMs on a Triton Server

You can deploy an LLM from a Megatron-Bridge checkpoint on Triton using the provided script.

### Deploy a Megatron-Bridge LLM Model

The following instructions are very similar to those for [deploying NeMo 2.0 models](../nemo_2/in-framework.md), with only a few key differences specific to Megatron-Bridge highlighted below.

- Use the `--megatron_checkpoint` argument to specify your Megatron-Bridge checkpoint file.


Executing the script will directly deploy the Megatron-Bridge LLM model and start the service on Triton.

1. Start the container using the steps described in the **Quick Example** section.

2. To begin serving the downloaded model, run the following script:

   ```shell
   python /opt/Export-Deploy/scripts/deploy/nlp/deploy_inframework_triton.py --megatron_checkpoint /opt/checkpoints/hf_llama31_8B_mbridge --triton_model_name llama
   ```

   The following parameters are defined in the ``deploy_inframework_triton.py`` script:

   - ``-nc``, ``--megatron_checkpoint``: Path to the Megatron-Bridge checkpoint file to deploy. (Required)
   - ``-tmn``, ``--triton_model_name``: Name to register the model under in Triton. (Required)
   - ``-tmv``, ``--triton_model_version``: Version number for the model in Triton. Default: 1
   - ``-sp``, ``--server_port``: Port for the REST server to listen for requests. Default: 8080
   - ``-sa``, ``--server_address``: HTTP address for the REST server. Default: 0.0.0.0
   - ``-tp``, ``--triton_port``: Port for the Triton server to listen for requests. Default: 8000
   - ``-tha``, ``--triton_http_address``: HTTP address for the Triton server. Default: 0.0.0.0
   - ``-ng``, ``--num_gpus``: Number of GPUs for the deployment. (Optional)
   - ``-nn``, ``--num_nodes``: Number of nodes for the deployment. (Optional)
   - ``-tps``, ``--tensor_parallelism_size``: Tensor parallelism size. Default: 1
   - ``-pps``, ``--pipeline_parallelism_size``: Pipeline parallelism size. Default: 1
   - ``-nlfps``, ``--num_layers_in_first_pipeline_stage``: Number of layers in the first pipeline stage. (Optional)
   - ``-nllps``, ``--num_layers_in_last_pipeline_stage``: Number of layers in the last pipeline stage. (Optional)
   - ``-cps``, ``--context_parallel_size``: Context parallelism size. Default: 1
   - ``-emps``, ``--expert_model_parallel_size``: Distributes MoE Experts across sub data parallel dimension. Default: 1
   - ``-eps``, ``--account_for_embedding_in_pipeline_split``: Account for embedding in the pipeline split. (Flag; set to enable)
   - ``-alps``, ``--account_for_loss_in_pipeline_split``: Account for loss in the pipeline split. (Flag; set to enable)
   - ``-mbs``, ``--max_batch_size``: Max batch size of the model. Default: 8
   - ``-dm``, ``--debug_mode``: Enable debug mode. (Flag; set to enable)
   - ``-efd``, ``--enable_flash_decode``: Enable flash decoding. (Flag; set to enable)
   - ``-cg``, ``--enable_cuda_graphs``: Enable CUDA graphs. (Flag; set to enable)
   - ``-lc``, ``--legacy_ckpt``: Load checkpoint saved with TE < 1.14. (Flag; set to enable)
   - ``-imsl``, ``--inference_max_seq_length``: Max sequence length for inference. Default: 4096
   - ``-mb``, ``--micro_batch_size``: Micro batch size for model execution. (Optional)

   *Note: Some parameters may be ignored or have no effect depending on the model and deployment environment. Refer to the script's help message for the most up-to-date list.*

3. To deploy a different model, just change the ``--megatron_checkpoint`` argument in the script.



## How To Send a Query

### Send a Query using the Script
You can send queries to your deployed Megatron-Bridge LLM using the provided query script. This script allows you to interact with the model via HTTP requests, sending prompts and receiving generated responses directly from the Triton server.

The example below demonstrates how to use the query script to send a prompt to your deployed model. You can customize the request with various parameters to control generation behavior, such as output length, sampling strategy, and more. For a full list of supported parameters, see below.


```shell
python /opt/Export-Deploy/scripts/deploy/nlp/query_inframework.py -mn llama -p "What is the capital of United States?"
```

**All Parameters:**
- `-u`, `--url`: URL for the Triton server (default: 0.0.0.0)
- `-mn`, `--model_name`: Name of the Triton model (required)
- `-p`, `--prompt`: Prompt text (required, mutually exclusive with prompt_file)
- `-pf`, `--prompt_file`: File to read the prompt from (required, mutually exclusive with prompt)
- `-mol`, `--max_output_len`: Max output token length (default: 128)
- `-tk`, `--top_k`: Top-k sampling (default: 1)
- `-tpp`, `--top_p`: Top-p sampling (default: 0.0)
- `-t`, `--temperature`: Sampling temperature (default: 1.0)
- `-it`, `--init_timeout`: Init timeout for the Triton server (default: 60.0)
- `-clp`, `--compute_logprob`: Returns log probabilities (flag)


### Send a Query using the Export-Deploy APIs

Please see the below if you would like to use APIs to send a query.

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
