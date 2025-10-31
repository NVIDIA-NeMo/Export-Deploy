# Deploy NeMo 2.0 Multimodal Models with Triton Inference Server

This section explains how to deploy [NeMo 2.0](https://github.com/NVIDIA-NeMo/NeMo) multimodal models with the NVIDIA Triton Inference Server.

## Quick Example

1. Follow the steps on the [Generate A NeMo 2.0 Checkpoint page](gen_nemo2_ckpt.md) to generate a NeMo 2.0 multimodal checkpoint.

2. In a terminal, go to the folder where the ``qwen2_vl_3b`` is located. Pull and run the Docker container image using the command shown below. Change the ``:vr`` tag to the version of the container you want to use:

   ```shell
   docker pull nvcr.io/nvidia/nemo:vr

   docker run --gpus all -it --rm --shm-size=4g -p 8000:8000 \
       -v ${PWD}/:/opt/checkpoints/ \
       -w /opt/Export-Deploy \
       --name nemo-fw \
       nvcr.io/nvidia/nemo:vr
   ```

3. Using a NeMo 2.0 multimodal model, run the following deployment script to verify that everything is working correctly. The script directly serves the NeMo 2.0 model on the Triton server:

   ```shell
   python /opt/Export-Deploy/scripts/deploy/multimodal/deploy_inframework_triton.py --nemo_checkpoint /opt/checkpoints/qwen2_vl_3b --triton_model_name qwen
   ```

4. If the test yields a shared memory-related error, increase the shared memory size using ``--shm-size`` (for example, gradually by 50%).

5. In a separate terminal, access the running container as follows:

   ```shell
   docker exec -it nemo-fw bash
   ```

6. To send a query to the Triton server, run the following script with an image:

   ```shell
   python /opt/Export-Deploy/scripts/deploy/multimodal/query_inframework.py \
       --model_name qwen \
       --prompt "Describe this image" \
       --image /path/to/image.jpg \
       --max_output_len 100
   ```

## Use a Script to Deploy NeMo 2.0 Multimodal Models on a Triton Inference Server

You can deploy a multimodal model from a NeMo checkpoint on Triton using the provided script.

### Deploy a NeMo Multimodal Model

Executing the script will directly deploy the NeMo 2.0 multimodal model and start the service on Triton.

1. Start the container using the steps described in the **Quick Example** section.

2. To begin serving the downloaded model, run the following script:

   ```shell
   python /opt/Export-Deploy/scripts/deploy/multimodal/deploy_inframework_triton.py --nemo_checkpoint /opt/checkpoints/qwen2_vl_3b --triton_model_name qwen
   ```

   The following parameters are defined in the ``deploy_inframework_triton.py`` script:

   - ``-nc``, ``--nemo_checkpoint``: Path to the NeMo 2.0 checkpoint file to deploy. (Required)
   - ``-tmn``, ``--triton_model_name``: Name to register the model under in Triton. (Required)
   - ``-tmv``, ``--triton_model_version``: Version number for the model in Triton. Default: 1
   - ``-sp``, ``--server_port``: Port for the REST server to listen for requests. Default: 8080
   - ``-sa``, ``--server_address``: HTTP address for the REST server. Default: 0.0.0.0
   - ``-trp``, ``--triton_port``: Port for the Triton server to listen for requests. Default: 8000
   - ``-tha``, ``--triton_http_address``: HTTP address for the Triton server. Default: 0.0.0.0
   - ``-tps``, ``--tensor_parallel_size``: Tensor parallelism size. Default: 1
   - ``-pps``, ``--pipeline_parallel_size``: Pipeline parallelism size. Default: 1
   - ``-mbs``, ``--max_batch_size``: Max batch size of the model. Default: 4
   - ``-dm``, ``--debug_mode``: Enable debug mode. (Flag; set to enable)
   - ``-pd``, ``--params_dtype``: Data type for model parameters. Choices: float16, bfloat16, float32. Default: bfloat16
   - ``-ibts``, ``--inference_batch_times_seqlen_threshold``: Inference batch times sequence length threshold. Default: 1000

   *Note: Some parameters may be ignored or have no effect depending on the model and deployment environment. Refer to the script's help message for the most up-to-date list.*

3. To deploy a different model, just change the ``--nemo_checkpoint`` argument in the script.


## How To Send a Query

You can send queries to the Triton Inference Server using either the provided script or the available APIs.

### Send a Query using the Script
This script allows you to interact with the multimodal model via HTTP requests, sending prompts and images and receiving generated responses directly from the Triton server.

The example below demonstrates how to use the query script to send a prompt and image to your deployed model. You can customize the request with various parameters to control generation behavior, such as output length, sampling strategy, and more. For a full list of supported parameters, see below.


```shell
python /opt/Export-Deploy/scripts/deploy/multimodal/query_inframework.py \
    --model_name qwen \
    --processor_name Qwen/Qwen2.5-VL-3B-Instruct \
    --prompt "What is in this image?" \
    --image /path/to/image.jpg \
    --max_output_len 100
```

**All Parameters:**
- `-u`, `--url`: URL for the Triton server (default: 0.0.0.0)
- `-mn`, `--model_name`: Name of the Triton model (required)
- `-pn`, `--processor_name`: Processor name for qwen-vl models (default: Qwen/Qwen2.5-VL-7B-Instruct)
- `-p`, `--prompt`: Prompt text (mutually exclusive with --prompt_file; required if --prompt_file not given)
- `-pf`, `--prompt_file`: File to read the prompt from (mutually exclusive with --prompt; required if --prompt not given)
- `-i`, `--image`: Path or URL to input image file (required)
- `-mol`, `--max_output_len`: Max output token length (default: 50)
- `-mbs`, `--max_batch_size`: Max batch size for inference (default: 4)
- `-tk`, `--top_k`: Top-k sampling (default: 1)
- `-tpp`, `--top_p`: Top-p sampling (default: 0.0)
- `-t`, `--temperature`: Sampling temperature (default: 1.0)
- `-rs`, `--random_seed`: Random seed for generation (optional)
- `-it`, `--init_timeout`: Init timeout for the Triton server in seconds (default: 60.0)


### Send a Query using the NeMo APIs

Please see the below if you would like to use APIs to send a query.

```python
from nemo_deploy.multimodal import NemoQueryMultimodalPytorch
from PIL import Image

nq = NemoQueryMultimodalPytorch(url="localhost:8000", model_name="qwen")
output = nq.query_multimodal(
    prompts=["What is in this image?"],
    images=[Image.open("/path/to/image.jpg")],
    max_length=100,
    top_k=1,
    top_p=0.0,
    temperature=1.0,
)
print(output)
```