# Deploy Automodel LLMs with vLLM and Triton Inference Server

This section shows how to use scripts and APIs to export a [NeMo AutoModel](https://docs.nvidia.com/nemo/automodel/latest/index.html) LLM to vLLM and deploy it with the NVIDIA Triton Inference Server.

## Quick Example

1. Pull down and run the Docker container image using the command shown below. Change the ``:vr`` tag to the version of the container you want to use:

   ```shell
   docker pull nvcr.io/nvidia/nemo:vr

   docker run --gpus all -it --rm --shm-size=4g -p 8000:8000 \
      -w /opt/Export-Deploy \
      --name nemo-fw \
      nvcr.io/nvidia/nemo:vr
   ```

2. Install vLLM by executing the following command inside the container:

   ```shell
   cd /opt/Export-Deploy
   uv sync --inexact --link-mode symlink --locked --extra vllm $(cat /opt/uv_args.txt)
   ```

3. Run the following deployment script to verify that everything is working correctly. The script exports the Llama NeMo checkpoint to vLLM and subsequently serves it on the Triton server:

   ```shell
   python /opt/Export-Deploy/scripts/deploy/nlp/deploy_vllm_triton.py \
       --model_path_id meta-llama/Llama-3.2-1B \
       --triton_model_name llama   
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


**Note:** The documentation for Automodel LLM deployment using vLLM is almost the same with the one for NeMo 2.0. Please check the [NeMo 2.0 documentation here](../../nemo_2/optimized/vllm.md).