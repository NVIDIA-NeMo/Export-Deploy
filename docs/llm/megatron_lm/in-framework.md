# Deploy Megatron-LM LLMs with Triton Inference Server

This section provides guidance on deploying Megatron-LM LLMs using the NVIDIA Triton Inference Server. The process closely mirrors the steps outlined in the [Megatron-Bridge documentation](../mbridge/in-framework.md), with only minor differences.

## Quick Example

1. In a terminal, go to the folder where your Megatron-LM checkpoint is located. Pull and run the Docker container image using the command shown below. Change the ``:vr`` tag to the version of the container you want to use:

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
   python /opt/Export-Deploy/scripts/deploy/nlp/deploy_inframework_triton.py --megatron_checkpoint /opt/checkpoints/megatron_lm_ckpt --triton_model_name model --model_format megatron
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
- Set `--model_format megatron` to indicate the model type.
- Set `--model_type` to gpt.


For more detailed instructions and additional information, please consult the [Megatron-Bridge documentation](../mbridge/in-framework.md).