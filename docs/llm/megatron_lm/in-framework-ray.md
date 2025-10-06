# Deploy Megatron-LM LLM using Ray Serve

This section provides guidance on deploying Megatron-LM LLMs using the Ray Serve. The process closely mirrors the steps outlined in the [Megatron-Bridge documentation](../mbridge/in-framework.md), with only minor differences.

## Quick Example

1. In a terminal, go to the folder where your Megatron-LM checkpoint is located. Pull and run the Docker container image using the command shown below. Change the ``:vr`` tag to the version of the container you want to use:

   ```shell
   docker pull nvcr.io/nvidia/nemo:vr

   docker run --gpus all -it --rm \
       --shm-size=4g \
       -p 1024:1024 \
       -v ${PWD}/:/opt/checkpoints/ \
       -w /opt/Export-Deploy \
       --name nemo-fw \
       nvcr.io/nvidia/nemo:vr
   ``` 

3. Deploy the model to Ray:

   ```shell
   python /opt/Export-Deploy/scripts/deploy/nlp/deploy_ray_inframework.py \
      --megatron_checkpoint /opt/checkpoints/checkpoints/llama3_145m-mlm_saved-distckpt/ \
      --model_id llama \
      --tensor_model_parallel_size 2 \
      --pipeline_model_parallel_size 2 \
      --num_gpus 4 \
      --model_type gpt
   ```

Notes:
- Use `--model_type gpt` for MegatronLM GPT-style checkpoints.
- Parallelism settings must be compatible with available GPUs (see Configure Model Parallelism).

4. In a separate terminal, access the running container as follows:

   ```shell
   docker exec -it nemo-fw bash
   ```

5. Test the deployed model:

   ```shell
   python scripts/deploy/nlp/query_ray_deployment.py \
      --model_id llama \
      --host 0.0.0.0 \
      --port 1024
   ```

For more detailed instructions and additional information, please consult the [Megatron-Bridge documentation](../mbridge/in-framework.md).