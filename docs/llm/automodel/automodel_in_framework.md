# Deploy NeMo AutoModel LLM Models in the Framework

This section demonstrates how to deploy NeMo AutoModel LLM Models within the framework (referred to as 'In-Framework for AutoModel LLM') using the NVIDIA Triton Inference Server. NeMo AutoModel workflows generate Hugging Face compatible checkpoints that provide a simplified interface for working with pre-trained language models. These checkpoints maintain high performance during inference while offering enhanced configurability through the Hugging Face ecosystem.


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
       nvcr.io/nvidia/nemo:vr
   ``` 

3. Log in to Hugging Face with your access token:

   ```shell
   huggingface-cli login
   ```

4. Deploy the model to Triton:

   ```python
   python scripts/deploy/nlp/deploy_inframework_hf_triton.py \
      --hf_model_id_path meta-llama/Llama-3.2-1B \
      --triton_model_name llama
   ```
   
   **Note:** If you encounter shared memory errors, increase ``--shm-size`` gradually by 50%.

5. In a new terminal, get the container ID:

   ```shell
   docker ps
   ```

6. Access the container:

   ```shell
   docker exec -it <container_id> bash
   ```

7. Test the deployed model:

   ```shell
   python scripts/deploy/nlp/query_inframework_hf.py \
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
   python scripts/deploy/nlp/deploy_inframework_hf_triton.py \
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

   Option 2 - Set environment variable:

   ```shell
   export HF_TOKEN=your_token_here
   ```

### Multi-GPU Deployment

For multi-GPU inference:

1. Use ``--tp_plan`` instead of ``--device_map`` (they are mutually exclusive)
2. For distributed inference across GPUs, use ``torchrun``. Example with 2 GPUs:

   ```shell
   torchrun --standalone --nnodes=1 --nproc_per_node=2 \
      scripts/deploy/nlp/deploy_inframework_hf_triton.py \
      --hf_model_id_path meta-llama/Llama-3.2-1B \
      --triton_model_name llama \
      --tp_plan auto
   ```

For more information:
   - Device mapping: [Hugging Face Loading Big Models docs](https://huggingface.co/docs/accelerate/main/concept_guides/big_model_inference)
   - Tensor parallelism: [Hugging Face Multi-GPU Inference docs](https://huggingface.co/docs/transformers/v4.47.0/en/perf_infer_gpu_multi)
