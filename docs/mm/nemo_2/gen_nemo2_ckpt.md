# Generate A NeMo 2.0 Checkpoint

To run the code examples, you will need a NeMo 2.0 checkpoint. Follow the steps below to generate a NeMo 2.0 checkpoint, which you can then use to test the export and deployment workflows for NeMo 2.0 models.

## Setup

1. Pull down and run [NeMo Framework](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo) the Docker container image using the command shown below. Change the ``:vr`` tag to the version of the container you want to use:

   ```shell
   docker pull nvcr.io/nvidia/nemo:vr

   docker run --gpus all -it --rm --shm-size=4g -p 8000:8000 -v ${PWD}/:/opt/checkpoints/ -w /opt/NeMo nvcr.io/nvidia/nemo:vr
   ```
   
2. Run the following command in the terminal and enter your Hugging Face access token to log in to Hugging Face:

   ```shell
   huggingface-cli login
   ```

## Generate Qwen VL Checkpoint (for In-Framework Deployment)

This checkpoint is used for in-framework deployment examples.

3. Run the following Python code to generate the NeMo 2.0 checkpoint:

   ```python
   from nemo.collections.llm import import_ckpt
   from nemo.collections import vlm
   from pathlib import Path

   if __name__ == '__main__':
      # Specify the Hugging Face model ID
      hf_model_id = "Qwen/Qwen2.5-VL-3B-Instruct"

      # Import the model and convert to NeMo 2.0 format
      import_ckpt(
         model=vlm.Qwen2VLModel(vlm.Qwen25VLConfig3B(), model_version='qwen25-vl')
         source=f"hf://{hf_model_id}",  # Hugging Face model source
         output_path=Path('/opt/checkpoints/qwen25_vl_3b')
      )
   ```

## Generate Llama 3.2-Vision Checkpoint (for TensorRT-LLM Deployment)

This checkpoint is used for optimized TensorRT-LLM deployment examples.

3. Run the following Python code to generate the NeMo 2.0 checkpoint:

   ```python
   from nemo.collections.llm import import_ckpt
   from nemo.collections import vlm
   from pathlib import Path

   if __name__ == '__main__':
      # Specify the Hugging Face model ID
      hf_model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"

      # Import the model and convert to NeMo 2.0 format
      import_ckpt(
         model=vlm.MLlamaModel(vlm.MLlamaConfig11BInstruct()),
         source=f"hf://{hf_model_id}",  # Hugging Face model source
         output_path=Path('/opt/checkpoints/hf_mllama_11b_nemo')
      )
   ```



