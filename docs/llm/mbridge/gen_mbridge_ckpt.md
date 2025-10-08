# Generate A Megatron-Bridge Checkpoint

To run the code examples, you will need a [Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge) checkpoint. Follow the steps below to generate a Megatron-Bridge checkpoint, which you can then use to test the export and deployment workflows for the models.

1. To access the Llama models, please visit the [Llama 3.1 8B Hugging Face page](https://huggingface.co/meta-llama/Llama-3.1-8B).

2. Pull down and run the Docker container image using the command shown below. Change the ``:vr`` tag to the version of the container you want to use:

   ```shell
   docker pull nvcr.io/nvidia/nemo:vr

   docker run --gpus all -it --rm --shm-size=4g -p 8000:8000 -v ${PWD}/:/opt/checkpoints/ -w /opt/NeMo nvcr.io/nvidia/nemo:vr
   ```
   
3. Run the following command in the terminal and enter your Hugging Face access token to log in to Hugging Face:

   ```shell
   huggingface-cli login
   ```
   
4. Run the following Python code to generate the NeMo 2.0 checkpoint:

   ```python
   from megatron.bridge import AutoBridge

   if __name__ == "__main__":
      AutoBridge.import_ckpt(
         "meta-llama/Llama-3.1-8B",
         "/opt/checkpoints/hf_llama31_8B_mbridge",        
      )
   ```



