# Generate A NeMo 2.0 Checkpoint

To run the code examples, you will need a NeMo 2.0 checkpoint. Follow the steps below to generate a NeMo 2.0 checkpoint, which you can then use to test the export and deployment workflows for NeMo models.

1. To access the Llama models, please visit the [Llama 3.1 Hugging Face page](https://huggingface.co/meta-llama/Llama-3.1-8B).

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
   from nemo.collections.llm import import_ckpt
   from nemo.collections.llm.gpt.model.llama import Llama31Config8B, LlamaModel
   from pathlib import Path

   if __name__ == "__main__":
       import_ckpt(
           model=LlamaModel(Llama31Config8B(seq_length=16384)),
           source='hf://meta-llama/Llama-3.1-8B',
           output_path=Path('/opt/checkpoints/hf_llama31_8B_nemo2.nemo')
       )
   ```



