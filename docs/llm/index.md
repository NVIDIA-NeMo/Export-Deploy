# Deploy Large Language Models

The NeMo Framework offers scripts and APIs to deploy both NeMo and Hugging Face LLMs to the NVIDIA Triton Inference Server. You can export models to TensorRT-LLM or vLLM for optimized inference before deployment. Note that this optimized deployment path supports only selected LLM models. Deployment capabilities for Hugging Face models have been added to the NeMo Framework as part of the NeMo AutoModel features, supporting models in the Hugging Face format.

## Deploy NeMo Models
You can find details about how to deploy NeMo models, including inference optimized models and PyTorch level models (in-framework), how to apply quantization, and how to send queries to deployed models below.


## Deploy Hugging Face Models
You can find details about how to deploy Hugging Face models either in-framework or using TensorRT-LLM, and serve them on the NVIDIA Triton Inference Server below.


## Supported NeMo Checkpoint Formats
The NeMo Framework saves models as nemo files, which include data related to the model, such as weights and configuration files. The format of these .nemo files has evolved over time. The framework supports deploying Megatron Core-based NeMo models using the distributed checkpoint format. If you saved the model using one of the latest NeMo Framework containers, you should not encounter any issues. Otherwise, you may receive an error message regarding the .nemo file format.

NeMo 1.0 checkpoints are either of the *nemo* or *qnemo* type. Both file types are supported for deployment. NeMo 2.0 saves all the model related files into a folder.

The NeMo 1.0 *.nemo* checkpoint and the NeMo 2.0 model folder include the model weights with default precision. It consists of a YAML config file, a folder for model weights, and the tokenizer (if it is not available online). Models are trained and stored in this format, with weight values in FP16 or BF16 precision. Additionally, for .nemo models trained in FP8 precision using [NVIDIA Transformer Engine](https://github.com/NVIDIA/TransformerEngine), it is possible to directly export them to an inference framework that supports FP8. Such models already come with scaling factors for low-precision [GEMMs](https://docs.nvidia.com/deeplearning/performance/dl-performance-matrix-multiplication/index.html) and do not require any extra calibration.

The *qnemo* checkpoint contains the quantized weights and scaling factors. It follows the [TensorRT-LLM checkpoint format](https://nvidia.github.io/TensorRT-LLM/architecture/checkpoint.html) with the addition of a tokenizer. It is derived from a corresponding *nemo* model. For detailed information on how to produce a *qnemo* checkpoint, please refer to [Quantization manual](https://docs.nvidia.com/nemo-framework/user-guide/latest/model-optimization/quantization/quantization.html).

The NeMo 1.0 and NeMo 2.0 checkpoint formats are supported by the TensorRT-LLM and vLLM inference-optimized deployment options. However, in-framework (PyTorch-level models) deployment only supports NeMo 2.0 models.


## Supported GPUs
TensorRT-LLM supports NVIDIA DGX H100 and NVIDIA H100 GPUs based on the NVIDIA Hopper, Ada Lovelace, Ampere, Turing, and Volta architectures. Certain specialized deployment paths, such as FP8 quantized models, require hardware with FP8 data type support, like NVIDIA H100 GPUs.


## Generate a NeMo 2.0 Checkpoint
Please follow the steps below to generate a NeMo 2.0 checkpoint and use it for testing the export and deployment paths for NeMo models.

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

   ```{python}
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

