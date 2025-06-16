# **NeMo Export and Deploy**

## Introduction

NVIDIA NeMo Export and Deploy library provides tools and APIs for exporting and deploying NeMo and Hugging Face models to production environments. It supports various deployment paths including TensorRT, TensorRT-LLM, and vLLM deployment through NVIDIA Triton Inference Server.

## Key Features

- Support for Large Language Models (LLMs) and Multimodal Models
- Export NeMo and Hugging Face models to optimized inference formats including TensorRT-LLM and vLLM
- Deploy NeMo and Hugging Face models using Ray Serve or NVIDIA Triton Inference Server
- Export quantized NeMo models (FP8, etc)
- Multi-GPU and distributed inference capabilities
- Multi-instance deployment options

## Key Requirements

- Python 3.10 or above (Recommended: Python 3.12)
- PyTorch 2.5 or above (Recommended: PyTorch 2.6)
- NVIDIA GPU
- TensorRT-LLM and vLLM
- Ray Serve
- NVIDIA Triton Inference Server

## Quick Start

### Using Docker

Export-Deploy provides support for TRT-LLM and vLLM.  

Build a container with TRT-LLM support:

```bash
docker build \
    -f docker/Dockerfile.ci \
    -t export-deploy \
    --build-arg INFERENCE_FRAMEWORK=trtllm \
    .
```

Or, alternatively to build a container with vLLM support, run:

```bash
docker build \
    -f docker/Dockerfile.ci \
    -t export-deploy \
    --build-arg INFERENCE_FRAMEWORK=vllm \
    .
```

Start an interactive terminal inside the container:

```bash
docker run \
    --rm \
    -it \
    --entrypoint bash \
    --workdir /opt/Export-Deploy \
    --shm-size=4g \
    --gpus all \
    -v ${PWD}:/opt/Export-Deploy \
    -v ${PWD}/checkpoints/:/opt/checkpoints/ \
    export-deploy
```

### Export and Deploy LLM Examples

The following examples demonstrate how to export and deploy Large Language Models (LLMs) using NeMo Export and Deploy. These examples cover both Hugging Face and NeMo model formats, showing how to export them to TensorRT-LLM and deploy using NVIDIA Triton Inference Server for high-performance inference.

#### Export Hugging Face Models to TensorRT-LLM and Deploy using Triton Inference Server

Please note that Llama models require special access permissions from Meta. To use Llama models, you must first accept Meta's license agreement and obtain access credentials. For instructions on obtaining access, please refer to the [section on generating NeMo checkpoints](#generate-a-nemo-checkpoint) below.

```python
from nemo_export.tensorrt_llm import TensorRTLLM
from nemo_deploy import DeployPyTriton

# Export model to TensorRT-LLM
exporter = TensorRTLLM(model_dir="/tmp/hf_llama32_1B_hf")
exporter.export_hf_model(
    hf_model_path="/opt/checkpoints/hf_llama32_1B_hf",
    tensor_parallelism_size=1,
)

# Generate output
output = exporter.forward(
    input_texts=["What is the color of a banana?"],
    top_k=1,
    top_p=0.0,
    temperature=1.0,
    max_output_len=20,
)
print("output: ", output)

# Deploy to Triton
nm = DeployPyTriton(model=exporter, triton_model_name="llama", http_port=8000)
nm.deploy()
nm.serve()
```

After running the code above, Triton Inference Server will start and begin serving the model. For instructions on how to query the deployed model and make inference requests, please refer to [Querying Deployed Models](#querying-deployed-models).

#### Export NeMo LLM Models to TensorRT-LLM and Deploy using Triton Inference Server

Before running the example below, ensure you have a NeMo checkpoint file. If you don't have a checkpoint yet, see the [section on generating NeMo checkpoints](#generate-a-nemo-checkpoint) for step-by-step instructions on creating one.

```python
from nemo_export.tensorrt_llm import TensorRTLLM
from nemo_deploy import DeployPyTriton

# Export model to TensorRT-LLM
exporter = TensorRTLLM(model_dir="/tmp/hf_llama32_1B_nemo2")
exporter.export(
    nemo_checkpoint_path="/opt/checkpoints/hf_llama32_1B_nemo2",
    tensor_parallelism_size=1,
)

# Generate output
output = exporter.forward(
    input_texts=["What is the color of a banana?"],
    top_k=1,
    top_p=0.0,
    temperature=1.0,
    max_output_len=20,
)
print("output: ", output)

# Deploy to Triton
nm = DeployPyTriton(model=exporter, triton_model_name="llama", http_port=8000)
nm.deploy()
nm.serve()
```

#### Export NeMo Models vLLM and Deploy using Triton Inference Server

```python
from nemo_export.vllm_exporter import vLLMExporter
from nemo_deploy import DeployPyTriton

# Export model to vLLM
exporter = vLLMExporter()
exporter.export(
    nemo_checkpoint="/opt/checkpoints/hf_llama32_1B_nemo2",
    model_dir="/tmp/hf_llama32_1B_nemo2",
    tensor_parallel_size=1,
)

# Generate output
output = exporter.forward(
    input_texts=["What is the color of a banana?"],
    top_k=1,
    top_p=0.0,
    temperature=1.0,
    max_output_len=20,
)
print("output: ", output)

# Deploy to Triton
nm = DeployPyTriton(model=exporter, triton_model_name="llama", http_port=8000)
nm.deploy()
nm.serve()
```

#### Deploy NeMo Models using Triton Inference Server

You can also deploy NeMo and Hugging Face models directly using Triton Inference Server without exporting to inference optimized libraries like TensorRT-LLM or vLLM. This provides a simpler deployment path while still leveraging Triton's scalable serving capabilities.

```python
from nemo_deploy import DeployPyTriton
from nemo_deploy.nlp.megatronllm_deployable import MegatronLLMDeployableNemo2

model = MegatronLLMDeployableNemo2(
    nemo_checkpoint_filepath="/opt/checkpoints/hf_llama32_1B_nemo2",
    num_devices=1,
    num_nodes=1,
)

# Deploy to Triton
nm = DeployPyTriton(model=model, triton_model_name="llama", http_port=8000)
nm.deploy()
nm.serve()
```

#### Deploy Hugging Face Models using Triton Inference Server

You can also deploy NeMo and Hugging Face models directly using Triton Inference Server without exporting to inference optimized libraries like TensorRT-LLM or vLLM. This provides a simpler deployment path while still leveraging Triton's scalable serving capabilities.

```python
from nemo_deploy import DeployPyTriton
from nemo_deploy.nlp.hf_deployable import HuggingFaceLLMDeploy

model = HuggingFaceLLMDeploy(
    hf_model_id_path="hf://meta-llama/Llama-3.2-1B",
    device_map="auto",
)

# Deploy to Triton
nm = DeployPyTriton(model=model, triton_model_name="llama", http_port=8000)
nm.deploy()
nm.serve()
```

### Export and Deploy Multimodal Examples

#### Export NeMo Multimodal Models to TensorRT-LLM and Deploy using Triton Inference Server

```python
from nemo_deploy import DeployPyTriton
from nemo_export.tensorrt_mm_exporter import TensorRTMMExporter

# Export multimodal model
exporter = TensorRTMMExporter(model_dir="/path/to/export/dir", modality="vision")
exporter.export(
    visual_checkpoint_path="/path/to/model.nemo",
    model_type="neva",
    llm_model_type="llama",
    tensor_parallel_size=1,
)

# Deploy to Triton
nm = DeployPyTriton(model=exporter, triton_model_name="neva", port=8000)
nm.deploy()
nm.serve()
```

### Querying Deployed Models

#### Query LLM Model

```python
from nemo_deploy.nlp import NemoQueryLLM

nq = NemoQueryLLM(url="localhost:8000", model_name="llama")
output = nq.query_llm(
    prompts=["What is the capital of France?"],
    max_output_len=100,
)
print(output)
```

#### Query Multimodal Model

```python
from nemo_deploy.multimodal import NemoQueryMultimodal

nq = NemoQueryMultimodal(url="localhost:8000", model_name="neva", model_type="neva")
output = nq.query(
    input_text="What is in this image?",
    input_media="/path/to/image.jpg",
    max_output_len=30,
)
print(output)
```

## Generate a NeMo Checkpoint

In order to run examples with NeMo models, a NeMo checkpoint is required. Please follow the steps below to generate a NeMo checkpoint.

1. To access the Llama models, please visit the [Llama 3.2 Hugging Face page](https://huggingface.co/meta-llama/Llama-3.2-1B).

2. Pull down and run the NeMo Framework Docker container image using the command shown below:

   ```shell
   docker pull nvcr.io/nvidia/nemo:25.04

   docker run --gpus all -it --rm --shm-size=4g -p 8000:8000 -v ${PWD}/:/opt/checkpoints/ -w /opt/NeMo nvcr.io/nvidia/nemo:25.04
   ```

3. Run the following command in the terminal and enter your Hugging Face access token to log in to Hugging Face:

   ```shell
   huggingface-cli login
   ```

4. Run the following Python code to generate the NeMo 2.0 checkpoint:

   ```python
   from nemo.collections.llm import import_ckpt
   from nemo.collections.llm.gpt.model.llama import Llama32Config1B, LlamaModel
   from pathlib import Path

   if __name__ == "__main__":
       import_ckpt(
           model=LlamaModel(Llama32Config1B()),
           source="hf://meta-llama/Llama-3.2-1B",
           output_path=Path("/opt/checkpoints/hf_llama32_1B_nemo2"),
       )

## Installation

For Export-Deploy without Mcore, TranformerEngine, TRT-LLM and vLLM support, just run:

```bash
pip install nemo-export-deploy
pip install nemo-run # Needs to be installed additionally
```

### Installation with Megatron-Core and TransformerEngine support

Prerequisites for pip installation:

A compatible C++ compiler
CUDA Toolkit with cuDNN and NVCC (NVIDIA CUDA Compiler) installed

```bash
git clone https://github.com/NVIDIA-NeMo/Export-Deploy
cd Export-Deploy

pip install torch pybind11 setuptools
pip install -e --build-isolation '.[te]'
```

### Installation with TRT-LLM or vLLM support

Additionally to Megatron-Core/TransformerEngine, users may also add TRT-LLM or vLLM support. Note that TRT-LLM and vLLM are mutually exclusive, attempting to install both together will likely result in an error.

For TRT-LLM, make sure to `sudo apt install libopenmpi-dev` before.

```bash
git clone https://github.com/NVIDIA-NeMo/Export-Deploy
cd Export-Deploy
pip install -e '.[trtllm]'
```

For vLLM:

```bash
git clone https://github.com/NVIDIA-NeMo/Export-Deploy
cd Export-Deploy
pip install -e '.[vllm]'
```

## Documentation

For detailed documentation, please refer to:

- [Export-Deploy User Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/overview.html)

## Contributing

We welcome contributions to NeMo RL! Please see our [Contributing Guidelines](CONTRIBUTING.md) for more information on how to get involved.

## License

- [NeMo GitHub Apache 2.0 license](https://github.com/NVIDIA/NeMo?tab=Apache-2.0-1-ov-file#readme)
- NeMo is licensed under the [NVIDIA AI PRODUCT AGREEMENT](https://www.nvidia.com/en-us/data-center/products/nvidia-ai-enterprise/eula/)
