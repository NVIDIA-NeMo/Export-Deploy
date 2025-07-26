<div align="center">

# NeMo Export-Deploy

</div>

<div align="center">

<!-- [![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0) -->
[![codecov](https://codecov.io/github/NVIDIA-NeMo/Export-Deploy/graph/badge.svg?token=4NMKZVOW2Z)](https://codecov.io/github/NVIDIA-NeMo/Export-Deploy)
[![CI/CD](https://img.shields.io/github/actions/workflow/status/NVIDIA-NeMo/Export-Deploy/cicd-main.yml?branch=main)](https://github.com/NVIDIA-NeMo/Export-Deploy/actions/workflows/cicd-main.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![GitHub Stars](https://img.shields.io/github/stars/NVIDIA-NeMo/Export-Deploy.svg?style=social&label=Star)](https://github.com/NVIDIA-NeMo/Export-Deploy/stargazers/)

<!-- **Library with tooling and APIs for exporting and deploying NeMo and Hugging Face models with support of backends like  TensorRT, TensorRT-LLM and vLLM through NVIDIA Triton Inference Server.** -->

[![📖 Documentation](https://img.shields.io/badge/docs-nvidia-informational?logo=book)](https://docs.nvidia.com/nemo/export-deploy/latest/index.html)
[![🔧 Installation](https://img.shields.io/badge/install-guide-blue?logo=terminal)](https://github.com/NVIDIA-NeMo/Export-Deploy?tab=readme-ov-file#-install)
[![🚀 Quick start](https://img.shields.io/badge/quick%20start-guide-success?logo=rocket)](https://github.com/NVIDIA-NeMo/Export-Deploy?tab=readme-ov-file#-get-started-quickly)
[![🤝 Contributing](https://img.shields.io/badge/contributing-guide-yellow?logo=github)](https://github.com/NVIDIA-NeMo/Export-Deploy/blob/main/CONTRIBUTING.md)

</div>

NeMo Framework is NVIDIA's GPU accelerated, end-to-end training framework for large language models (LLMs), multi-modal models and speech models. It enables seamless scaling of training (both pretraining and post-training) workloads from single GPU to thousand-node clusters for both 🤗Hugging Face/PyTorch and Megatron models. It includes a suite of libraries and recipe collections to help users train models from end to end. The Export-Deploy library ("NeMo Export-Deploy") provides tools and APIs for exporting and deploying NeMo and 🤗Hugging Face models to production environments. It supports various deployment paths including TensorRT, TensorRT-LLM, and vLLM deployment through NVIDIA Triton Inference Server.

## 🚀 Key Features

- Support for Large Language Models (LLMs) and Multimodal Models
- Export NeMo and Hugging Face models to optimized inference formats including TensorRT-LLM and vLLM
- Deploy NeMo and Hugging Face models using Ray Serve or NVIDIA Triton Inference Server
- Export quantized NeMo models (FP8, etc)
- Multi-GPU and distributed inference capabilities
- Multi-instance deployment options

## 🔧 Install

For quick exploration of NeMo Export-Deploy, we recommend installing our pip package:

```bash
pip install nemo-export-deploy
```

This installation comes without extra dependencies like [TransformerEngine](https://github.com/NVIDIA/TransformerEngine/), [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) or [vLLM](https://github.com/vllm-project/vllm). The installation serves for navigating around and for exploring the project.

For a feature-complete install, please refer to the following sections.

### Use NeMo-FW Container

Best experience, highest performance and full feature support is guaranteed by the [NeMo Framework container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/nemo/tags). Please fetch the most recent `$TAG` and run the following command to start a container:

```bash
docker run --rm -it -w /workdir -v $(pwd):/workdir \
  --entrypoint bash \
  --gpus all \
  nvcr.io/nvidia/nemo:${TAG}
```

#### Install TRT-LLM (or vLLM)

Starting with version 25.07, the NeMo FW container no longer includes TRT-LLM and vLLM pre-installed. Please run the following command inside the container:

For TRT-LLM:

```bash
cd /opt/Export-Deploy
uv sync --link-mode symlink --locked --extra trtllm $(cat /opt/uv_args.txt)
```

For vLLM:

```bash
cd /opt/Export-Deploy
uv sync --link-mode symlink --locked --extra vllm $(cat /opt/uv_args.txt)
```

### Build with Dockerfile

For containerized development, use our Dockerfile for building your own container. There are three flavors: `INFERENCE_FRAMEWORK=inframework`, `INFERENCE_FRAMEWORK=trtllm` and `INFERENCE_FRAMEWORK=vllm`:

```bash
docker build \
    -f docker/Dockerfile.ci \
    -t nemo-export-deploy \
    --build-arg INFERENCE_FRAMEWORK=$INFERENCE_FRAMEWORK \
    .
```

Start your container:

```bash
docker run --rm -it -w /workdir -v $(pwd):/workdir \
  --entrypoint bash \
  --gpus all \
  nemo-export-deploy
```

### Install from Source

For complete feature coverage, we recommend to install [TransformerEngine](https://github.com/NVIDIA/TransformerEngine/?tab=readme-ov-file#pip-installation) and additionally either [TensorRT-LLM](https://nvidia.github.io/TensorRT-LLM/0.20.0/installation/linux.html) or [vLLM](https://docs.vllm.ai/en/latest/getting_started/installation/gpu.html#pre-built-wheels).

#### Recommended Requirements

- Python 3.12
- PyTorch 2.7
- CUDA 12.8
- Ubuntu 24.04

#### Install TransformerEngine + InFramework

For highly optimized TransformerEngine path with TRT-LLM backend, please make sure to install the following prerequisites first:

```bash
pip install torch==2.7.0 setuptools pybind11 wheel_stub  # Required for TE
```

Now proceed with the main installation:

```bash
git clone https://github.com/NVIDIA-NeMo/Export-Deploy
cd Export-Deploy/
pip install --no-build-isolation .[te,inframework]
```

#### Install TransformerEngine + TRT-LLM

For highly optimized TransformerEngine path with TRT-LLM backend, please make sure to install the following prerequisites first:

```bash
sudo apt-get -y install libopenmpi-dev  # Required for TRT-LLM
pip install torch==2.7.0 setuptools pybind11 wheel_stub  # Required for TE
```

Now proceed with the main installation:

```bash
pip install --no-build-isolation .[te,trtllm]
```

#### Install TransformerEngine + vLLM

For highly optimized TransformerEngine path with TRT-LLM backend, please make sure to install the following prerequisites first:

```bash
pip install torch==2.7.0 setuptools pybind11 wheel_stub  # Required for TE
```

Now proceed with the main installation:

```bash
pip install --no-build-isolation .[te,vllm]
```

## 🚀 Get Started Quickly

The following steps are based on a self-built [container](#build-with-dockerfile).

### Generate a NeMo Checkpoint

In order to run examples with NeMo models, a NeMo checkpoint is required. Please follow the steps below to generate a NeMo checkpoint.

1. To access the Llama models, please visit the [Llama 3.2 Hugging Face page](https://huggingface.co/meta-llama/Llama-3.2-1B).

2. Pull down and run the NeMo Framework Docker container image using the command shown below:

   ```shell
   docker run --gpus all -it --rm -p 8000:8000 \
    --entrypoint bash \
    --workdir /opt/Export-Deploy \
    --shm-size=4g \
    --gpus all \
    -v ${PWD}:/opt/Export-Deploy \
    nemo-export-deploy
   ```

3. Run the following command in the terminal and enter your Hugging Face access token to log in to Hugging Face:

   ```shell
   huggingface-cli login
   ```

4. Run the following Python code to generate the NeMo 2.0 checkpoint:

   ```shell
   python scripts/export/export_hf_to_nemo2.py \
    --hf_model meta-llama/Llama-3.2-1B \
    --output_path /opt/checkpoints/hf_llama32_1B_nemo2 \
    --config Llama32Config1B
   ```

## 🚀 Export and Deploy Examples

The following examples demonstrate how to export and deploy Large Language Models (LLMs) using NeMo Export-Deploy. These examples cover both Hugging Face and NeMo model formats, showing how to export them to TensorRT-LLM and deploy using NVIDIA Triton Inference Server for high-performance inference.

### Export and Deploy Hugging Face Models to TensorRT-LLM and Triton Inference Server

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

After running the code above, Triton Inference Server will start and begin serving the model. For instructions on how to query the deployed model and make inference requests, please refer to [Query Deployed Models](#-query-deployed-models).

### Export and Deploy NeMo LLM Models to TensorRT-LLM and Triton Inference Server

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

### Export and Deploy NeMo Models to vLLM and Triton Inference Server

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

### Deploy NeMo Models Directly with Triton Inference Server

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

### Deploy Hugging Face Models Directly with Triton Inference Server

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

### Export and Deploy Multimodal Models to TensorRT-LLM and Triton Inference Server

```python
from nemo_deploy import DeployPyTriton
from nemo_export.tensorrt_mm_exporter import TensorRTMMExporter

# Export multimodal model
exporter = TensorRTMMExporter(model_dir="/path/to/export/dir", modality="vision")
exporter.export(
    visual_checkpoint_path="/path/to/model.nemo",
    model_type="mllama",
    llm_model_type="mllama",
    tensor_parallel_size=1,
)

# Deploy to Triton
nm = DeployPyTriton(model=exporter, triton_model_name="mllama", port=8000)
nm.deploy()
nm.serve()
```

## 🔍 Query Deployed Models

### Query LLM Model

```python
from nemo_deploy.nlp import NemoQueryLLM

nq = NemoQueryLLM(url="localhost:8000", model_name="llama")
output = nq.query_llm(
    prompts=["What is the capital of France?"],
    max_output_len=100,
)
print(output)
```

### Query Multimodal Model

```python
from nemo_deploy.multimodal import NemoQueryMultimodal

nq = NemoQueryMultimodal(url="localhost:8000", model_name="mllama", model_type="mllama")
output = nq.query(
    input_text="What is in this image?",
    input_media="/path/to/image.jpg",
    max_output_len=30,
)
print(output)
```

## 🤝 Contributing

We welcome contributions to NeMo Export-Deploy! Please see our [Contributing Guidelines](https://github.com/NVIDIA-NeMo/Export-Deploy/blob/main/CONTRIBUTING.md) for more information on how to get involved.

## License

NeMo Export-Deploy is licensed under the [Apache License 2.0](https://github.com/NVIDIA-NeMo/Export-Deploy?tab=Apache-2.0-1-ov-file).
