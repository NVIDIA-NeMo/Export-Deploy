# **NeMo Export and Deploy**

## Introduction

NVIDIA NeMo Export and Deploy library provides tools and APIs for exporting and deploying NeMo models to production environments. It supports various deployment paths including TensorRT, TensorRT-LLM, vLLM, and in-framework deployment through NVIDIA Triton Inference Server.

## Key Features

- Support for Large Language Models (LLMs) and Multimodal Models
- Export NeMo models to optimized inference formats including TensorRT-LLM and vLLM
- Deploy NeMo and Hugging Face models using Ray Serve or NVIDIA Triton Inference Server
- Quantization support for efficient deployment (FP8, etc.)
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

Build a container with all dependencies:

```bash
docker build -f docker/Dockerfile.ci -t nemo-export-deploy .
```

Start an interactive terminal inside the container:

```bash
docker run --rm -it --entrypoint bash --runtime nvidia --gpus all nemo-export-deploy
```

### Export and Deploy Examples

#### 1. Export TensorRT-LLM and Deploy using Triton Inference Server

```python
from nemo.export.tensorrt_llm import TensorRTLLM
from nemo.deploy import DeployPyTriton

# Export model to TensorRT-LLM
trt_llm_exporter = TensorRTLLM(model_dir="/path/to/export/dir")
trt_llm_exporter.export(
    nemo_checkpoint_path="/path/to/model.nemo",
    model_type="llama",
    tensor_parallelism_size=1,
)

# Deploy to Triton
nm = DeployPyTriton(model=trt_llm_exporter, triton_model_name="llama", http_port=8000)
nm.deploy()
nm.serve()
```

#### 2. Export vLLM and Deploy using Triton Inference Server

```python
from nemo.export.vllm_exporter import vLLMExporter
from nemo.deploy import DeployPyTriton

# Export model to vLLM
exporter = vLLMExporter()
exporter.export(
    nemo_checkpoint="/path/to/model.nemo",
    model_dir="/path/to/export/dir",
    model_type="llama",
    tensor_parallel_size=1,
)

# Deploy to Triton
nm = DeployPyTriton(model=exporter, triton_model_name="llama", http_port=8000)
nm.deploy()
nm.serve()
```

#### 3. Deploy Multimodal Model

```python
from nemo.export.tensorrt_mm_exporter import TensorRTMMExporter
from nemo.deploy import DeployPyTriton

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
from nemo.deploy import NemoQueryLLM

nq = NemoQueryLLM(url="localhost:8000", model_name="llama")
output = nq.query_llm(
    prompts=["What is the capital of France?"],
    max_output_len=100
)
print(output)
```

#### Query Multimodal Model

```python
from nemo.deploy.multimodal import NemoQueryMultimodal

nq = NemoQueryMultimodal(url="localhost:8000", model_name="neva", model_type="neva")
output = nq.query(
    input_text="What is in this image?",
    input_media="/path/to/image.jpg",
    max_output_len=30
)
print(output)
```

## Deployment Options

### 1. NVIDIA NIM (Enterprise Solution)
For enterprise deployment, NVIDIA offers NIM (NVIDIA Inference Microservices) which provides:
- On-premises and cloud deployment
- Fastest inference using TensorRT-LLM
- Advanced batching algorithms
- Production-ready deployment

### 2. In-Framework Deployment
For development and testing:
- Direct deployment within NeMo Framework
- Support for most NeMo models
- Multi-node and multi-GPU inference
- Easiest to use but slower performance

### 3. Optimized Deployment
For production performance:
- Export to TensorRT, TensorRT-LLM, or vLLM
- Quantization support (FP8)
- Maximum throughput and efficiency
- Advanced optimization features

## Support Matrix

NeMo-Framework provides different levels of support based on OS/Platform and installation method:

| OS / Platform              | Install from PyPi | Source into NGC container |
|----------------------------|-------------------|---------------------------|
| `linux` - `amd64/x84_64`   | Limited support   | Full support              |
| `linux` - `arm64`          | Limited support   | Limited support           |
| `darwin` - `amd64/x64_64`  | Deprecated        | Deprecated                |
| `darwin` - `arm64`         | Limited support   | Limited support           |
| `windows` - `amd64/x64_64` | No support yet    | No support yet            |
| `windows` - `arm64`        | No support yet    | No support yet            |

## Documentation

For detailed documentation, please refer to:
- [NeMo Framework User Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/playbooks/index.html)
- [NeMo 2.0 Quickstart](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/quickstart.html)
- [NeMo 2.0 Feature Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/features/index.html)

## Contributing

We welcome contributions to NeMo RL! Please see our [Contributing Guidelines](CONTRIBUTING.md) for more information on how to get involved.

## License

- [NeMo GitHub Apache 2.0 license](https://github.com/NVIDIA/NeMo?tab=Apache-2.0-1-ov-file#readme)
- NeMo is licensed under the [NVIDIA AI PRODUCT AGREEMENT](https://www.nvidia.com/en-us/data-center/products/nvidia-ai-enterprise/eula/)
