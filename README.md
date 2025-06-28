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

NeMo-Export-Deploy provides support for TRT-LLM and vLLM.  

Build a container with TRT-LLM support:

```bash
docker build \
    -f docker/Dockerfile.ci \
    -t nemo-export-deploy \
    --build-arg INFERENCE_FRAMEWORK=trtllm \
    .
```

Or, alternatively to build a container with vLLM support, run:

```bash
docker build \
    -f docker/Dockerfile.ci \
    -t nemo-export-deploy \
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
    nemo-export-deploy
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

For NeMo Export-Deploy without Mcore, TranformerEngine, TRT-LLM and vLLM support, just run:

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

pip install torch setuptools pybind11 wheel_stub
pip install -e --no-build-isolation '.[te]'
```

### Installation with TRT-LLM or vLLM support

Additionally to Megatron-Core/TransformerEngine, users may also add TRT-LLM or vLLM support. Note that TRT-LLM and vLLM are mutually exclusive, attempting to install both together will likely result in an error.

For TRT-LLM/TE, make sure to install `libopenmpi-dev` before.

```bash
sudo apt-get update
sudo apt-get install -y libopenmpi-dev
```

Now, proceed with the actuall installation:

```bash
git clone https://github.com/NVIDIA-NeMo/Export-Deploy
cd Export-Deploy
pip install torch setuptools pybind11 wheel_stub
pip install -e --no-build-isolation '.[te,trllm]'
```

For vLLM:

```bash
git clone https://github.com/NVIDIA-NeMo/Export-Deploy
cd Export-Deploy
pip install -e '.[vllm]'
```

## Documentation

For detailed documentation, please refer to:

- [NeMo-Export-Deploy User Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/overview.html)

## Deploy Models Using Ray Serve Scripts

This section demonstrates how to deploy Large Language Models (LLMs) using Ray Serve with the deployment scripts provided in `/scripts/deploy/nlp`. Ray Serve provides scalable, distributed serving capabilities for machine learning models. This feature is only available in NeMo versions >= 25.07. Currently single node, multi-gpu deployment is supported. 

### Prerequisites

#### Using Docker Container

The recommended way to run Ray Serve deployment scripts is using the pre-built Docker container that includes all necessary dependencies. Change the :vr tag to the version of the container you want to use.

1. **Pull the NeMo Docker container:**

   ```bash
   docker pull nvcr.io/nvidia/nemo:vr
   ```

2. **Start an interactive container session:**

   ```bash
   docker run \
       --gpus all \
       -it \
       --rm \
       --shm-size=4g \
       -p 8000:8000 \
       -p 8265:8265 \
       -v ${PWD}/checkpoints:/opt/checkpoints/ \
       -w /opt/Export-Deploy \
       nvcr.io/nvidia/nemo:vr
   ```

3. **Alternative: Build container from source**

   If you prefer to build the container yourself:

   ```bash
   # For TensorRT-LLM support
   docker build \
       -f docker/Dockerfile.ci \
       -t nemo-export-deploy \
       --build-arg INFERENCE_FRAMEWORK=trtllm \
       .

   # For vLLM support
   docker build \
       -f docker/Dockerfile.ci \
       -t nemo-export-deploy \
       --build-arg INFERENCE_FRAMEWORK=vllm \
       .
   ```

4. **Access models with Hugging Face token (if needed):**

   If you plan to use models that require Hugging Face authentication (like Llama models), set up your token inside the container:

   ```bash
   # Inside the container
   huggingface-cli login
   # Or set the environment variable
   export HF_TOKEN=your_token_here
   ```

### Deploy Hugging Face Models Using Ray Serve

The `deploy_ray_hf.py` script allows you to deploy Hugging Face models directly using Ray Serve.

> **Note:** All the following script examples should be executed inside the Docker container. Make sure you have started the container as described in the [Prerequisites](#prerequisites) section.

#### Quick Example

```bash
python scripts/deploy/nlp/deploy_ray_hf.py \
    --model_path "meta-llama/Llama-3.2-1B" \
    --model_id "llama-3.2-1b" \
    --host "0.0.0.0" \
    --port 8000 \
    --num_gpus 1 \
    --num_replicas 1 \
    --num_gpus_per_replica 1 \
    --trust_remote_code
```

#### Parameters

- `--model_path`: Path to the HuggingFace model or model identifier from HuggingFace Hub (required)
- `--task`: HuggingFace task type (default: "text-generation")
- `--trust_remote_code`: Whether to trust remote code when loading the model
- `--device_map`: Device mapping strategy for model placement (default: "auto")
- `--max_memory`: Maximum memory allocation when using balanced device map
- `--model_id`: Identifier for the model in API responses (default: "nemo-model")
- `--host`: Host address to bind the Ray Serve server to (default: "0.0.0.0")
- `--port`: Port number for the Ray Serve server (default: 1024)
- `--num_cpus`: Number of CPUs for the Ray cluster (default: all available)
- `--num_gpus`: Number of GPUs for the Ray cluster (default: 1)
- `--include_dashboard`: Whether to include the Ray dashboard
- `--num_replicas`: Number of model replicas to deploy (default: 1)
- `--num_gpus_per_replica`: Number of GPUs per model replica (default: 1)
- `--num_cpus_per_replica`: Number of CPUs per model replica (default: 8)
- `--cuda_visible_devices`: Comma-separated list of CUDA visible devices (default: "0,1")

#### Example with Custom Configuration

```bash
python scripts/deploy/nlp/deploy_ray_hf.py \
    --model_path "/path/to/local/hf/model" \
    --model_id "custom-llama" \
    --host "localhost" \
    --port 9000 \
    --num_gpus 2 \
    --num_replicas 2 \
    --num_gpus_per_replica 1 \
    --device_map "balanced" \
    --include_dashboard
```

### Deploy NeMo Models Using Ray Serve (In-Framework)

The `deploy_ray_inframework.py` script deploys NeMo checkpoint files using Ray Serve with support for various parallelism strategies.

#### Quick Example

```bash
python scripts/deploy/nlp/deploy_ray_inframework.py \
    --nemo_checkpoint "/opt/checkpoints/hf_llama32_1B_nemo2" \
    --model_id "nemo-llama" \
    --host "0.0.0.0" \
    --port 8000 \
    --num_gpus 1 \
    --tensor_model_parallel_size 1 \
    --pipeline_model_parallel_size 1
```

#### Parameters

- `--nemo_checkpoint`: Path to the .nemo checkpoint file (required)
- `--num_gpus`: Number of GPUs to use per node (default: 1)
- `--num_nodes`: Number of nodes to use for deployment (default: 1)
- `--tensor_model_parallel_size`: Size of tensor model parallelism (default: 1)
- `--pipeline_model_parallel_size`: Size of pipeline model parallelism (default: 1)
- `--expert_model_parallel_size`: Size of expert model parallelism (default: 1)
- `--context_parallel_size`: Size of context parallelism (default: 1)
- `--model_id`: Identifier for the model in API responses (default: "nemo-model")
- `--host`: Host address to bind the Ray Serve server to (default: "0.0.0.0")
- `--port`: Port number for the Ray Serve server (default: 1024)
- `--num_cpus`: Number of CPUs for the Ray cluster (default: all available)
- `--num_cpus_per_replica`: Number of CPUs per model replica (default: 8)
- `--include_dashboard`: Whether to include the Ray dashboard
- `--cuda_visible_devices`: Comma-separated list of CUDA visible devices (default: "0,1")
- `--enable_cuda_graphs`: Whether to enable CUDA graphs for faster inference
- `--enable_flash_decode`: Whether to enable Flash Attention decode
- `--num_replicas`: Number of replicas for the deployment (default: 1)
- `--legacy_ckpt`: Whether to use legacy checkpoint format

#### Example with Multi-GPU Parallelism

```bash
python scripts/deploy/nlp/deploy_ray_inframework.py \
    --nemo_checkpoint "/opt/checkpoints/large_model.nemo" \
    --model_id "large-nemo-model" \
    --num_gpus 4 \
    --tensor_model_parallel_size 2 \
    --pipeline_model_parallel_size 2 \
    --num_replicas 1 \
    --enable_cuda_graphs \
    --enable_flash_decode \
    --include_dashboard
```

### Deploy TensorRT-LLM Models Using Ray Serve

The `deploy_ray_trtllm.py` script provides deployment of TensorRT-LLM optimized models with Ray Serve. It supports both direct deployment of pre-built engines and on-the-fly conversion from NeMo or Hugging Face models.

#### Deploy Pre-built TensorRT-LLM Engine

```bash
python scripts/deploy/nlp/deploy_ray_trtllm.py \
    --trt_llm_path "/path/to/trtllm/engine" \
    --model_type "llama" \
    --model_id "trtllm-model" \
    --host "0.0.0.0" \
    --port 8000 \
    --num_gpus 1 \
    --tensor_parallelism_size 1
```

#### Export and Deploy NeMo Checkpoint to TensorRT-LLM

```bash
python scripts/deploy/nlp/deploy_ray_trtllm.py \
    --nemo_checkpoint_path "/opt/checkpoints/hf_llama32_1B_nemo2" \
    --model_type "llama" \
    --model_id "trtllm-from-nemo" \
    --tensor_parallelism_size 1 \
    --max_batch_size 8 \
    --max_input_len 2048 \
    --max_output_len 1024 \
    --host "0.0.0.0" \
    --port 8000
```

#### Export and Deploy Hugging Face Model to TensorRT-LLM

```bash
python scripts/deploy/nlp/deploy_ray_trtllm.py \
    --hf_model_path "/path/to/hf/model" \
    --model_type "llama" \
    --model_id "trtllm-from-hf" \
    --tensor_parallelism_size 1 \
    --max_batch_size 16 \
    --max_input_len 4096 \
    --max_output_len 2048 \
    --use_cpp_runtime \
    --enable_chunked_context
```

#### Parameters

**Model Source (mutually exclusive - one required):**
- `--trt_llm_path`: Path to pre-built TensorRT-LLM model directory
- `--nemo_checkpoint_path`: Path to NeMo checkpoint file to export
- `--hf_model_path`: Path to HuggingFace model to export

**Model Configuration:**
- `--model_type`: Model architecture (default: "llama")
- `--tensor_parallelism_size`: Number of tensor parallelism (default: 1)
- `--pipeline_parallelism_size`: Number of pipeline parallelism (default: 1)
- `--max_batch_size`: Maximum batch size (default: 8)
- `--max_input_len`: Maximum input sequence length (default: 2048)
- `--max_output_len`: Maximum output sequence length (default: 1024)

**Runtime Options:**
- `--use_python_runtime`: Use Python runtime (default behavior)
- `--use_cpp_runtime`: Use C++ runtime (overrides Python runtime)
- `--enable_chunked_context`: Enable chunked context (C++ runtime only)
- `--max_tokens_in_paged_kv_cache`: Maximum tokens in paged KV cache (C++ runtime only)
- `--multi_block_mode`: Enable multi-block mode
- `--lora_ckpt_list`: List of LoRA checkpoint paths

**API Configuration:**
- `--model_id`: Model identifier for API responses (default: "tensorrt-llm-model")
- `--host`: Host address (default: "0.0.0.0")
- `--port`: Port number (default: 1024)

**Ray Cluster Configuration:**
- `--num_cpus`: Number of CPUs for Ray cluster (default: all available)
- `--num_gpus`: Number of GPUs for Ray cluster (default: 1)
- `--include_dashboard`: Include Ray dashboard
- `--num_replicas`: Number of model replicas (default: 1)
- `--num_gpus_per_replica`: GPUs per replica (default: 1)
- `--num_cpus_per_replica`: CPUs per replica (default: 8)
- `--cuda_visible_devices`: CUDA visible devices (default: "0,1")

#### Example with Advanced Configuration

```bash
python scripts/deploy/nlp/deploy_ray_trtllm.py \
    --nemo_checkpoint_path "/opt/checkpoints/large_model.nemo" \
    --model_type "llama" \
    --model_id "optimized-large-model" \
    --tensor_parallelism_size 2 \
    --max_batch_size 32 \
    --max_input_len 8192 \
    --max_output_len 4096 \
    --use_cpp_runtime \
    --enable_chunked_context \
    --multi_block_mode \
    --num_replicas 2 \
    --num_gpus_per_replica 2 \
    --include_dashboard
```

### Querying Deployed Models

After deploying a model using any of the Ray Serve scripts, you can query it using the provided query script or by making HTTP requests directly.

#### Using HTTP Requests

**Completions Endpoint:**
```bash
curl -X POST "http://localhost:8000/v1/completions/" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "your-model-id",
        "prompt": "What is the capital of France?",
        "max_tokens": 100,
        "temperature": 0.7
    }'
```

**Chat Completions Endpoint:**
```bash
curl -X POST "http://localhost:8000/v1/chat/completions/" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "your-model-id",
        "messages": [{"role": "user", "content": "Hello, how are you?"}],
        "max_tokens": 50,
        "temperature": 0.7
    }'
```

**Models Endpoint:**
```bash
curl "http://localhost:8000/v1/models"
```

**Health Check:**
```bash
curl "http://localhost:8000/v1/health"
```

### Advanced Deployment Scenarios

#### Multi-Replica Deployment with Load Balancing

Deploy multiple replicas for higher throughput:

```bash
python scripts/deploy/nlp/deploy_ray_hf.py \
    --model_path "meta-llama/Llama-3.2-1B" \
    --model_id "load-balanced-llama" \
    --num_replicas 4 \
    --num_gpus_per_replica 1 \
    --num_gpus 4
```

## Contributing

We welcome contributions to NeMo Export and Deploy! Please see our [Contributing Guidelines](CONTRIBUTING.md) for more information on how to get involved.

## License

NeMo Export-Deploy is licensed under the [Apache License 2.0](https://github.com/NVIDIA-NeMo/Export-Deploy?tab=Apache-2.0-1-ov-file).
