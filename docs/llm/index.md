# Export and Deploy Large Language Models

The Export-Deploy library provides comprehensive tools and APIs for exporting and deploying Large Language Models (LLMs) to production environments. This library supports multiple checkpoint formats and offers various deployment paths including TensorRT-LLM and vLLM deployment through NVIDIA Triton Inference Server and Ray Serve.

## Overview

The Export-Deploy library enables seamless conversion of LLMs from various checkpoint formats into optimized inference engines, supporting both single-GPU, multi-GPU and multi-node deployments. Whether you're working with Megatron Bridge, Hugging Face models, or other formats, the library provides unified APIs for model export and deployment.

## Supported Model/Checkpoint Formats

The library supports several checkpoint formats, each with specific capabilities and deployment options:


### Megatron Bridge Model/Checkpoints

[Megatron Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge) is designed to serve as the successor to the NeMo 2.0. This new library will eventually replace NeMo 2.0.

**Supported Export and Deployment Paths:**
- Model deployment with Triton and Ray Serve

**Export and Deployment Paths Coming Soon:**
- TensorRT-LLM export and deployment with Triton and Ray Serve
- vLLM export and deployment with Triton and Ray Serve


### AutoModel Model/Checkpoints

[AutoModel](https://github.com/NVIDIA-NeMo/Automodel) checkpoints are Hugging Face-compatible formats generated through NeMo AutoModel workflows, providing a simplified interface for working with pre-trained language models.

**Supported Export and Deployment Paths:**
- Model deployment with Triton and Ray Serve
- TensorRT-LLM export and deployment with Triton and Ray Serve
- vLLM export and deployment with Triton and Ray Serve


### Megatron-LM Model/Checkpoints

[Megatron-LM](https://github.com/NVIDIA/Megatron-LM) checkpoints represent models trained using the Megatron-LM framework, offering support for large-scale distributed training and inference.

**Supported Export and Deployment Paths:**
- Model deployment with Triton



