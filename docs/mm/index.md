# Export and Deploy Multimodal Models

The Export-Deploy library provides comprehensive tools and APIs for exporting and deploying Multimodal Models (MMs) to production environments. This library supports multiple checkpoint formats and offers various deployment paths including TensorRT-LLM deployment through NVIDIA Triton Inference Server.

## Overview

The Export-Deploy library enables seamless conversion of MMs from various checkpoint formats into optimized inference engines, supporting both single-GPU, multi-GPU and multi-node deployments. Whether you're working with NeMo 2.0 models, Megatron Bridge, Hugging Face models, or other formats, the library provides unified APIs for model export and deployment.

## Supported Model/Checkpoint Formats

The library supports several checkpoint formats, each with specific capabilities and deployment options:


### NeMo 2.0 Model/Checkpoints

[NeMo 2.0](https://github.com/NVIDIA-NeMo/NeMo) represents the current checkpoint format from the NeMo Framework, storing all model-related files in a directory structure rather than a single archive file. NeMo 2.0 model format will be deprecated soon.

**Supported Export and Deployment Paths:**
- Model deployment with Triton
- TensorRT-LLM export and deployment with Triton


### Megatron Bridge Model/Checkpoints

[Megatron Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge) is designed to serve as the successor to the NeMo 2.0. This new library will eventually replace NeMo 2.0.

**Export and Deployment Paths Coming Soon:**
- Model deployment with Triton and Ray Serve
- TensorRT-LLM export and deployment with Triton and Ray Serve


### AutoModel Model/Checkpoints

[AutoModel](https://github.com/NVIDIA-NeMo/Automodel) checkpoints are Hugging Face-compatible formats generated through NeMo AutoModel workflows, providing a simplified interface for working with pre-trained language models.

**Export and Deployment Paths Coming Soon:**
- Model deployment with Triton
- TensorRT-LLM export and deployment with Triton






