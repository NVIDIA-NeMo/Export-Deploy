# Export and Deploy Large Language Models

The Export-Deploy library provides comprehensive tools and APIs for exporting and deploying Large Language Models (LLMs) to production environments. This library supports multiple checkpoint formats and offers various deployment paths including TensorRT-LLM and vLLM deployment through NVIDIA Triton Inference Server and Ray Serve.

## Overview

The Export-Deploy library enables seamless conversion of LLMs from various checkpoint formats into optimized inference engines, supporting both single-GPU, multi-GPU and multi-node deployments. Whether you're working with NeMo 2.0 models, Megatron Bridge, Hugging Face models, or other formats, the library provides unified APIs for model export and deployment.

## Supported Checkpoint Formats

The library supports several checkpoint formats, each with specific capabilities and deployment options:


### NeMo 2.0 Checkpoints

[NeMo 2.0](https://github.com/NVIDIA-NeMo/NeMo) represents the current checkpoint format from the NeMo Framework, storing all model-related files in a directory structure rather than a single archive file. NeMo 2.0 model format will be deprecated soon.

**Supported Export and Deployment Paths:**
- Model deployment with Triton and Ray Serve
- TensorRT-LLM export and deployment with Triton and Ray Serve
- vLLM export and deployment with Triton


### Megatron Bridge Checkpoints

[Megatron Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge) checkpoints are designed to serve as the successor to the NeMo 2.0 checkpoint format. This new format will eventually replace NeMo 2.0, providing a more robust and interoperable solution for model export and deployment workflows.

**Supported Export and Deployment Paths:**
- Model deployment with Triton and Ray Serve

**Export and Deployment Paths Coming Soon:**
- TensorRT-LLM export and deployment with Triton and Ray Serve
- vLLM export and deployment with Triton


### AutoModel Checkpoints

[AutoModel](https://github.com/NVIDIA-NeMo/Automodel) checkpoints are Hugging Face-compatible formats generated through NeMo AutoModel workflows, providing a simplified interface for working with pre-trained language models.

**Supported Export and Deployment Paths:**
- Model deployment with Triton and Ray Serve
- TensorRT-LLM export and deployment with Triton and Ray Serve
- vLLM export and deployment with Triton


### Megatron-LM Checkpoints

[Megatron-LM](https://github.com/NVIDIA/Megatron-LM) checkpoints represent models trained using the Megatron-LM framework, offering support for large-scale distributed training and inference.

**Supported Export and Deployment Paths:**
- Model deployment with Triton



## Next Steps

- [NeMo 2.0](nemo_2/index.md) - Detailed guide for NeMo 2.0 checkpoint export and deployment
- [Megatron Bridge](mbridge/index.md) - Detailed guide for Megatron Bridge checkpoint export and deployment
- [Automodel](automodel/index.md) - Detailed guide for Automodel (Hugging Face) checkpoint export and deployment
- [Megatron-LM Models](megatron_lm/index) - Detailed guide for Megatron-LM checkpoint export and deployment


