# Changelog

* [NeMo Export-Deploy Release](https://github.com/NVIDIA-NeMo/Export-Deploy/releases)  

## NVIDIA NeMo-Export-Deploy 0.2.1
* Bug fixes for HuggingFace model deployment (#459)
  - Fixed HuggingFace deployable implementations for both Triton and Ray Serve backends
  - Improved tokenizer handling in HuggingFace deployment scripts
* Minor fixes for Ray deployment (#464)
  - Additional bug fixes in Ray deployment utilities

## NVIDIA NeMo-Export-Deploy 0.2.0
* MegatronLM and Megatron-Bridge model deployment support with Triton Inference Server and Ray Serve
* Multi-node multi-instance Ray Serve based deployment for NeMo 2, Megatron-Bridge, and Megatron-LM models. 
* Update vLLM export to use NeMo->HF->vLLM export path
* Multi-Modal deployment for NeMo 2 models with Triton Inference Server
* NeMo Retriever Text Reranking ONNX and TensorRT export support

## NVIDIA NeMo-Export-Deploy 0.1.0
* Pip installers for export and deploy  
* RayServe support for multi-instance deployment  
* TensorRT-LLM PyTorch backend  
* mcore inference optimizations
