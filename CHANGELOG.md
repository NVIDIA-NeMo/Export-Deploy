# Changelog

* [NeMo Export-Deploy Release](https://github.com/NVIDIA-NeMo/Export-Deploy/releases)  

## NVIDIA NeMo-Export-Deploy 0.3.1
* Fix vLLM top_p parameter handling in HuggingFace Ray deployment (#524)
* Pin peft dependency to <0.14.0 for compatibility (#524)

## NVIDIA NeMo-Export-Deploy 0.3.0
* Update TensorRT-LLM export to use NeMo->HF->TensorRT-LLM export path
* Add chat template support for VLM deployment.
* Bug fixes and folder name updates such as updating nlp to llm.

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
