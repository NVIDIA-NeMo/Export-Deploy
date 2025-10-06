# Export and Deploy Megatron-Bridge LLMs

The [Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge) checkpoint format is the new standard for saving and deploying large language models (LLMs) trained with the Megatron-Bridge library. 

With the Export-Deploy library, you can seamlessly export and deploy Megatron-Bridge checkpoints across a variety of production environments. The following export and deployment paths are supported for Megatron-Bridge models:

- **Model deployment with Triton and Ray Serve:** Directly serve Megatron-Bridge models using NVIDIA Triton Inference Server or Ray Serve for scalable inference.
- **TensorRT-LLM export and deployment with Triton and Ray Serve:** Convert Megatron-Bridge checkpoints into optimized TensorRT-LLM engines for high-performance inference, deployable via Triton or Ray Serve. Support for this feature is coming soon.
- **vLLM export and deployment with Triton:** Export Megatron-Bridge models to the vLLM format for efficient serving with Triton. Support for this feature is coming soon.


```{toctree}
:maxdepth: 4
:titlesonly:
:hidden:

Generate a Megatron-Bridge Checkpoint <gen_mbridge_ckpt.md>
Deploy with Triton <in-framework.md>
Deploy with Ray Serve <in-framework-ray.md>
Export and Deploy <optimized/index.md>
```

