# Export and Deploy NeMo 2.0 LLMs

The NeMo 2.0 checkpoint format is the current standard for saving and deploying large language models (LLMs) trained with the NeMo Framework. 

With the Export-Deploy library, you can seamlessly export and deploy the NeMo 2.0 checkpoints across a variety of production environments. The following export and deployment paths are supported for NeMo 2.0 models:

- **Model deployment with Triton and Ray Serve:** Directly serve NeMo 2.0 models using the NVIDIA Triton Inference Server or Ray Serve for scalable inference.
- **TensorRT-LLM export and deployment with Triton and Ray Serve:** Convert NeMo 2.0 checkpoints into optimized TensorRT-LLM engines for high-performance inference, deployable via Triton or Ray Serve.
- **vLLM export and deployment with Triton:** Export NeMo 2.0 models to the vLLM format for efficient serving with Triton.


```{toctree}
:maxdepth: 4
:titlesonly:
:hidden:

Generate a NeMo 2.0 Checkpoint <gen_nemo2_ckpt.md>
Deploy with Triton <in-framework.md>
Deploy with Ray Serve <in-framework-ray.md>
Export and Deploy <optimized/index.md>
Send a Query <send-query.md>
```
