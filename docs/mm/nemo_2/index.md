# Export and Deploy NeMo 2.0 MMs

The [NeMo 2.0](https://github.com/NVIDIA-NeMo/NeMo) model and checkpoint format is the current standard for saving and deploying Multimodals (MMs) trained with the NeMo Framework. 

With the Export-Deploy library, you can seamlessly export and deploy the NeMo 2.0 models and checkpoints across a variety of production environments. The following export and deployment paths are supported for NeMo 2.0 models:

- **Model deployment with Triton:** Directly serve NeMo 2.0 models using the NVIDIA Triton Inference Server.
- **TensorRT-LLM export and deployment with Triton:** Convert NeMo 2.0 checkpoints into optimized TensorRT-LLM engines for high-performance inference, deployable via Triton Inference Server.


```{toctree}
:maxdepth: 4
:titlesonly:
:hidden:

Generate a NeMo 2.0 Checkpoint <gen_nemo2_ckpt.md>
Deploy with Triton <in-framework.md>
Export and Deploy <optimized/index.md>
```
