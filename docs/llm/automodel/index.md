# Export and Deploy Automodel LLMs

NeMo AutoModel is NVIDIA’s open-source, PyTorch-based library designed for efficient, scalable training and fine-tuning of large language models (LLMs) and vision-language models (VLMs) using DTensor-native SPMD parallelism. This section provides a practical, step-by-step guide to exporting, deploying, and serving AutoModel LLMs in a variety of environments. Built for maximum compatibility, NeMo AutoModel uses Hugging Face model architectures and checkpoint formats—so the workflows described here are applicable to most LLMs available on Hugging Face. Whether you’re looking to deploy at scale or experiment locally, you’ll find everything you need to get started.

You will find step-by-step guides for deploying Automodel LLMs with both Triton Inference Server and Ray Serve, as well as instructions for exporting models and integrating them into your own applications. Whether you are looking to serve models at scale or experiment with local deployments, the following pages will help you get started quickly.


```{toctree}
:maxdepth: 4
:titlesonly:
:hidden:

Deploy with Triton  <automodel-in-framework.md>
Deploy with Ray Serve  <automodel-ray.md>
Export and Deploy <optimized/index.md>
```