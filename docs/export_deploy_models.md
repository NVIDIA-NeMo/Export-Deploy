# Deploy NeMo Framework Models

NVIDIA NeMo Framework offers various deployment paths for NeMo models, tailored to different domains such as Large Language Models (LLMs) and Multimodal Models (MMs). There are two primary deployment paths for NeMo models: deploying NeMo models with exporting to inference optimized libraries such as TensorRT, TensorRT-LLM, or vLLM and deploying NeMo models at the PyTorch level (in-framework), both with the NVIDIA Triton Inference Server. To begin serving your model on these two deployment paths, all you need is a NeMo checkpoint.

While a number of deployment paths are currently available for different domains, others are still in development. As each unique deployment path for a domain becomes available, it will be added to this section.

## Deploy NeMo Models by Exporting Inference Optimized libraries

For scenarios requiring optimized performance, NeMo models can leverage inference optimized libraries such as TensorRT, TensorRT-LLM, and vLLM, specialized libraries for accelerating and optimizing inference on NVIDIA GPUs. This process involves converting NeMo models into a format compatible with the library using the nemo.export module.  Moreover, NeMo offers quantization methods including Post-Training Quantization that can be used to produce low-precision checkpoint formats for efficient deployment, for example, in FP8.

NVIDIA also offers NIM for enterprises seeking a reliable and scalable solution to deploy generative AI models. This option is currently only available for LLMs.


## NVIDIA NIM for LLMs

Enterprises seeking a comprehensive solution that covers both on-premises and cloud deployment can use the NVIDIA Inference Microservices (NIM). This approach leverages the NVIDIA AI Enterprise suite, which includes support for NVIDIA NeMo, Triton Inference Server, TensorRT-LLM, and other AI software.

This option is ideal for organizations requiring a reliable and scalable solution to deploy generative AI models in production environments. It also stands out as the fastest inference option, offering user-friendly scripts and APIs. Leveraging the TensorRT-LLM Triton backend, it achieves rapid inference using advanced batching algorithms, including in-flight batching. Note that this deployment path supports only selected LLM models.

To learn more about NVIDIA NIM, visit the [NVIDIA website](https://www.nvidia.com/en-gb/launchpad/ai/generative-ai-inference-with-nim).


## Deploy PyTorch-level Models (In-Framework)

Deploying PyTorch-level models involves running models directly within the NeMo Framework. This approach is straightforward and eliminates the need to export models to another library. It is ideal for development and testing phases, where ease-of-use and flexibility are critical. The NeMo Framework supports multi-node and multi-GPU inference, maximizing throughput. This method allows for rapid iterations and direct testing within the NeMo environment. Although this is the slowest option, it provides support for almost all NeMo models.

Please check the links to learn more about deploying [Large Language Models (LLMs)](llm/index.md) and [Multimodal Models (MMs)](mm/index.md).

