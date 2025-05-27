# Deploy NeMo Models by Exporting to Inference Optimized Libraries

NeMo Framework offers scripts and APIs to export models to two inference optimized libraries, TensorRT-LLM and vLLM, and to deploy the exported model with the NVIDIA Triton Inference Server. Please check the table below to see which models are supported.


## Supported LLMs

The following table shows the supported LLMs and their inference-optimized libraries in the distributed NeMo checkpoint format.



| Model Name| Model Parameters| NeMo 2.0 to TensorRT-LLM | NeMo 2.0 to vLLM |
|:--------- | :-------------- |--------------------------|------------------|
|GPT        |   2B, 8B, 43B   | &check;                  | &cross;          | 
|Nemotron   |   8B, 22B       | &check;                  | &cross;          | 
|Llama 2    |   7B, 13B, 70B  | &check;                  | &check;          |
|Llama 3    |   8B, 70B       | &check;                  | &check;           | 
|Llama 3.1  |   8B, 70B, 405B | &check;                  | &check;          |
|Falcon     |   7B, 40B       | &check;                  | &cross;                 |
|Gemma      |   2B, 7B        | &check;                  | &check;                 |
|StarCoder1 |   15B           | &check;                  | &cross;                 |
|StarCoder2 |   3B, 7B, 15B   | &check;                  | &check;                 | 
|Mistral    |   7B            | &check;                  | &check;                 |
|Mixtral    |   8x7B          | &check;                  | &check;                 |


**Note:** As we transition support for deploying community models from NeMo 1.0 to NeMo 2.0, not all models are supported in NeMo 2.0 yet. The support matrix above shows which models are currently available. To use a model not yet supported in NeMo 2.0, please refer to the documentation for version 24.07, which uses NeMo 1.0 instead.

You can find details about TensorRT-LLM and vLLM-based deployment options below.