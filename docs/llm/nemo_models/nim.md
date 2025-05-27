# Deploy NeMo Models Using NIM LLM Containers

NVIDIA NIM for LLMs is a containerized solution designed to deploy PyTorch-level NeMo Large Language Models (LLMs) efficiently using the TensorRT-LLM backend. NIM provides optimized inference performance while maintaining ease of deployment through containerization. This section demonstrates how to deploy NeMo models using a NIM container. There are three deployment paths available for exporting a NeMo 2.0 checkpoint to TensorRT-LLM and deploying it in a NIM LLM container, each suited for different use cases and requirements. Please note that NIM LLM container must support the model in order to use the deployment paths.


## Exporting NeMo 2.0 Models to Hugging Face
The primary deployment path for NeMo models in NIM LLM containers involves exporting the model to Hugging Face format and leveraging NIM's built-in Hugging Face framework support. This approach offers several advantages including the simplified deployment process using standard NIM tooling.

Requirements:

- A corresponding Hugging Face model implementation must exist.
- The NeMo model must have Hugging Face export support implemented.

The deployment process involves two main steps:

#. Export NeMo 2.0 model to Hugging Face checkpoint format

   You can find detailed examples of converting NeMo models to Hugging Face format in these tutorials:
   
   - [Fine-Tuning LLMs for Function Calling](https://github.com/NVIDIA/NeMo/blob/main/tutorials/llm/function_calling/nemo2-chat-sft-function-calling.ipynb)
   - [PEFT in NeMo 2.0](https://docs.nvidia.com/nemo-framework/user-guide/latest/sft_peft/peft_nemo2.html)
   
   These tutorials provide comprehensive guidance on using the ``export_ckpt`` API in NeMo. Please note that NeMo 2.0 LoRA models are only supported in NIM LLM containers via Hugging Face export. Since there is no engine creation needed for LoRA models, NIM LLM container will include LoRA Hugging Face checkpoint dynamically in runtime.

#. Deploy using NIM LLM container

   For detailed deployment instructions, refer to [the official NIM documentation](https://docs.nvidia.com/nim/large-language-models/latest/ft-support.html#usage)


If Hugging Face export is not available for your NeMo model, you can use the alternative deployment paths described below. These paths enable direct export and deployment of NeMo models using NeMo Framework and NIM LLM containers without requiring Hugging Face conversion.


## Exporting NeMo 2.0 Models to TensorRT-LLM in NIM LLM Container
Another deployment option is to export NeMo models directly to TensorRT-LLM format within a NIM LLM container. This approach requires installing the NeMo Export module in the container but provides more direct control over the export process.

This example demonstrates exporting a Llama 3.1 8B Instruct model to TensorRT-LLM format. We'll assume the NeMo 2.0 checkpoint is located at `/opt/checkpoints/llama-3.1-8b-instruct-nemo2`. You can obtain such a checkpoint through fine-tuning - for details, see the :ref:`Supervised Fine-Tuning <developer-guide-sft>` guide.

#. To use the latest NeMo codebase and access the [export_to_trt_llm.py](https://github.com/NVIDIA/NeMo/blob/main/scripts/export/export_to_trt_llm.py) script, clone the NeMo Framework repository:

   ```shell
   git clone https://github.com/NVIDIA/NeMo.git /opt/NeMo
   ```

#. Go to [NIM NGC container collection](https://catalog.ngc.nvidia.com/containers?filters=nvidia_nim%7CNVIDIA+NIM%7Cnimmcro_nvidia_nim&orderBy=weightPopularDESC&query=&page=&pageSize=) and find a recent NIM image to use. In our example, we choose `nvcr.io/nim/meta/llama-3.1-8b-instruct:1.8.0-RTX` that comes with TensorRT-LLM 0.17.1. Note that currently using the latest ``nemo.export`` requires TensorRT-LLM version 0.16.0 or higher.

#. Run the container mounting both the cloned NeMo sources and the folder with your checkpoint:

   ```shell
   docker run --gpus all -it --rm --shm-size=4g \
       -v /opt/NeMo:/opt/NeMo \
       -v /opt/checkpoints:/opt/checkpoints \
       -w /workspace nvcr.io/nim/meta/llama-3.1-8b-instruct:1.8.0-RTX bash
   ```

#. Run the following commands to produce the engine. In the example, use tensor parallelism of two:

   ```shell
   export PYTHONPATH=/opt/NeMo  # To use the latest code for NeMo 2.0 checkpoint support
   export NIM_MODEL_NAME=/workspace/llama-3.1-8b-instruct-nim  # To inform NIM scripts where the model resides

   mkdir $NIM_MODEL_NAME

   python /opt/NeMo/scripts/export/export_to_trt_llm.py \
       --nemo_checkpoint /opt/checkpoints/llama-3.1-8b-instruct-nemo2 \
       --model_repository $NIM_MODEL_NAME \
       --tensor_parallelism_size 2
   ```

   Wait for the command to complete the TensorRT-LLM engine build. It will produce the model format ready to be run with the NIM container.

#. Run the following script to start the server. The script is included in the NIM containers. It sets up the server within several minutes (which in general depends on the model size).

   ```shell
   export NIM_SERVED_MODEL_NAME=test-model  # Model name used to query it

   /opt/nim/start_server.sh > server.log 2>&1 &
   ```

   Wait until the server starts. You can inspect the `server.log` file until it reports the server is running at the default 8000 port.

#. Once the server is up, you can query it with an example prompt using ``curl``:

   ```shell
   apt-get update && apt-get install -y curl

   curl -X 'POST' \
       'http://0.0.0.0:8000/v1/completions' \
       -H 'accept: application/json' \
       -H 'Content-Type: application/json' \
       -d '{
         "model": "'"$NIM_SERVED_MODEL_NAME"'",
         "prompt": "hello world!",
         "top_p": 1,
         "n": 1,
         "max_tokens": 15,
         "stream": false,
         "frequency_penalty": 1.0,
         "stop": ["hello"]
       }'
   ```

## Exporting NeMo 2.0 Models to TensorRT-LLM in NeMo FW Container
 You can generate TensorRT-LLM engines in the NeMo Framework container and deploy them in the NIM LLM container. However, this approach requires matching TensorRT-LLM versions between both containers. When the versions match, this is the recommended path for deploying NeMo models to the NIM LLM container. If the TensorRT-LLM versions differ between containers, you should use the NIM container to build the engine as demonstrated in the previous section.

#. First, generate a TensorRT-LLM engine for the llama-3.1-8b-instruct-nemo2 model by following the instructions in the :ref:`Deploy NeMo Models by Exporting TensorRT-LLM <deploy-nemo-framework-models-tensorrt-llm>` guide.

#. Then, start the NIM LLM container as follows:

   ```shell
   docker run --gpus all -it --rm --shm-size=4g \
       -v /opt/checkpoints:/opt/checkpoints \
       -w /workspace nvcr.io/nim/meta/llama-3.1-8b-instruct:1.8.0-RTX bash
   ```

   Please make sure the TensorRT-LLM engine is located in the `/opt/checkpoints` folder.

#. Run the following script to start the server. The script is included in the NIM containers. It sets up the server within several minutes (which in general depends on the model size).

   ```shell
   export NIM_MODEL_NAME=/opt/checkpoints/llama-3.1-8b-instruct-nim
   export NIM_SERVED_MODEL_NAME=test-model  # Model name used to query it
      
   /opt/nim/start_server.sh > server.log 2>&1 &
   ```

   Wait until the server starts. You can inspect the `server.log` file until it reports the server is running at the default 8000 port.
