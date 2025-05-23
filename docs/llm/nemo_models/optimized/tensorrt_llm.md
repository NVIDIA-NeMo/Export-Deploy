# Deploy NeMo Models by Exporting TensorRT-LLM 

This section shows how to use scripts and APIs to export a NeMo LLM (*.nemo*) or quantized LLM (*.qnemo*) to TensorRT-LLM and deploy it with the NVIDIA Triton Inference Server.

## Quick Example

1. Follow the steps in the [Deploy NeMo LLM main page](../../index.md) to generate a NeMo 2.0 Llama checkpoint.

2. In a terminal, go to the folder where the ``hf_llama31_8B_nemo2.nemo`` file is located. Pull down and run the Docker container image using the command shown below. Change the ``:vr`` tag to the version of the container you want to use:

   ```shell
   docker pull nvcr.io/nvidia/nemo:vr

   docker run --gpus all -it --rm --shm-size=4g -p 8000:8000 \
       -v ${PWD}/hf_llama31_8B_nemo2.nemo:/opt/checkpoints/hf_llama31_8B_nemo2.nemo \
       -w /opt/NeMo \
       nvcr.io/nvidia/nemo:vr
   ```

3. Run the following deployment script to verify that everything is working correctly. The script exports the Llama NeMo checkpoint to TensorRT-LLM and subsequently serves it on the Triton server:

    ```shell
    python scripts/deploy/nlp/deploy_triton.py \
        --nemo_checkpoint /opt/checkpoints/hf_llama31_8B_nemo2.nemo \
        --model_type llama \
        --triton_model_name llama \
        --tensor_parallelism_size 1
   ```

4. If the test yields a shared memory-related error, increase the shared memory size using ``--shm-size`` (gradually by 50%, for example).

5. In a separate terminal, run the following command to get the container ID of the running container. Please access the ``nvcr.io/nvidia/nemo:vr`` image to get the container ID.

   ```shell
   docker ps
   ```
   
6. Access the running container and replace ``container_id`` with the actual container ID as follows:

   ```shell
   docker exec -it container_id bash
   ```

7. To send a query to the Triton server, run the following script:

   ```shell
   python scripts/deploy/nlp/query.py -mn llama -p "What is the color of a banana?" -mol 5
   ```
   
8. To export and deploy a different model such as Llama3, Mixtral, or Starcoder, change the ``model_type`` in the `deploy_triton.py <https://github.com/NVIDIA/NeMo/blob/main/scripts/deploy/nlp/deploy_triton.py>`_ script. Please check below to see the list of the model types. For quantized *qnemo* models, specifying ``model_type`` is not necessary as this information is included in the TensorRT-LLM checkpoint.


## Use a Script to Deploy NeMo LLMs on a Triton Server

You can deploy a LLM from a NeMo checkpoint on Triton using the provided script.

### Export and Deploy a LLM Model 

After executing the script, it will export the model to TensorRT-LLM and then initiate the service on Triton.

1. Start the container using the steps described in the **Quick Example** section.

2. To begin serving the [downloaded model](../../index.md) or a [quantized](https://docs.nvidia.com/nemo-framework/user-guide/latest/model-optimization/quantization/quantization.html) model, run the following script:

   ```shell
   python scripts/deploy/nlp/deploy_triton.py \
       --nemo_checkpoint /opt/checkpoints/hf_llama31_8B_nemo2.nemo \
       --model_type llama \
       --triton_model_name llama \
       --tensor_parallelism_size 1
   ```

   The following parameters are defined in the ``deploy_triton.py`` script:

   - ``--nemo_checkpoint``: path of the .nemo or .qnemo checkpoint file.
   - ``--model_type``: type of the model. choices=["gptnext", "gpt", "llama", "falcon", "starcoder", "mixtral", "gemma"].
   - ``--triton_model_name``: name of the model on Triton.
   - ``--triton_model_version``: version of the model. Default is 1.
   - ``--triton_port``: port for the Triton server to listen for requests. Default is 8000.
   - ``--triton_http_address``: HTTP address for the Triton server. Default is 0.0.0.0.
   - ``--triton_model_repository``: TensorRT temp folder. Default is ``/tmp/trt_llm_model_dir/``.
   - ``--tensor_parallelism_size``: number of GPUs to split the tensors for tensor parallelism. Default is 1.
   - ``--pipeline_parallelism_size``: number of GPUs to split the model for pipeline parallelism. Default is 1.
   - ``--dtype``: data type of the model on TensorRT-LLM. Default is "bfloat16". Currently, only "bfloat16" is supported.
   - ``--max_input_len``: maximum input length of the model. Default is 256. 
   - ``--max_output_len``: maximum output length of the model. Default is 256. 
   - ``--max_batch_size``: maximum batch size of the model. Default is 8. 
   - ``--max_num_tokens``: maximum number of tokens. Default is None.
   - ``--opt_num_tokens``: optimum number of tokens. Default is None.
   - ``--ptuning_nemo_checkpoint``: source .nemo file for prompt embeddings table.
   - ``--task_ids``: unique task names for the prompt embedding.
   - ``--max_prompt_embedding_table_size``: max prompt embedding table size.
   - ``--lora_ckpt``: a checkpoint list of LoRA weights.
   - ``--use_lora_plugin``: activates the LoRA plugin which enables embedding sharing.
   - ``--lora_target_modules``: adds LoRA to specified modules, but only activates when the ``--use_lora_plugin`` is enabled.
   - ``--max_lora_rank``: maximum LoRA rank for different LoRA modules. It is used to compute the workspace size of the LoRA plugin.
   - ``--no_paged_kv_cache``: disables the paged kv cache in the TensorRT-LLM.
   - ``--disable_remove_input_padding``: disables the remove input padding option of TensorRT-LLM.
   - ``--use_parallel_embedding``: enables the parallel embedding feature of TensorRT-LLM.
   - ``--export_fp8_quantized``: manually overrides the FP8 quantization settings.
   - ``--use_fp8_kv_cache``: manually overrides the FP8 KV-cache quantization settings.


   **Note:** The parameters described here are generalized and should be compatible with any NeMo checkpoint. It is important, however, that you check the LLM model table in the main :ref:`Deploy NeMo LLM page <deploy-nemo-framework-models-llm>` for optimized inference model compatibility. We are actively working on extending support to other checkpoints.

3. To export and deploy a different model such as Llama3, Mixtral, or Starcoder, change the *model_type* in `deploy_triton.py <https://github.com/NVIDIA/NeMo/blob/main/scripts/deploy/nlp/deploy_triton.py>`_ script. Please see the table below to learn more about which *model_type* is used for a LLM model (not required for *.qnemo* models).

   |Model Name | model_type  | 
   |:--------- |-------------|
   |GPT        | gpt         |          
   |Nemotron   | gpt         |                
   |Llama 2    | llama       |
   |Llama 3    | llama       |
   |Llama 3.1  | llama       | 
   |Llama 3.2  | llama       |
   |Gemma      | gemma       |   
   |StarCoder1 | starcoder   |                     
   |StarCoder2 | starcoder   |          
   |Mistral    | llama       |        
   |Mixtral    | mixtral     |     
   

4. Whenever the script is executed, it initiates the service by exporting the NeMo checkpoint to the TensorRT-LLM. If you want to skip the exporting step in the optimized inference option, you can specify an empty directory to save the TensorRT-LLM engine produced. Stop the running container and then run the following command to specify an empty directory:

   ```shell
   mkdir tmp_triton_model_repository

   docker run --gpus all -it --rm --shm-size=4g -p 8000:8000 \
       -v ${PWD}:/opt/checkpoints/ \
       -w /opt/NeMo \
       nvcr.io/nvidia/nemo:vr

   python scripts/deploy/nlp/deploy_triton.py \
       --nemo_checkpoint /opt/checkpoints/hf_llama31_8B_nemo2.nemo \
       --model_type llama \
       --triton_model_name llama \
       --triton_model_repository /opt/checkpoints/tmp_triton_model_repository \
       --tensor_parallelism_size 1
   ```

   The checkpoint will be exported to the specified folder after executing the script mentioned above so that it can be reused later.

5. To load the exported model directly, run the following script within the container:

   ```shell
   python scripts/deploy/nlp/deploy_triton.py \
       --triton_model_name llama \
       --triton_model_repository /opt/checkpoints/tmp_triton_model_repository \
       --model_type llama
   ```

6. Access the models with a Hugging Face token.

   If you want to run inference using the StarCoder1, StarCoder2, or LLama3 models, you'll need to generate a Hugging Face token that has access to these models. Visit `Hugging Face <https://huggingface.co/>`__ for more information. After you have the token, perform one of the following steps.

   - Log in to Hugging Face:

   ```shell
   huggingface-cli login
   ```

   - Or, set the HF_TOKEN environment variable:

   ```shell
   export HF_TOKEN=your_token_here
   ```


### Use Prompt Embedding Tables

You can use learned virtual tokens to perform a downstream stream task during inference. Once the virtual tokens are learned using the NeMo Framework training container, all the tokens are saved in a .nemo file. You can feed this file into the script as shown in the following command. Since there is no NeMo checkpoint specifically for the virtual token available on NVIDIA NGC or Hugging Face, you’ll need to find or generate a checkpoint.

1. Assuming there is a checkpoint for the prompt embedding table, run the following command:

   ```shell
   python scripts/deploy/nlp/deploy_triton.py \
       --nemo_checkpoint /opt/checkpoints/hf_llama31_8B_nemo2.nemo \
       --model_type llama \
       --triton_model_name llama \
       --triton_model_repository /opt/checkpoints/tmp_triton_model_repository \
       --max_prompt_embedding_table_size 1024 \
       --ptuning_nemo_checkpoint /opt/checkpoints/my_ptuning_table.nemo \
       --task_ids "task 1" \
       --tensor_parallelism_size 1
   ```

   ``max_prompt_embedding_table_size`` parameter should be set as the total number of virtual tokens for all of the downstream tasks.

2. To pass multiple NeMo checkpoints, run the following command:

   ```shell
   python scripts/deploy/nlp/deploy_triton.py \
       --nemo_checkpoint /opt/checkpoints/hf_llama31_8B_nemo2.nemo \
       --model_type llama \
       --triton_model_name llama \
       --triton_model_repository /opt/checkpoints/tmp_triton_model_repository \
       --max_prompt_embedding_table_size 1024 \
       --ptuning_nemo_checkpoint /opt/checkpoints/my_ptuning_table-1.nemo /opt/checkpoints/my_ptuning_table-2.nemo \
       --task_ids "task 1" "task 2" \
       --tensor_parallelism_size 1
   ```

   Please ensure that the combined total number of virtual tokens in `my_ptuning_table-1.nemo` and `my_ptuning_table-2.nemo` doesn't exceed the `max_prompt_embedding_table_size` parameter.


## Use NeMo Export and Deploy Module APIs to Run Inference

Up until now, we have used scripts for exporting and deploying LLM models. However, NeMo’s deploy and export modules offer straightforward APIs for deploying models to Triton and exporting NeMo checkpoints to TensorRT-LLM.

### Export a LLM Model to TensorRT-LLM

You can use the APIs in the export module to export a NeMo checkpoint to TensorRT-LLM. The following code example assumes the ``hf_llama31_8B_nemo2.nemo`` checkpoint has already been downloaded and mounted to the ``/opt/checkpoints/`` path. Additionally, the ``/opt/checkpoints/tmp_trt_llm`` path is also assumed to exist.

1. Run the following command:

   ```{python}
   from nemo.export.tensorrt_llm import TensorRTLLM

   trt_llm_exporter = TensorRTLLM(model_dir="/opt/checkpoints/tmp_trt_llm/")
   trt_llm_exporter.export(
       nemo_checkpoint_path="/opt/checkpoints/hf_llama31_8B_nemo2.nemo",
       model_type="llama",
       tensor_parallelism_size=1,
   )
   
   trt_llm_exporter.forward(
       ["What is the best city in the world?"],
       max_output_token=15,
       top_k=1,
       top_p=0.0,
       temperature=1.0,
   )
   ```

2. Be sure to check the TensorRTLLM class docstrings for details.

3. For quantized *qnemo* models -- see [Quantization](https://docs.nvidia.com/nemo-framework/user-guide/latest/model-optimization/quantization/quantization.html) -- the same ``TensorRTLLM`` class can be used. In this case, specifying ``model_type`` is not necessary. Alternatively, advanced users can build engines directly using ``trtllm-build`` command, see [TensorRT-LLM documentation](https://github.com/NVIDIA/TensorRT-LLM/tree/main/examples/llama#fp8-post-training-quantization)

   ```shell
   trtllm-build \
       --checkpoint_dir /opt/checkpoints/llama3-70b-base-fp8-qnemo \
       --output_dir /path/to/trt_llm_engine_folder \
       --gemm_plugin fp8 \
       --max_batch_size 8 \
       --max_input_len 2048 \
       --max_seq_len 2560 \
       --workers 2
   ```


### Deploy a LLM Model to TensorRT-LLM

You can use the APIs in the deploy module to deploy a TensorRT-LLM model to Triton. The following code example assumes the ``hf_llama31_8B_nemo2.nemo`` checkpoint has already been downloaded and mounted to the ``/opt/checkpoints/`` path. Additionally, the ``/opt/checkpoints/tmp_trt_llm`` path is also assumed to exist.

1. Run the following command:

   ```{python}
   from nemo.export.tensorrt_llm import TensorRTLLM
   from nemo.deploy import DeployPyTriton

   trt_llm_exporter = TensorRTLLM(model_dir="/opt/checkpoints/tmp_trt_llm/")
   trt_llm_exporter.export(
       nemo_checkpoint_path="/opt/checkpoints/hf_llama31_8B_nemo2.nemo",
       model_type="llama",
       tensor_parallelism_size=1,
   )

   nm = DeployPyTriton(model=trt_llm_exporter, triton_model_name="llama", http_port=8000)
   nm.deploy()
   nm.serve()
   ```

2. The same instructions apply to quantized checkpoints.

### Direct TensorRT-LLM Export for FP8-trained Models

If you have a FP8-trained checkpoint, produced during pre-training or fine-tuning with NVIDIA Transformer Engine, you can convert it to a FP8 TensorRT-LLM engine directly using ``nemo.export``. The entry point is the same as with regular *.nemo* and *.qnemo* checkpoints:

```{python}
from nemo.export.tensorrt_llm import TensorRTLLM

trt_llm_exporter = TensorRTLLM(model_dir="/opt/checkpoints/tmp_trt_llm/")
trt_llm_exporter.export(
    nemo_checkpoint_path="/opt/checkpoints/llama2-7b-base-fp8.nemo",
    model_type="llama",
)
trt_llm_exporter.forward(["Hi, how are you?", "I am good, thanks, how about you?"])
```

The export settings for quantization can be adjusted via ``trt_llm_exporter.export`` arguments:

* ``fp8_quantized: Optional[bool] = None``: enables/disables FP8 quantization
* ``fp8_kvcache: Optional[bool] = None``: enables/disables FP8 quantization for KV-cache

By default, the quantization settings are auto-detected from the NeMo checkpoint.
