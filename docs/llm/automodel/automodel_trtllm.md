
# Deploy Hugging Face Models by Exporting to TensorRT-LLM 

This section shows how to use scripts and APIs to export a Hugging Face model to TensorRT-LLM, and deploy it with the NVIDIA Triton Inference Server.


## Quick Example

1. Pull down and run the Docker container image using the command shown below. Change the ``:vr`` tag to the version of the container you want to use:

   ```shell
   docker pull nvcr.io/nvidia/nemo:vr

   docker run --gpus all -it --rm --shm-size=4g -p 8000:8000 \
      -w /opt/NeMo \
      nvcr.io/nvidia/nemo:vr
   ```

2. Run the following deployment script to verify that everything is working correctly. The script exports the Hugging Face model to TensorRT-LLM and subsequently serves it on the Triton server:

   ```shell
   python scripts/deploy/nlp/deploy_triton.py \
      --hf_model_id_path meta-llama/Meta-Llama-3-8B-Instruct \
      --model_type LlamaForCausalLM \
      --triton_model_name llama \
      --tensor_parallelism_size 1
   ```

3. If the test yields a shared memory-related error, increase the shared memory size using ``--shm-size`` (gradually by 50%, for example).

4. In a separate terminal, run the following command to get the container ID of the running container:

   ```shell
   docker ps
   ```

5. Access the running container and replace ``container_id`` with the actual container ID as follows:

   ```shell
   docker exec -it container_id bash
   ```

6. To send a query to the Triton server, run the following script:

   ```shell
   python scripts/deploy/nlp/query.py -mn llama -p "What is the color of a banana?" -mol 5
   ```

## Use a Script to Deploy Hugging Face Models on a Triton Server

You can deploy a Hugging Face model on Triton using the provided script.

### Export and Deploy a Hugging Face Model 

After executing the script, it will export the model to TensorRT-LLM and then initiate the service on Triton.

1. Start the container using the steps described in the **Quick Example** section.

2. If you want to deploy a model that needs to be downloaded from Hugging Face, you need to generate a Hugging Face token that has access to these models. Visit `Hugging Face <https://huggingface.co/>`__ for more information. After you have the token, perform one of the following steps.

   - Log in to Hugging Face:

   ```shell
   huggingface-cli login
   ```

   - Or, set the HF_TOKEN environment variable:

   ```shell
   export HF_TOKEN=your_token_here
   ```

   **Note: **If you're using a locally downloaded model, you don't need to provide a Hugging Face token unless the model requires it for downloading additional resources.

2. To begin serving a Hugging Face model, you can use either a model ID from the Hugging Face hub or a path to a locally downloaded model:

   a. Using a Hugging Face model ID:

   ```shell

   python scripts/deploy/nlp/deploy_triton.py \
      --hf_model_id_path meta-llama/Meta-Llama-3-8B-Instruct \
      --model_type LlamaForCausalLM \
      --triton_model_name llama \
      --tensor_parallelism_size 1
   ```

   b. To use a locally downloaded model:

   ```shell
   python scripts/deploy/nlp/deploy_triton.py \
      --hf_model_id_path /path/to/your/local/model \
      --model_type LlamaForCausalLM \
      --triton_model_name llama \
      --tensor_parallelism_size 1
   ```

   The following parameters are defined in the ``deploy_triton.py`` script:

   - ``--hf_model_id_path``: path or identifier of the Hugging Face model. This can be either:
     - A Hugging Face model ID (e.g., "meta-llama/Meta-Llama-3-8B-Instruct")
     - A local path to a downloaded model directory (e.g., "/path/to/your/local/model")
   - ``--model_type``: type of the model. See the table below for supported model types.
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

3. The following table shows the supported Hugging Face model types and their corresponding ``model_type`` values:

   
   | Hugging Face Model   |              model_type       |  
   | :------------------- | ------------------------------|
   | :GPT2LMHeadModel     |              GPTForCausalLM    |
   | :GPT2LMHeadCustomModel|             GPTForCausalLM|
   | :GPTBigCodeForCausalLM |            GPTForCausalLM|
   | :Starcoder2ForCausalLM |            GPTForCausalLM|
   | :JAISLMHeadModel       |            GPTForCausalLM|
   | :GPTForCausalLM       |             GPTForCausalLM|
   | :NemotronForCausalLM   |            GPTForCausalLM|
   | :OPTForCausalLM       |             OPTForCausalLM|
   | :BloomForCausalLM     |             BloomForCausalLM|
   | :RWForCausalLM        |             FalconForCausalLM|
   | :FalconForCausalLM    |             FalconForCausalLM|
   | :PhiForCausalLM       |             PhiForCausalLM|
   | :Phi3ForCausalLM      |             Phi3ForCausalLM|
   | :Phi3VForCausalLM     |             Phi3ForCausalLM|
   | :Phi3SmallForCausalLM |             Phi3ForCausalLM|
   | :PhiMoEForCausalLM   |              Phi3ForCausalLM|
   | :MambaForCausalLM     |             MambaForCausalLM|
   | :GPTNeoXForCausalLM  |              GPTNeoXForCausalLM|
   | :GPTJForCausalLM     |              GPTJForCausalLM|
   | :MptForCausalLM      |              MPTForCausalLM|
   | :MPTForCausalLM      |              MPTForCausalLM|
   | :GLMModel             |             ChatGLMForCausalLM|
   | :ChatGLMModel         |             ChatGLMForCausalLM|
   | :ChatGLMForCausalLM   |             ChatGLMForCausalLM|
   | :ChatGLMForConditionalGeneration|   ChatGLMForCausalLM|
   | :LlamaForCausalLM   |               LLaMAForCausalLM|
   | :LlavaLlamaModel     |              LLaMAForCausalLM|
   | :ExaoneForCausalLM    |             LLaMAForCausalLM|
   | :MistralForCausalLM    |            LLaMAForCausalLM|
   | :MixtralForCausalLM     |           LLaMAForCausalLM|
   | :ArcticForCausalLM     |            LLaMAForCausalLM|
   | :Grok1ModelForCausalLM |            GrokForCausalLM|
   | :InternLMForCausalLM   |            LLaMAForCausalLM|
   | :InternLM2ForCausalLM   |           LLaMAForCausalLM|
   | :InternLMXComposer2ForCausalLM |    LLaMAForCausalLM|
   | :GraniteForCausalLM   |             LLaMAForCausalLM|
   | :GraniteMoeForCausalLM |            LLaMAForCausalLM|
   | :MedusaForCausalLM    |             MedusaForCausalLm|
   | :MedusaLlamaForCausalLM |           MedusaForCausalLm|
   | :ReDrafterForCausalLM   |           ReDrafterForCausalLM|
   | :BaichuanForCausalLM   |            BaichuanForCausalLM|
   | :BaiChuanForCausalLM   |            BaichuanForCausalLM|
   | :SkyworkForCausalLM    |            LLaMAForCausalLM|
   | :GEMMA                 |            GemmaForCausalLM|
   | :GEMMA2                |            GemmaForCausalLM|
   | :QWenLMHeadModel        |           QWenForCausalLM|
   | :QWenForCausalLM        |           QWenForCausalLM|
   | :Qwen2ForCausalLM       |           QWenForCausalLM|
   | :Qwen2MoeForCausalLM    |           QWenForCausalLM|
   | :Qwen2ForSequenceClassification |   QWenForCausalLM|
   | :Qwen2VLForConditionalGeneration|   QWenForCausalLM|
   | :Qwen2VLModel        |              QWenForCausalLM|
   | :WhisperEncoder      |              WhisperEncoder|
   | :EncoderModel         |             EncoderModel|
   | :DecoderModel         |             DecoderModel|
   | :DbrxForCausalLM      |             DbrxForCausalLM|
   | :RecurrentGemmaForCausalLM |        RecurrentGemmaForCausalLM|
   | :CogVLMForCausalLM      |           CogVLMForCausalLM|
   | :DiT                  |             DiT|
   | :DeepseekForCausalLM   |            DeepseekForCausalLM|
   | :DeciLMForCausalLM     |            DeciLMForCausalLM|
   | :DeepseekV2ForCausalLM  |           DeepseekV2ForCausalLM|
   | :EagleForCausalLM       |           EagleForCausalLM|
   | :CohereForCausalLM       |          CohereForCausalLM|
   | :MLLaMAModel             |          MLLaMAForCausalLM|
   | :MllamaForConditionalGeneration |   MLLaMAForCausalLM|
   | :BertForQuestionAnswering      |    BertForQuestionAnswering|
   | :BertForSequenceClassification |    BertForSequenceClassification|
   | :BertModel                    |     BertModel|
   | :RobertaModel                 |     RobertaModel|
   | :RobertaForQuestionAnswering  |     RobertaForQuestionAnswering|
   | :RobertaForSequenceClassification | RobertaForSequenceClassification|
   
4. Whenever the script is executed, it initiates the service by exporting the Hugging Face model to TensorRT-LLM. If you want to skip the exporting step in the optimized inference option, you can specify an empty directory to save the TensorRT-LLM engine produced. Stop the running container and then run the following command to specify an empty directory:

   ```shell

   mkdir tmp_triton_model_repository

   docker run --gpus all -it --rm --shm-size=4g -p 8000:8000 \
       -v ${PWD}:/opt/checkpoints/ \
       -w /opt/NeMo \
       nvcr.io/nvidia/nemo:vr

   python scripts/deploy/nlp/deploy_triton.py \
       --hf_model_id_path /path/to/your/local/model \
       --model_type LlamaForCausalLM \
       --triton_model_name llama \
       --triton_model_repository /opt/checkpoints/tmp_triton_model_repository \
       --tensor_parallelism_size 1
   ```

   The model will be exported to the specified folder after executing the script mentioned above so that it can be reused later.

5. To load the exported model directly, run the following script within the container:

   ```shell
   python scripts/deploy/nlp/deploy_triton.py \
       --triton_model_name llama \
       --triton_model_repository /opt/checkpoints/tmp_triton_model_repository \
       --model_type LlamaForCausalLM
   ```

## Use NeMo Export and Deploy Module APIs to Run Inference

Up until now, we have used scripts for exporting and deploying Hugging Face models. However, NeMo's deploy and export modules offer straightforward APIs for deploying models to Triton and exporting Hugging Face models to TensorRT-LLM.

### Export a Hugging Face Model to TensorRT-LLM

You can use the APIs in the export module to export a Hugging Face model to TensorRT-LLM. The following code example assumes the ``/opt/checkpoints/tmp_trt_llm`` path exists.

1. Run the following command:

   ```{python}

   from nemo.export.tensorrt_llm import TensorRTLLM

   trt_llm_exporter = TensorRTLLM(model_dir="/opt/checkpoints/tmp_trt_llm/")
   # Using a Hugging Face model ID
   trt_llm_exporter.export_hf_model(
       hf_model_path="meta-llama/Meta-Llama-3-8B-Instruct",
       model_type="LlamaForCausalLM",
       tensor_parallelism_size=1,
   )
   # Or using a local model path
   
   trt_llm_exporter.export_hf_model(
       hf_model_path="/path/to/your/local/model",
       model_type="LlamaForCausalLM",
       tensor_parallelism_size=1,
   }
      
   trt_llm_exporter.forward(
       ["What is the best city in the world?"],
       max_output_token=15,
       top_k=1,
       top_p=0.0,
       temperature=1.0,
   )
    ```

2. Be sure to check the TensorRTLLM class docstrings for details.

### Deploy a Hugging Face Model to TensorRT-LLM

You can use the APIs in the deploy module to deploy a TensorRT-LLM model to Triton. The following code example assumes the ``/opt/checkpoints/tmp_trt_llm`` path exists.

1. Run the following command:

   ```{python}
   from nemo.export.tensorrt_llm import TensorRTLLM
   from nemo.deploy import DeployPyTriton

   trt_llm_exporter = TensorRTLLM(model_dir="/opt/checkpoints/tmp_trt_llm/")
   # Using a Hugging Face model ID
   trt_llm_exporter.export_hf_model(
       hf_model_path="meta-llama/Llama-2-7b-hf",
       model_type="LlamaForCausalLM",
       tensor_parallelism_size=1,
   )
   # Or using a local model path
   
   trt_llm_exporter.export_hf_model(
       hf_model_path="/path/to/your/local/model",
       model_type="LlamaForCausalLM",
       tensor_parallelism_size=1,
   )

   nm = DeployPyTriton(model=trt_llm_exporter, triton_model_name="llama", http_port=8000)
   nm.deploy()
   nm.serve() 
   ```