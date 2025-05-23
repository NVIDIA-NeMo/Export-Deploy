
# Deploy NeMo Models by Exporting vLLM

This section shows how to use scripts and APIs to export a NeMo LLM to vLLM and deploy it with the NVIDIA Triton Inference Server.

## Quick Example

1. Follow the steps in the [Deploy NeMo LLM main page](../../index.md) to generate a NeMo 2.0 Llama checkpoint.

2. In a terminal, go to the folder where the ``hf_llama31_8B_nemo2.nemo`` file is located. Pull down and run the Docker container image using the command shown below. Change the ``:vr`` tag to the version of the container you want to use:

   ```shell
   docker pull nvcr.io/nvidia/nemo:vr

   docker run --gpus all -it --rm --shm-size=4g -p 8000:8000 -v ${PWD}/hf_llama31_8B_nemo2.nemo:/opt/checkpoints/hf_llama31_8B_nemo2.nemo -w /opt/NeMo nvcr.io/nvidia/nemo:vr
   ```

3. In the container, activate the virtual environment (venv) that contains the vLLM installation.

   ```shell
   source /opt/venv/bin/activate
   ```

4. Run the following deployment script to verify that everything is working correctly. The script exports the Llama NeMo checkpoint to vLLM and subsequently serves it on the Triton server:

   ```shell
   python scripts/deploy/nlp/deploy_vllm_triton.py \
       --nemo_checkpoint /opt/checkpoints/hf_llama31_8B_nemo2.nemo \
       --model_type llama \
       --triton_model_name llama \
       --tensor_parallelism_size 1
   ```

5. In a separate terminal, run the following command to get the container ID of the running container. Please access the ``nvcr.io/nvidia/nemo:vr`` image to get the container ID.

   ```shell
   docker ps
   ```

6. Access the running container and replace ``container_id`` with the actual container ID in the below command.

   ```shell
   docker exec -it container_id bash
   ```

7. To send a query to the Triton server, run the following script:

   ```shell
   python scripts/deploy/nlp/query.py -mn llama -p "The capital of Canada is" -mol 50
   ```

8. To export and deploy a different model such as Llama3, Mixtral, or Starcoder, change the *model_type* in the *scripts/deploy/nlp/deploy_vllm_triton.py* script. Please check below to see the list of the model types.


## Use a Script to Deploy NeMo LLMs on a Triton Server

You can deploy a LLM from a NeMo checkpoint on Triton using the provided script.

### Export and Deploy a LLM Model

After executing the script, it will export the model to vLLM and then initiate the service on Triton.

1. Start the container using the steps described in the **Quick Example** section.

2. To begin serving the downloaded model, run the following script:

   ```shell
   python scripts/deploy/nlp/deploy_vllm_triton.py \
       --nemo_checkpoint /opt/checkpoints/hf_llama31_8B_nemo2.nemo \
       --model_type llama \
       --triton_model_name llama \
       --tensor_parallelism_size 1
   ```

   The following parameters are defined in the ``deploy_vllm_triton.py`` script:

   - ``--nemo_checkpoint``: path of the .nemo or .qnemo checkpoint file.
   - ``--model_type``: type of the model. Can be "llama", "mistral", "mixtral", "starcoder2", "gemma".
   - ``--triton_model_name``: name of the model on Triton.
   - ``--triton_model_version``: version of the model. Default is 1.
   - ``--triton_port``: port for the Triton server to listen for requests. Default is 8000.
   - ``--triton_http_address``: HTTP address for the Triton server. Default is 0.0.0.0.
   - ``--triton_model_repository``: Temporary folder for weight conversion. Default is a new folder in /tmp/.
   - ``--tensor_parallelism_size``: Number of GPUs to split the tensors for tensor parallelism. Default is 1.
   - ``--dtype``: data type of the deployed model. Default is "bfloat16".
   - ``--max_model_len``: maximum input + output length of the model. Default is 512. 
   - ``--max_batch_size``: maximum batch size of the model. Default is 8. 
   - ``--debug_mode``: enables more verbose output. 
   - ``--weight_storage``: strategy for storing converted weights for vLLM. Can be "auto", "cache", "file", "memory". Use ``--help`` for more information.
   
   **Note:** The parameters described here are generalized and should be compatible with any NeMo checkpoint. It is important, however, that you check the LLM model table in the main [Deploy NeMo LLM main page](../../index.md) for optimized inference model compatibility. We are actively working on extending support to other checkpoints.

3. To export and deploy a different model such as Llama3, Mixtral, and Starcoder, change the *model_type* parameter in the *scripts/deploy/nlp/deploy_vllm_triton.py* script. Please see the table below to learn more about which *model_type* is used for a LLM model.
 
   |Model Name| model_type   |
   |:---------|--------------|
   |Llama 2   | llama        |
   |Llama 3   | llama        |
   |Gemma     | gemma        |
   |StarCoder2| starcoder2   |
   |Mistral   | mistral      |
   |Mixtral   | mixtral      | 
   
4. Export faster by caching weights.

   Whenever the deployment script is executed, it initiates the service by exporting the NeMo checkpoint to vLLM, which includes converting weights to a compatible format. By default, for a single-GPU use case, the conversion happens in-memory and is quick. For multiple GPUs, the conversion happens through a temporary file, and there is an option to keep that file between runs for quicker deployment. To do that, you'll need to create an empty directory and make it available to the deployment script.

   Stop the running container and then run the following command to specify an empty directory:

   ```shell
   mkdir tmp_triton_model_repository

   docker run --gpus all -it --rm --shm-size=4g -p 8000:8000 -v ${PWD}:/opt/checkpoints/ -w /opt/NeMo nvcr.io/nvidia/nemo:vr

   python scripts/deploy/nlp/deploy_vllm_triton.py \
       --nemo_checkpoint /opt/checkpoints/hf_llama31_8B_nemo2.nemo \
       --model_type llama \
       --triton_model_name llama \
       --triton_model_repository /opt/checkpoints/tmp_triton_model_repository \
       --weight_storage cache \
       --tensor_parallelism_size 1
   ```
   
   The ``--weight_storage cache`` setting indicates that weights will be converted through a file in the directory specified by ``--triton_model_repository``. This file will only be overwritten if it’s older than the input .nemo file.

5. Access the models with a Hugging Face token.

   If you want to run inference using the StarCoder1, StarCoder2, or LLama3 models, you'll need to generate a Hugging Face token that has access to these models. Visit `Hugging Face <https://huggingface.co/>`__ for more information. After you have the token, perform one of the following steps.

   - Log in to Hugging Face:

   ```shell
   huggingface-cli login
   ```

   - Or, set the HF_TOKEN environment variable:

   ```shell
   export HF_TOKEN=your_token_here
   ```

## Use NeMo Export and Deploy Module APIs to Run Inference

Up until now, we have used scripts for exporting and deploying LLM models. However, NeMo's deploy and export modules offer straightforward APIs for deploying models to Triton and exporting NeMo checkpoints to vLLM.


### Export an LLM Model to vLLM

You can use the APIs in the export module to export a NeMo checkpoint to vLLM. The following code example assumes the ``hf_llama31_8B_nemo2.nemo`` checkpoint has already been downloaded and mounted to the ``/opt/checkpoints/`` path.

```{python}
import os
from nemo.export.vllm_exporter import vLLMExporter

checkpoint_file = "/opt/checkpoints/hf_llama31_8B_nemo2.nemo"
model_dir = "/opt/checkpoints/hf_llama31_8B_nemo2.nemo/vllm_export"

# Export the checkpoint to vLLM, prepare for inference
exporter = vLLMExporter()
exporter.export(
    nemo_checkpoint=checkpoint_file,
    model_dir=model_dir,
    model_type="llama",
)
   
# Run inference and print the output
output = exporter.forward(["What is the best city in the world?"], max_output_len=50, top_k=1, top_p=0.0, temperature=1.0)
print("output: ", output)
```

Be sure to check the vLLMExporter class docstrings for details.


### Deploy an LLM Model on the Triton Server using vLLM

You can use the APIs in the deploy module to deploy a vLLM model to Triton. Use the Export example above to export the model to vLLM first, just drop the ``forward`` and ``print`` calls at the end. Then initialize the Triton server and serve the model:

```{python}
from nemo.deploy import DeployPyTriton

nm = DeployPyTriton(model=exporter, triton_model_name="llama", http_port=8000)
nm.deploy()
nm.serve()
```

