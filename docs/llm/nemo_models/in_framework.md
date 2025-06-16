# Deploy NeMo Models in the Framework

This section demonstrates how to deploy PyTorch-level NeMo LLMs within the framework (referred to as ‘In-Framework’) using the NVIDIA Triton Inference Server.

## Quick Example

1. Follow the steps in the [Deploy NeMo LLM main page](../index.md) to generate a NeMo 2.0 Llama checkpoint.

2. Pull down and run the Docker container image using the command shown below. Change the ``:vr`` tag to the version of the container you want to use:

   ```shell
   docker pull nvcr.io/nvidia/nemo:vr

   docker run --gpus all -it --rm --shm-size=4g -p 8000:8000 -v ${PWD}/:/opt/checkpoints/ -w /opt/NeMo nvcr.io/nvidia/nemo:vr
   ```

3. Using a NeMo 2.0 model, run the following deployment script to verify that everything is working correctly. The script directly serves the NeMo 2.0 model on the Triton server:

   ```shell
   python scripts/deploy/nlp/deploy_inframework_triton.py --nemo_checkpoint /opt/checkpoints/hf_llama31_8B_nemo2.nemo --triton_model_name llama
   ```

4. If the test yields a shared memory-related error, increase the shared memory size using ``--shm-size`` (gradually by 50%, for example).

5. In a separate terminal, run the following command to get the container ID of the running container. Please find the ``nvcr.io/nvidia/nemo:vr`` image to get the container ID.

   ```shell
   docker ps
   ```

6. Access the running container and replace ``container_id`` with the actual container ID as follows:

   ```shell
   docker exec -it container_id bash
   ```

7. To send a query to the Triton server, run the following script:

   ```shell
   python scripts/deploy/nlp/query_inframework.py -mn llama -p "What is the color of a banana?" -mol 5
   ```

Please note that only NeMo 2.0 checkpoints are supported by the In-Framework deployment option.

## Use a Script to Deploy NeMo LLMs on a Triton Server

You can deploy an LLM from a NeMo checkpoint on Triton using the provided script.

### Deploy a NeMo LLM Model

Executing the script will directly deploy the NeMo LLM model and initiate the service on Triton.

1. Start the container using the steps described in the **Quick Example** section.

2. To begin serving the downloaded model, run the following script:

   ```shell
   python scripts/deploy/nlp/deploy_inframework_triton.py --nemo_checkpoint /opt/checkpoints/hf_llama31_8B_nemo2.nemo --triton_model_name llama
   ```

   The following parameters are defined in the ``deploy_inframework_triton.py`` script:

   - ``--nemo_checkpoint``: path of the .nemo or .qnemo checkpoint file.
   - ``--triton_model_name``: name of the model on Triton.
   - ``--triton_model_version``: version of the model. Default is 1.
   - ``--triton_port``: port for the Triton server to listen for requests. Default is 8000.
   - ``--triton_http_address``: HTTP address for the Triton server. Default is 0.0.0.0.
   - ``--num_gpus``: number of GPUs to use for inference. Large models require multi-gpu export. *This parameter is deprecated*.
   - ``--max_batch_size``: maximum batch size of the model. Default is 8.
   - ``--debug_mode``: enables additional debug logging messages from the script

3. To deploy a different model, just change the ``--nemo_checkpoint`` in the [scripts/deploy/nlp/deploy_inframework_triton.py](https://github.com/NVIDIA/Export-Deploy/blob/main/scripts/deploy/nlp/deploy_inframework_triton.py) script.

4. Access the models with a Hugging Face token.

   If you want to run inference using the StarCoder1, StarCoder2, or LLama3 models, you'll need to generate a Hugging Face token that has access to these models. Visit `Hugging Face <https://huggingface.co/>`_ for more information. After you have the token, perform one of the following steps.

   - Log in to Hugging Face:

   ```shell
   huggingface-cli login
   ```

   - Or, set the HF_TOKEN environment variable:

   ```shell
   export HF_TOKEN=your_token_here
   ```
