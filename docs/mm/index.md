# Deploy NeMo Multimodal Models

## Optimized Inference for Multimodal Models using TensorRT

For scenarios requiring optimized performance, NeMo multimodal models can leverage TensorRT. This process involves converting NeMo models into a format compatible with TensorRT using the nemo.export module.

## Supported GPUs
TensorRT-LLM supports NVIDIA DGX H100 and NVIDIA H100 GPUs based on the NVIDIA Hopper, Ada Lovelace, Ampere, Turing, and Volta architectures. Certain specialized deployment paths, such as FP8 quantized models, require hardware with FP8 data type support, like NVIDIA H100 GPUs.

## Supported Models

The following table shows the supported models.

| Model Name   | NeMo Precision | TensorRT Precision |
| :----------  | -------------- |--------------------|
| Neva         | bfloat16       | bfloat16           |
| Video Neva   | bfloat16       | bfloat16           |
| LITA/VITA    | bfloat16       | bfloat16           |
| VILA         | bfloat16       | bfloat16           |
| SALM         | bfloat16       | bfloat16           |


### Access the Models with a Hugging Face Token

If you want to run inference using the LLama3 model, you'll need to generate a Hugging Face token that has access to these models. Visit `Hugging Face <https://huggingface.co/>`_ for more information. After you have the token, perform one of the following steps.

- Log in to Hugging Face:

    ```shell
    huggingface-cli login
    ```

- Or, set the HF_TOKEN environment variable:

    ```shell
    export HF_TOKEN=your_token_here
    ```

### Export and Deploy a NeMo Multimodal Checkpoint to TensorRT

This section provides an example of how to quickly and easily deploy a NeMo checkpoint to TensorRT. Neva will be used as an example model. Please consult the table above for a complete list of supported models.

1. Run the Docker container image using the command shown below. Change the ``:vr`` tag to the version of the container you want to use:

   ```shell
   docker pull nvcr.io/nvidia/nemo:vr
   
   docker run --gpus all -it --rm --shm-size=4g -p 8000:8000 -v /path/to/nemo_neva.nemo:/opt/checkpoints/nemo_neva.nemo -w /opt/NeMo nvcr.io/nvidia/nemo:vr
   ```

2. Run the following deployment script to verify that everything is working correctly. The script exports the downloaded NeMo checkpoint to TensorRT-LLM and subsequently serves it on the Triton server:

   .. code-block:: python

      python scripts/deploy/multimodal/deploy_triton.py --visual_checkpoint /opt/checkpoints/nemo_neva.nemo --model_type neva --llm_model_type llama --triton_model_name neva --modality vision

   If you only want to export the NeMo checkpoint to TensorRT, use the ``examples/multimodal/multimodal_llm/neva/neva_export.py``.

3. If the test yields a shared memory-related error, increase the shared memory size using ``--shm-size`` (gradually by 50%, for example).

4. In a separate terminal, run the following command to get the container ID of the running container. Please find the ``nvcr.io/nvidia/nemo:24.vr`` image to get the container ID.

   ```shell
   docker ps
   ```

5. Access the running container and replace ``container_id`` with the actual container ID as follows:

   ```shell
   docker exec -it container_id bash
   ``` 

6. To send a query to the Triton server, run the following script:

   ```shell
   python scripts/deploy/multimodal/query.py -mn neva -mt=neva -int="What is in this image?" -im=/path/to/image.jpg
   ```
   
7. To export and deploy a different model, such as Video Neva, change the *model_type* and *modality* in the *scripts/deploy/multimodal/deploy_triton.py* script.


### Use a Script to Run Inference on a Triton Server

You can deploy a multimodal model from a NeMo checkpoint on Triton using the provided script. This deployment uses TensorRT to achieve optimized inference.

#### Export and Deploy a Multimodal Model to TensorRT


After executing the script, it will export the model to TensorRT and then initiate the service on Triton.

1. Start the container using the steps described in the previous section.

2. To begin serving the model, run the following script:

   ```shell
   python scripts/deploy/multimodal/deploy_triton.py --visual_checkpoint /opt/checkpoints/nemo_neva.nemo --model_type neva --llm_model_type llama --triton_model_name neva
   ```
   
   The following parameters are defined in the ``deploy_triton.py`` script:

   - ``modality`` - modality of the model. choices=["vision", "audio"]. By default, it is set to "vision".
   - ``visual_checkpoint`` - path of the .nemo of visual model or the path to perception model checkpoint for SALM model
   - ``llm_checkpoint`` - path of .nemo of LLM. Would be set as visual_checkpoint if not provided
   - ``model_type`` - type of the model. choices=["neva", "video-neva", "lita", "vila", "vita", "salm"].
   - ``llm_model_type`` - type of LLM. choices=["gptnext", "gpt", "llama", "falcon", "starcoder", "mixtral", "gemma"].
   - ``triton_model_name`` name of the model on Triton.
   - ``triton_model_version`` - version of the model. Default is 1.
   - ``triton_port`` - port for the Triton server to listen for requests. Default is 8000.
   - ``triton_http_address`` - HTTP address for the Triton server. Default is 0.0.0.0.
   - ``triton_model_repository`` - TensorRT temp folder. Default is ``/tmp/trt_model_dir/``.
   - ``num_gpus`` - number of GPUs to use for inference. Large models require multi-gpu export.
   - ``dtype`` - data type of the model on TensorRT-LLM. Default is "bfloat16". Currently, only "bfloat16" is supported.
   - ``max_input_len`` - maximum input length of the model.
   - ``max_output_len`` - maximum output length of the model.
   - ``max_batch_size`` - maximum batch size of the model.
   - ``max_multimodal_len`` - maximum lenghth of multimodal input
   - ``vision_max_batch_size`` - maximum batch size for input images for vision encoder. Default is 1. For models like LITA and VITA on video inference, this should be set to 256. 

   **NOTE:** The parameters described here are generalized and should be compatible with any NeMo checkpoint. It is important, however, that you check the Supported Models table above for optimized inference model compatibility. We are actively working on extending support to other checkpoints.

   Whenever the script is executed, it initiates the service by exporting the NeMo checkpoint to the TensorRT. If you want to skip the exporting step in the optimized inference option, you can specify an empty directory.

3. To export and deploy a different model, such as Video Neva, change the *model_type* and *modality* in the *scripts/deploy/multimodal/deploy_triton.py* script. Please see the table below to learn more about which *model_type* and *modality* is used for a multimodal model.
 
   | Model Name  | model_type   | modality   |
   | :---------- | ------------ |------------|
   | Neva        |  neva        | vision     |  
   | Video Neva  |  video-neva  | vision     |
   | LITA        |  lita        | vision     |
   | VILA        |  vila        | vision     |
   | VITA        |  vita        | vision     |
   | SALM        |  salm        | audio      |
   

4. Stop the running container and then run the following command to specify an empty directory:

   ```shell
   mkdir tmp_triton_model_repository

   docker run --gpus all -it --rm --shm-size=4g -p 8000:8000 -v ${PWD}:/opt/checkpoints/ -w /opt/NeMo nvcr.io/nvidia/nemo:vr

   python scripts/deploy/multimodal/deploy_triton.py --visual_checkpoint /opt/checkpoints/nemo_neva.nemo --model_type neva --llm_model_type llama --triton_model_name neva --triton_model_repository /opt/checkpoints/tmp_triton_model_repository --modality vision
   ```
   
   The checkpoint will be exported to the specified folder after executing the script mentioned above.

5. To load the exported model directly, run the following script within the container:

   ```shell
   python scripts/deploy/multimodal/deploy_triton.py --triton_model_name neva --triton_model_repository /opt/checkpoints/tmp_triton_model_repository --model_type neva --llm_model_type llama --modality vision
   ```
   
#### Send a Query


After starting the service using the provided scripts from the previous section, it will wait for incoming requests. You can send a query to this service in several ways.

* Use the Query Script: Execute the query script within the currently running container.
* PyTriton: Utilize PyTriton to send requests directly.
* HTTP Requests: Make HTTP requests using various tools or libraries.

The following example shows how to execute the query script within the currently running container.

1. To use a query script, run the following command. For VILA/LITA/VITA models, the input_text should add ``<image>\n`` before the actual text, such as ``<image>\n What is in this image?``:

   ```shell
   python scripts/deploy/multimodal/query.py --url "http://localhost:8000" --model_name neva --model_type neva --input_text "What is in this image?" --input_media /path/to/image.jpg
   ```
   
2. Change the url and the ``model_name`` based on your server and the model name of your service. The code in the script can be used as a basis for your client code as well. ``input_media`` is the path to the image or audio file you want to use as input. 


### Use NeMo Export and Deploy Module APIs to Run Inference

Up until now, we have used scripts for exporting and deploying Multimodal models. However, NeMo’s deploy and export modules offer straightforward APIs for deploying models to Triton and exporting NeMo checkpoints to TensorRT.


#### Export a Multimodal Model to TensorRT

You can use the APIs in the export module to export a NeMo checkpoint to TensorRT-LLM. The following code example assumes the ``nemo_neva.nemo`` checkpoint has already mounted to the ``/opt/checkpoints/`` path. Additionally, the ``/opt/data/image.jpg`` is also assumed to exist.

1. Run the following command:

   ```python
   from nemo.export.tensorrt_mm_exporter import TensorRTMMExporter

   exporter = TensorRTMMExporter(model_dir="/opt/checkpoints/tmp_triton_model_repository/", modality="vision")
   exporter.export(visual_checkpoint_path="/opt/checkpoints/nemo_neva.nemo", model_type="neva", llm_model_type="llama", tensor_parallel_size=1)
   output = exporter.forward("What is in this image?", "/opt/data/image.jpg", max_output_token=30, top_k=1, top_p=0.0, temperature=1.0)
   print("output: ", output)
   ```

2. Be sure to check the TensorRTMMExporter class docstrings for details.


#### Deploy a Multimodal Model to TensorRT

You can use the APIs in the deploy module to deploy a TensorRT-LLM model to Triton. The following code example assumes the ``nemo_neva.nemo`` checkpoint has already mounted to the ``/opt/checkpoints/`` path.

1. Run the following command:

   ```python
   from nemo.export.tensorrt_mm_exporter import TensorRTMMExporter
   from nemo.deploy import DeployPyTriton

   exporter = TensorRTMMExporter(model_dir="/opt/checkpoints/tmp_triton_model_repository/", modality="vision")
   exporter.export(visual_checkpoint_path="/opt/checkpoints/nemo_neva.nemo", model_type="neva", llm_model_type="llama", tensor_parallel_size=1)

   nm = DeployPyTriton(model=exporter, triton_model_name="neva", port=8000)
   nm.deploy()
   nm.serve()
   ```

#### Send a Query

The NeMo Framework provides NemoQueryMultimodal APIs to send a query to the Triton server for convenience. These APIs are only accessible from the NeMo Framework container.

1. To run the request example using NeMo APIs, run the following command:

   ```python
   from nemo.deploy.multimodal import NemoQueryMultimodal

   nq = NemoQueryMultimodal(url="localhost:8000", model_name="neva", model_type="neva")
   output = nq.query(input_text="What is in this image?", input_media="/opt/data/image.jpg", max_output_len=30, top_k=1, top_p=0.0, temperature=1.0)
   print(output)
   ```

2. Change the url and the ``model_name`` based on your server and the model name of your service. Please check the NemoQueryMultimodal docstrings for details.

#### Other Examples

For a comprehensive guide on exporting a speech language model like SALM (to obtain a perception model and merge LoRA weights), please refer to [this document](https://github.com/NVIDIA/NeMo/tree/main/examples/multimodal/speech_llm/export)
