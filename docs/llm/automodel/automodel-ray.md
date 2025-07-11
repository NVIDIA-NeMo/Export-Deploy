# Deploy NeMo AutoModel LLM Models using Ray

This section demonstrates how to deploy NeMo AutoModel LLM Models using Ray Serve (referred to as 'Ray for AutoModel LLM'). Ray deployment support has been added in addition to Triton, to support single node multi-instance deployment. Ray Serve provides a scalable and flexible platform for deploying machine learning models, offering features such as automatic scaling, load balancing, and multi-replica deployment.

## Quick Example

1. If you need access to the Llama-3.2-1B model, visit the [Llama 3.2 Hugging Face page](https://huggingface.co/meta-llama/Llama-3.2-1B) to request access.

2. Pull and run the Docker container image. Replace ``:vr`` with your desired version:

   ```shell
   docker pull nvcr.io/nvidia/nemo:vr

   docker run --gpus all -it --rm \
       --shm-size=4g \
       -p 1024:1024 \
       -v ${PWD}/:/opt/checkpoints/ \
       -w /opt/Export-Deploy \
       nvcr.io/nvidia/nemo:vr
   ``` 

3. Log in to Hugging Face with your access token:

   ```shell
   huggingface-cli login
   ```

4. Deploy the model to Ray:

   ```python
   python scripts/deploy/nlp/deploy_ray_hf.py \
      --model_path meta-llama/Llama-3.2-1B \
      --model_id llama \
      --num_replicas 2 \
      --num_gpus 2 \
      --num_gpus_per_replica 1 \
      --cuda_visible_devices "0,1"
   ```
   
   **Note:** If you encounter shared memory errors, increase ``--shm-size`` gradually by 50%.

5. In a new terminal, get the container ID:

   ```shell
   docker ps
   ```

6. Access the container:

   ```shell
   docker exec -it <container_id> bash
   ```

7. Test the deployed model:

   ```shell
   python scripts/deploy/nlp/query_ray_deployment.py \
      --model_id llama \
      --host 0.0.0.0 \
      --port 1024
   ```

## Detailed Deployment Guide

### Deploy a NeMo AutoModel LLM Model

Follow these steps to deploy your model on Ray Serve:

1. Start the container as shown in the **Quick Example** section.

2. Deploy your model:

   ```shell
   python scripts/deploy/nlp/deploy_ray_hf.py \
      --model_path meta-llama/Llama-3.2-1B \
      --model_id llama \
      --num_replicas 2 \
      --num_gpus 2 \
      --num_gpus_per_replica 1 \
      --cuda_visible_devices "0,1"
   ```

   Available Parameters:
   
   - ``--model_path``: Path to a local Hugging Face model directory or model ID from the Hugging Face Hub.
   - ``--task``: Task type for the Hugging Face model (currently only 'text-generation' is supported).
   - ``--device_map``: Device mapping strategy for model placement (e.g., 'auto', 'sequential', etc.).
   - ``--trust_remote_code``: Allow loading remote code from the Hugging Face Hub.
   - ``--model_id``: Identifier for the model in the API responses.
   - ``--host``: Host address to bind the Ray Serve server to. Default is 0.0.0.0.
   - ``--port``: Port number to use for the Ray Serve server. Default is 1024.
   - ``--num_cpus``: Number of CPUs to allocate for the Ray cluster. If None, will use all available CPUs.
   - ``--num_gpus``: Number of GPUs to allocate for the Ray cluster. Default is 1.
   - ``--include_dashboard``: Whether to include the Ray dashboard for monitoring.
   - ``--num_replicas``: Number of model replicas to deploy. Default is 1.
   - ``--num_gpus_per_replica``: Number of GPUs per model replica. Default is 1.
   - ``--num_cpus_per_replica``: Number of CPUs per model replica. Default is 8.
   - ``--cuda_visible_devices``: Comma-separated list of CUDA visible devices. Default is "0,1".
   - ``--max_memory``: Maximum memory allocation when using balanced device map.

3. To use a different model, modify the ``--model_path`` parameter. You can specify either a local path or a Hugging Face model ID.

4. For models requiring authentication (e.g., StarCoder1, StarCoder2, LLama3):

   Option 1 - Log in via CLI:
   
   ```shell
   huggingface-cli login
   ```

   Option 2 - Set environment variable:

   ```shell
   export HF_TOKEN=your_token_here
   ```

### Multi-Replica Deployment

Ray Serve excels at single-node multi-instance deployment. This allows you to deploy multiple instances of the same model to handle increased load:

1. Deploy multiple replicas using the ``--num_replicas`` parameter:

   ```shell
   python scripts/deploy/nlp/deploy_ray_hf.py \
      --model_path meta-llama/Llama-3.2-1B \
      --model_id llama \
      --num_replicas 4 \
      --num_gpus 4 \
      --num_gpus_per_replica 1 \
      --cuda_visible_devices "0,1,2,3"
   ```

2. For models that require multiple GPUs per replica:

   ```shell
   python scripts/deploy/nlp/deploy_ray_hf.py \
      --model_path meta-llama/Llama-3.2-1B \
      --model_id llama \
      --num_replicas 2 \
      --num_gpus 4 \
      --num_gpus_per_replica 2 \
      --cuda_visible_devices "0,1,2,3"
   ```

3. Ray automatically handles load balancing across replicas, distributing incoming requests to available instances.

**Important GPU Configuration Notes:**
- ``--num_gpus`` should equal ``--num_replicas`` × ``--num_gpus_per_replica``
- ``--cuda_visible_devices`` should list all GPUs that will be used
- Ensure the number of devices in ``--cuda_visible_devices`` matches ``--num_gpus``

### Testing Ray Deployment

Use the ``query_ray_deployment.py`` script to test your deployed model:

1. Basic testing:

   ```shell
   python scripts/deploy/nlp/query_ray_deployment.py \
      --model_id llama \
      --host 0.0.0.0 \
      --port 1024
   ```

2. The script will test multiple endpoints:
   - Health check endpoint: ``/v1/health``
   - Models list endpoint: ``/v1/models``
   - Text completions endpoint: ``/v1/completions/``

3. Available parameters for testing:
   - ``--host``: Host address of the Ray Serve server. Default is 0.0.0.0.
   - ``--port``: Port number of the Ray Serve server. Default is 1024.
   - ``--model_id``: Identifier for the model in the API responses. Default is "nemo-model".

### Advanced Configuration

For more advanced deployment scenarios:

1. **Custom Resource Allocation**:

   ```shell
   python scripts/deploy/nlp/deploy_ray_hf.py \
      --model_path meta-llama/Llama-3.2-1B \
      --model_id llama \
      --num_replicas 3 \
      --num_gpus 3 \
      --num_gpus_per_replica 1 \
      --num_cpus 48 \
      --num_cpus_per_replica 16 \
      --cuda_visible_devices "0,1,2"
   ```

2. **Memory Management**:

   ```shell
   python scripts/deploy/nlp/deploy_ray_hf.py \
      --model_path meta-llama/Llama-3.2-1B \
      --model_id llama \
      --num_replicas 2 \
      --num_gpus 2 \
      --num_gpus_per_replica 1 \
      --device_map balanced \
      --max_memory 75GiB \
      --cuda_visible_devices "0,1"
   ```

## API Endpoints

Once deployed, your model will be available through OpenAI-compatible API endpoints:

- **Health Check**: ``GET /v1/health``
- **List Models**: ``GET /v1/models``
- **Text Completions**: ``POST /v1/completions/``
- **Chat Completions**: ``POST /v1/chat/completions/``

Example API request:

```bash
curl -X POST http://localhost:1024/v1/completions/ \
  -H "Content-Type: application/json" \
  -d '{
    "model": "llama",
    "prompt": "The capital of France is",
    "max_tokens": 50,
    "temperature": 0.7
  }'
```

## Troubleshooting

1. **Out of Memory Errors**: Reduce ``--num_replicas`` or ``--num_gpus_per_replica``
2. **Port Already in Use**: Change the ``--port`` parameter
3. **Ray Cluster Issues**: Ensure no other Ray processes are running: ``ray stop``
4. **GPU Allocation**: Verify ``--cuda_visible_devices`` matches your available GPUs
5. **GPU Configuration Errors**: Ensure ``--num_gpus`` = ``--num_replicas`` × ``--num_gpus_per_replica``
6. **CUDA Device Mismatch**: Make sure the number of devices in ``--cuda_visible_devices`` equals ``--num_gpus``

For more information on Ray Serve, visit the [Ray Serve documentation](https://docs.ray.io/en/latest/serve/index.html). 