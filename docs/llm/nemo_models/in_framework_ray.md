# Deploy NeMo Models using Ray

This section demonstrates how to deploy NeMo LLM models using Ray Serve (referred to as 'Ray for NeMo Models'). Ray deployment support provides scalable and flexible deployment for NeMo models, offering features such as automatic scaling, load balancing, and multi-replica deployment with support for advanced parallelism strategies.

**Note:** Currently, only single-node deployment is supported.

## Quick Example

1. Follow the steps in the [Deploy NeMo LLM main page](../index.md) to generate a NeMo 2.0 checkpoint.

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

3. Deploy the NeMo model to Ray:

   ```shell
   python scripts/deploy/nlp/deploy_ray_inframework.py \
      --nemo_checkpoint /opt/checkpoints/hf_llama31_8B_nemo2.nemo \
      --model_id llama \
      --num_replicas 1 \
      --num_gpus 1 \
      --tensor_model_parallel_size 1 \
      --pipeline_model_parallel_size 1 \
      --cuda_visible_devices "0"
   ```

4. In a new terminal, get the container ID:

   ```shell
   docker ps
   ```

5. Access the container:

   ```shell
   docker exec -it <container_id> bash
   ```

6. Test the deployed model:

   ```shell
   python scripts/deploy/nlp/query_ray_deployment.py \
      --model_id llama \
      --host 0.0.0.0 \
      --port 1024
   ```

## Detailed Deployment Guide

### Deploy a NeMo LLM Model

Follow these steps to deploy your NeMo model on Ray Serve:

1. Start the container as shown in the **Quick Example** section.

2. Deploy your model:

   ```shell
   python scripts/deploy/nlp/deploy_ray_inframework.py \
      --nemo_checkpoint /opt/checkpoints/hf_llama31_8B_nemo2.nemo \
      --model_id llama \
      --num_replicas 1 \
      --num_gpus 2 \
      --tensor_model_parallel_size 2 \
      --pipeline_model_parallel_size 1 \
      --cuda_visible_devices "0,1"
   ```

   Available Parameters:
   
   - ``--nemo_checkpoint``: Path to the .nemo checkpoint file (required).
   - ``--num_gpus``: Number of GPUs to use per node. Default is 1.
   - ``--tensor_model_parallel_size``: Size of the tensor model parallelism. Default is 1.
   - ``--pipeline_model_parallel_size``: Size of the pipeline model parallelism. Default is 1.
   - ``--expert_model_parallel_size``: Size of the expert model parallelism. Default is 1.
   - ``--context_parallel_size``: Size of the context parallelism. Default is 1.
   - ``--model_id``: Identifier for the model in the API responses. Default is "nemo-model".
   - ``--host``: Host address to bind the Ray Serve server to. Default is 0.0.0.0.
   - ``--port``: Port number to use for the Ray Serve server. Default is 1024.
   - ``--num_cpus``: Number of CPUs to allocate for the Ray cluster. If None, will use all available CPUs.
   - ``--num_cpus_per_replica``: Number of CPUs per model replica. Default is 8.
   - ``--include_dashboard``: Whether to include the Ray dashboard for monitoring.
   - ``--cuda_visible_devices``: Comma-separated list of CUDA visible devices. Default is "0,1".
   - ``--enable_cuda_graphs``: Whether to enable CUDA graphs for faster inference.
   - ``--enable_flash_decode``: Whether to enable Flash Attention decode.
   - ``--num_replicas``: Number of replicas for the deployment. Default is 1.
   - ``--legacy_ckpt``: Whether to use legacy checkpoint format.

3. To use a different model, modify the ``--nemo_checkpoint`` parameter with the path to your .nemo checkpoint file.


### Model Parallelism Configuration

NeMo models support advanced parallelism strategies for large model deployment:

1. **Tensor Model Parallelism**: Distributes model layers across multiple GPUs:

   ```shell
   python scripts/deploy/nlp/deploy_ray_inframework.py \
      --nemo_checkpoint /opt/checkpoints/hf_llama31_8B_nemo2.nemo \
      --model_id large_llama \
      --num_gpus 4 \
      --tensor_model_parallel_size 4 \
      --pipeline_model_parallel_size 1 \
      --cuda_visible_devices "0,1,2,3"
   ```

2. **Pipeline Model Parallelism**: Distributes model layers sequentially across GPUs:

   ```shell
   python scripts/deploy/nlp/deploy_ray_inframework.py \
      --nemo_checkpoint /opt/checkpoints/hf_llama31_8B_nemo2.nemo \
      --model_id large_llama \
      --num_gpus 4 \
      --tensor_model_parallel_size 1 \
      --pipeline_model_parallel_size 4 \
      --cuda_visible_devices "0,1,2,3"
   ```

3. **Combined Parallelism**: Uses both tensor and pipeline parallelism:

   ```shell
   python scripts/deploy/nlp/deploy_ray_inframework.py \
      --nemo_checkpoint /opt/checkpoints/hf_llama31_8B_nemo2.nemo \
      --model_id large_llama \
      --num_gpus 8 \
      --tensor_model_parallel_size 2 \
      --pipeline_model_parallel_size 4 \
      --cuda_visible_devices "0,1,2,3,4,5,6,7"
   ```

### Multi-Replica Deployment

Deploy multiple replicas of your NeMo model for increased throughput:

1. **Single GPU per replica**:

   ```shell
   python scripts/deploy/nlp/deploy_ray_inframework.py \
      --nemo_checkpoint /opt/checkpoints/hf_llama31_8B_nemo2.nemo \
      --model_id llama \
      --num_replicas 4 \
      --num_gpus 4 \
      --tensor_model_parallel_size 1 \
      --pipeline_model_parallel_size 1 \
      --cuda_visible_devices "0,1,2,3"
   ```

2. **Multiple GPUs per replica with tensor parallelism**:

   ```shell
   python scripts/deploy/nlp/deploy_ray_inframework.py \
      --nemo_checkpoint /opt/checkpoints/hf_llama31_8B_nemo2.nemo \
      --model_id large_llama \
      --num_replicas 2 \
      --num_gpus 8 \
      --tensor_model_parallel_size 4 \
      --pipeline_model_parallel_size 1 \
      --cuda_visible_devices "0,1,2,3,4,5,6,7"
   ```

**Important GPU Configuration Notes:**
- GPUs per replica = Total GPUs ÷ ``--num_replicas``
- Each replica needs: ``--tensor_model_parallel_size`` × ``--pipeline_model_parallel_size`` × ``--context_parallel_size`` GPUs
- Ensure ``--cuda_visible_devices`` lists all GPUs that will be used

### Performance Optimization

Enable performance optimizations for faster inference:

1. **CUDA Graphs**: Reduces kernel launch overhead:

   ```shell
   python scripts/deploy/nlp/deploy_ray_inframework.py \
      --nemo_checkpoint /opt/checkpoints/hf_llama31_8B_nemo2.nemo \
      --model_id llama \
      --enable_cuda_graphs \
      --num_gpus 2 \
      --tensor_model_parallel_size 2 \
      --cuda_visible_devices "0,1"
   ```

2. **Flash Attention Decode**: Optimizes attention computation:

   ```shell
   python scripts/deploy/nlp/deploy_ray_inframework.py \
      --nemo_checkpoint /opt/checkpoints/hf_llama31_8B_nemo2.nemo \
      --model_id llama \
      --enable_flash_decode \
      --num_gpus 2 \
      --tensor_model_parallel_size 2 \
      --cuda_visible_devices "0,1"
   ```

3. **Combined Optimizations**:

   ```shell
   python scripts/deploy/nlp/deploy_ray_inframework.py \
      --nemo_checkpoint /opt/checkpoints/hf_llama31_8B_nemo2.nemo \
      --model_id llama \
      --enable_cuda_graphs \
      --enable_flash_decode \
      --num_gpus 4 \
      --tensor_model_parallel_size 2 \
      --pipeline_model_parallel_size 2 \
      --cuda_visible_devices "0,1,2,3"
   ```

### Testing Ray Deployment

Use the ``query_ray_deployment.py`` script to test your deployed NeMo model:

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
   python scripts/deploy/nlp/deploy_ray_inframework.py \
      --nemo_checkpoint /opt/checkpoints/hf_llama31_8B_nemo2.nemo \
      --model_id llama \
      --num_replicas 2 \
      --num_gpus 4 \
      --tensor_model_parallel_size 2 \
      --num_cpus 32 \
      --num_cpus_per_replica 16 \
      --cuda_visible_devices "0,1,2,3"
   ```

2. **Legacy Checkpoint Support**:

   ```shell
   python scripts/deploy/nlp/deploy_ray_inframework.py \
      --nemo_checkpoint /opt/checkpoints/hf_llama31_8B_nemo2.nemo \
      --model_id llama \
      --legacy_ckpt \
      --num_gpus 2 \
      --tensor_model_parallel_size 2 \
      --cuda_visible_devices "0,1"
   ```

## API Endpoints

Once deployed, your NeMo model will be available through OpenAI-compatible API endpoints:

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

1. **Out of Memory Errors**: Reduce ``--num_replicas`` or adjust parallelism settings
2. **Port Already in Use**: Change the ``--port`` parameter
3. **Ray Cluster Issues**: Ensure no other Ray processes are running: ``ray stop``
4. **GPU Allocation**: Verify ``--cuda_visible_devices`` matches your available GPUs
5. **Parallelism Configuration Errors**: Ensure total parallelism per replica matches available GPUs per replica
6. **CUDA Device Mismatch**: Make sure the number of devices in ``--cuda_visible_devices`` equals total GPUs
7. **Checkpoint Loading Issues**: Verify the ``.nemo`` checkpoint path is correct and accessible
8. **Legacy Checkpoint**: Use ``--legacy_ckpt`` flag for older checkpoint formats

**Note:** Only NeMo 2.0 checkpoints are supported by default. For older checkpoints, use the ``--legacy_ckpt`` flag.

For more information on Ray Serve, visit the [Ray Serve documentation](https://docs.ray.io/en/latest/serve/index.html). 