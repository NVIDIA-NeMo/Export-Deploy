# Deploy AutoModel LLMs using Ray Serve

This section demonstrates how to deploy [NeMo AutoModel](https://docs.nvidia.com/nemo/automodel/latest/index.html) LLM models using Ray Serve (referred to as 'Ray for AutoModel LLM'). To support single-node, multi-instance deployment, Ray is now offered as an alternative to Triton. Ray Serve provides a scalable and flexible platform for deploying machine learning models, offering features such as automatic scaling, load balancing, and multi-replica deployment.

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
       --name nemo-fw \
       nvcr.io/nvidia/nemo:vr
   ``` 

3. Log in to Hugging Face with your access token:

   ```shell
   huggingface-cli login
   ```

4. Deploy the model to Ray:

   ```python
   python /opt/Export-Deploy/scripts/deploy/nlp/deploy_ray_hf.py \
      --model_path meta-llama/Llama-3.2-1B \
      --model_id llama \
      --num_replicas 2 \
      --num_gpus 2 \
      --num_gpus_per_replica 1 \
      --cuda_visible_devices "0,1"
   ```
   
   **Note:** If you encounter shared memory errors, increase ``--shm-size`` gradually by 50%.

5. In a separate terminal, access the running container as follows:

   ```shell
   docker exec -it nemo-fw bash
   ```

6. Test the deployed model:

   ```shell
   python /opt/Export-Deploy/scripts/deploy/nlp/query_ray_deployment.py \
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
   python /opt/Export-Deploy/scripts/deploy/nlp/deploy_ray_hf.py \
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

### Deploy Multiple Replicas

Ray Serve excels at single-node multi-instance deployment. This allows you to deploy multiple instances of the same model to handle increased load:

1. Deploy multiple replicas using the ``--num_replicas`` parameter:

   ```shell
   python /opt/Export-Deploy/scripts/deploy/nlp/deploy_ray_hf.py \
      --model_path meta-llama/Llama-3.2-1B \
      --model_id llama \
      --num_replicas 4 \
      --num_gpus 4 \
      --num_gpus_per_replica 1 \
      --cuda_visible_devices "0,1,2,3"
   ```

2. For models that require multiple GPUs per replica:

   ```shell
   python /opt/Export-Deploy/scripts/deploy/nlp/deploy_ray_hf.py \
      --model_path meta-llama/Llama-3.2-1B \
      --model_id llama \
      --num_replicas 2 \
      --num_gpus 4 \
      --num_gpus_per_replica 2 \
      --cuda_visible_devices "0,1,2,3"
   ```

3. Ray automatically handles load balancing across replicas, distributing incoming requests to available instances.

**Important GPU Configuration Notes:**
- ``--num_gpus`` should equal ``--num_replicas`` × ``--num_gpus_per_replica``.
- ``--cuda_visible_devices`` should list all GPUs that will be used
- Ensure the number of devices in ``--cuda_visible_devices`` matches ``--num_gpus``.

### Test Ray Deployment

Use the ``query_ray_deployment.py`` script to test your deployed model:

1. Basic testing:

   ```shell
   python /opt/Export-Deploy/scripts/deploy/nlp/query_ray_deployment.py \
      --model_id llama \
      --host 0.0.0.0 \
      --port 1024
   ```

2. The script will test multiple endpoints:
   - Health check endpoint: ``/v1/health``.
   - Models list endpoint: ``/v1/models``.
   - Text completions endpoint: ``/v1/completions/``.

3. Available parameters for testing:
   - ``--host``: Host address of the Ray Serve server. Default is 0.0.0.0.
   - ``--port``: Port number of the Ray Serve server. Default is 1024.
   - ``--model_id``: Identifier for the model in the API responses. Default is ``nemo-model``.

### Configure Advanced Deployments

For more advanced deployment scenarios:

1. **Custom Resource Allocation**:

   ```shell
   python /opt/Export-Deploy/scripts/deploy/nlp/deploy_ray_hf.py \
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
   python /opt/Export-Deploy/scripts/deploy/nlp/deploy_ray_hf.py \
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

- **Health Check**: ``GET /v1/health``.
- **List Models**: ``GET /v1/models``.
- **Text Completions**: ``POST /v1/completions/``.
- **Chat Completions**: ``POST /v1/chat/completions/``.

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

1. **Out of Memory Errors**: Reduce ``--num_replicas`` or ``--num_gpus_per_replica``.
2. **Port Already in Use**: Change the ``--port`` parameter.
3. **Ray Cluster Issues**: Ensure no other Ray processes are running: ``ray stop``.
4. **GPU Allocation**: Verify ``--cuda_visible_devices`` matches your available GPUs.
5. **GPU Configuration Errors**: Ensure ``--num_gpus`` = ``--num_replicas`` × ``--num_gpus_per_replica``.
6. **CUDA Device Mismatch**: Make sure the number of devices in ``--cuda_visible_devices`` equals ``--num_gpus``.

For more information on Ray Serve, visit the [Ray Serve documentation](https://docs.ray.io/en/latest/serve/index.html). 

### Multi-node on SLURM using ray.sub

Use `scripts/deploy/utils/ray.sub` to bring up a Ray cluster across multiple SLURM nodes and run your AutoModel deployment automatically. This script starts a Ray head and workers, manages ports, and launches a driver command when the cluster is ready.

- **Script location**: `scripts/deploy/utils/ray.sub`
- **Upstream reference**: See the NeMo RL cluster setup doc for background on this pattern: [NVIDIA-NeMo RL cluster guide](https://github.com/NVIDIA-NeMo/RL/blob/main/docs/cluster.md)

#### Prerequisites

- SLURM with container support for `srun --container-image` and `--container-mounts`.
- A container image that includes Export-Deploy at `/opt/Export-Deploy`.
- Any model access/auth if required (e.g., `huggingface-cli login` or `HF_TOKEN`).

#### Quick start (2 nodes, 16 GPUs total)

1) Set environment variables used by `ray.sub`:

```bash
export CONTAINER=nvcr.io/nvidia/nemo:vr
export MOUNTS="${PWD}/:/opt/checkpoints/"
export GPUS_PER_NODE=8

# Driver command to run after the cluster is ready (multi-node AutoModel deployment)
export COMMAND="python /opt/Export-Deploy/scripts/deploy/nlp/deploy_ray_hf.py --model_path meta-llama/Llama-3.2-1B --model_id llama --num_replicas 16 --num_gpus 16 --num_gpus_per_replica 1"
```

2) Submit the job:

```bash
sbatch --nodes=2 --account <ACCOUNT> --partition <PARTITION> \
  --job-name automodel-ray --time 01:00:00 \
  /opt/Export-Deploy/scripts/deploy/utils/ray.sub
```

The script will:
- Start a Ray head on node 0 and one Ray worker per remaining node
- Wait until all nodes register their resources
- Launch the `COMMAND` on the head node (driver) once the cluster is healthy

3) Attaching and monitoring:
- Logs: `$SLURM_SUBMIT_DIR/<jobid>-logs/` contains `ray-head.log` and `ray-worker-<n>.log`.
- Interactive shell: the job creates `<jobid>-attach.sh`. For head: `bash <jobid>-attach.sh`. For worker i: `bash <jobid>-attach.sh i`.
- Ray status: once attached to the head container, run `ray status`.

4) Query the deployment (from within the head container):

```bash
python /opt/Export-Deploy/scripts/deploy/nlp/query_ray_deployment.py \
  --model_id llama --host 0.0.0.0 --port 1024
```

#### Notes

- Set `--num_gpus` in the deploy command to the total GPUs across all nodes; ensure `--num_gpus = --num_replicas × --num_gpus_per_replica`.
- If your cluster uses GRES, `ray.sub` auto-detects and sets `--gres=gpu:<GPUS_PER_NODE>`; ensure `GPUS_PER_NODE` matches the node GPU count.
- You usually do not need to set `--cuda_visible_devices` for multi-node; Ray workers handle per-node visibility.