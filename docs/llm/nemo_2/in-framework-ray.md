# Deploy NeMo 2.0 LLMs with Ray Serve

This section demonstrates how to deploy NeMo LLM models using Ray Serve. Ray Serve deployment support provides scalable and flexible deployment for NeMo models, offering features such as automatic scaling, load balancing, and multi-replica deployment with support for advanced parallelism strategies.

**Note:** Single-node examples are shown below. For multi-node clusters managed by SLURM, you can deploy across nodes using the `ray.sub` helper described in the section "Multi-node on SLURM using ray.sub".

## Quick Example

1. Follow the steps on the [Generate A NeMo 2.0 Checkpoint page](gen_nemo2_ckpt.md) to generate a NeMo 2.0 Llama checkpoint.

2. In a terminal, go to the folder where the ``hf_llama31_8B_nemo2.nemo`` file is located. Pull and run the Docker container image. Replace ``:vr`` with your desired version:

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

3. Deploy the NeMo model with Ray Serve:

   ```shell
   python /opt/Export-Deploy/scripts/deploy/nlp/deploy_ray_inframework.py \
      --nemo_checkpoint /opt/checkpoints/hf_llama31_8B_nemo2.nemo \
      --model_id llama \
      --num_replicas 1 \
      --num_gpus 1 \
      --tensor_model_parallel_size 1 \
      --pipeline_model_parallel_size 1 \
      --cuda_visible_devices "0"
   ```

4. In a separate terminal, access the running container as follows:

   ```shell
   docker exec -it nemo-fw bash
   ```

5. Test the deployed model:

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
   python /opt/Export-Deploy/scripts/deploy/nlp/deploy_ray_inframework.py \
      --nemo_checkpoint /opt/checkpoints/hf_llama31_8B_nemo2.nemo \
      --model_id llama \
      --num_replicas 1 \
      --num_gpus 2 \
      --tensor_model_parallel_size 2 \
      --pipeline_model_parallel_size 1 \
      --cuda_visible_devices "0,1"
   ```

   Available Parameters:
   
   - ``--nemo_checkpoint``: Path to the NeMo checkpoint file (required).
   - ``--num_gpus``: Number of GPUs to use per node. Default is 1.
   - ``--tensor_model_parallel_size``: Size of the tensor model parallelism. Default is 1.
   - ``--pipeline_model_parallel_size``: Size of the pipeline model parallelism. Default is 1.
   - ``--expert_model_parallel_size``: Size of the expert model parallelism. Default is 1.
   - ``--context_parallel_size``: Size of the context parallelism. Default is 1.
   - ``--model_id``: Identifier for the model in the API responses. Default is ``nemo-model``.
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

3. To use a different model, modify the ``--nemo_checkpoint`` parameter with the path to your NeMo checkpoint file.


### Configure Model Parallelism

NeMo models support advanced parallelism strategies for large model deployment:

1. **Tensor Model Parallelism**: Distributes model layers across multiple GPUs:

   ```shell
   python /opt/Export-Deploy/scripts/deploy/nlp/deploy_ray_inframework.py \
      --nemo_checkpoint /opt/checkpoints/hf_llama31_8B_nemo2.nemo \
      --model_id large_llama \
      --num_gpus 4 \
      --tensor_model_parallel_size 4 \
      --pipeline_model_parallel_size 1 \
      --cuda_visible_devices "0,1,2,3"
   ```

2. **Pipeline Model Parallelism**: Distributes model layers sequentially across GPUs:

   ```shell
   python /opt/Export-Deploy/scripts/deploy/nlp/deploy_ray_inframework.py \
      --nemo_checkpoint /opt/checkpoints/hf_llama31_8B_nemo2.nemo \
      --model_id large_llama \
      --num_gpus 4 \
      --tensor_model_parallel_size 1 \
      --pipeline_model_parallel_size 4 \
      --cuda_visible_devices "0,1,2,3"
   ```

3. **Combined Parallelism**: Uses both tensor and pipeline parallelism:

   ```shell
   python /opt/NeMo-Export-Deploy/scripts/deploy/nlp/deploy_ray_inframework.py \
      --nemo_checkpoint /opt/checkpoints/hf_llama31_8B_nemo2.nemo \
      --model_id large_llama \
      --num_gpus 8 \
      --tensor_model_parallel_size 2 \
      --pipeline_model_parallel_size 4 \
      --cuda_visible_devices "0,1,2,3,4,5,6,7"
   ```

### Deploy Multiple Replicas

Deploy multiple replicas of your NeMo model for increased throughput:

1. **Single GPU per replica**:

   ```shell
   python /opt/Export-Deploy/scripts/deploy/nlp/deploy_ray_inframework.py \
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
   python /opt/Export-Deploy/scripts/deploy/nlp/deploy_ray_inframework.py \
      --nemo_checkpoint /opt/checkpoints/hf_llama31_8B_nemo2.nemo \
      --model_id large_llama \
      --num_replicas 2 \
      --num_gpus 8 \
      --tensor_model_parallel_size 4 \
      --pipeline_model_parallel_size 1 \
      --cuda_visible_devices "0,1,2,3,4,5,6,7"
   ```

**Important GPU Configuration Notes:**
- GPUs per replica = Total GPUs ÷ ``--num_replicas``.
- Each replica needs: ``--tensor_model_parallel_size`` × ``--pipeline_model_parallel_size`` × ``--context_parallel_size`` GPUs.
- Ensure ``--cuda_visible_devices`` lists all GPUs that will be used.

### Optimize Performance

Enable performance optimizations for faster inference:

1. **Flash Attention Decode**: Optimizes attention computation:

   ```shell
   python /opt/Export-Deploy/scripts/deploy/nlp/deploy_ray_inframework.py \
      --nemo_checkpoint /opt/checkpoints/hf_llama31_8B_nemo2.nemo \
      --model_id llama \
      --enable_flash_decode \
      --num_gpus 2 \
      --tensor_model_parallel_size 2 \
      --cuda_visible_devices "0,1"
   ```

2. **Flash Attention Decode and Cuda Graphs**:

   ```shell
   python /opt/Export-Deploy/scripts/deploy/nlp/deploy_ray_inframework.py \
      --nemo_checkpoint /opt/checkpoints/hf_llama31_8B_nemo2.nemo \
      --model_id llama \
      --enable_cuda_graphs \
      --enable_flash_decode \
      --num_gpus 4 \
      --tensor_model_parallel_size 2 \
      --pipeline_model_parallel_size 2 \
      --cuda_visible_devices "0,1,2,3"
   ```

### Test Ray Deployment

Use the ``query_ray_deployment.py`` script to test your deployed NeMo model:

1. Basic testing:

   ```shell
   python /opt/Export-Deploy/scripts/deploy/nlp/query_ray_deployment.py \
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
   - ``--model_id``: Identifier for the model in the API responses. Default is ``nemo-model``.

### Configure Advanced Deployments

For more advanced deployment scenarios:

1. **Custom Resource Allocation**:

   ```shell
   python /opt/Export-Deploy/scripts/deploy/nlp/deploy_ray_inframework.py \
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
   python /opt/Export-Deploy/scripts/deploy/nlp/deploy_ray_inframework.py \
      --nemo_checkpoint /opt/checkpoints/hf_llama31_8B_nemo2.nemo \
      --model_id llama \
      --legacy_ckpt \
      --num_gpus 2 \
      --tensor_model_parallel_size 2 \
      --cuda_visible_devices "0,1"
   ```

### Deploy MegatronLM and MBridge Models

You can deploy checkpoints saved in MegatronLM or MBridge formats by using the `--megatron_checkpoint` flag instead of `--nemo_checkpoint`.

- MBridge example:

```shell
python /opt/Export-Deploy/scripts/deploy/nlp/deploy_ray_inframework.py \
   --megatron_checkpoint /opt/checkpoints/llama3_145m-mbridge_saved-distckpt/ \
   --model_id llama \
   --tensor_model_parallel_size 2 \
   --pipeline_model_parallel_size 2 \
   --num_gpus 4
```

- MegatronLM example:

```shell
python /opt/Export-Deploy/scripts/deploy/nlp/deploy_ray_inframework.py \
   --megatron_checkpoint /opt/checkpoints/checkpoints/llama3_145m-mlm_saved-distckpt/ \
   --model_id llama \
   --tensor_model_parallel_size 2 \
   --pipeline_model_parallel_size 2 \
   --num_gpus 4 \
   --model_type gpt
```

Notes:
- Use `--model_type gpt` for MegatronLM GPT-style checkpoints.
- Parallelism settings must be compatible with available GPUs (see Configure Model Parallelism).

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

1. **Out of Memory Errors**: Reduce ``--num_replicas`` or adjust parallelism settings.
2. **Port Already in Use**: Change the ``--port`` parameter.
3. **Ray Cluster Issues**: Ensure no other Ray processes are running: ``ray stop``.
4. **GPU Allocation**: Verify ``--cuda_visible_devices`` matches your available GPUs.
5. **Parallelism Configuration Errors**: Ensure total parallelism per replica matches available GPUs per replica.
6. **CUDA Device Mismatch**: Make sure the number of devices in ``--cuda_visible_devices`` equals total GPUs.
7. **Checkpoint Loading Issues**: Verify the ``.nemo`` checkpoint path is correct and accessible.
8. **Legacy Checkpoint**: Use ``--legacy_ckpt`` flag for older checkpoint formats.

**Note:** Only NeMo 2.0 checkpoints are supported by default. For older checkpoints, use the ``--legacy_ckpt`` flag.

For more information on Ray Serve, visit the [Ray Serve documentation](https://docs.ray.io/en/latest/serve/index.html). 

### Multi-node on SLURM using ray.sub

Use `scripts/deploy/utils/ray.sub` to bring up a Ray cluster across multiple SLURM nodes and run your in-framework NeMo deployment automatically. This script configures the Ray head and workers, handles ports, and can optionally run a driver command once the cluster is online.

- **Script location**: `scripts/deploy/utils/ray.sub`
- **Upstream reference**: See the NeMo RL cluster setup doc for background on this pattern: [NVIDIA-NeMo RL cluster guide](https://github.com/NVIDIA-NeMo/RL/blob/main/docs/cluster.md)

#### Prerequisites

- A SLURM cluster with container support for `srun --container-image` and `--container-mounts`.
- A container image that includes Export-Deploy at `/opt/Export-Deploy` and the needed dependencies.
- A `.nemo` checkpoint accessible on the cluster filesystem.

#### Quick start (2 nodes, 16 GPUs total)

1) Set environment variables to parameterize `ray.sub` (these are read by the script at submission time):

```bash
export CONTAINER=nvcr.io/nvidia/nemo:vr
export MOUNTS="${PWD}/:/opt/checkpoints/"

# Optional tuning
export GPUS_PER_NODE=8                   # default 8; set to your node GPU count

# Driver command to run after the cluster is ready (multi-node NeMo deployment)
export COMMAND="python /opt/Export-Deploy/scripts/deploy/nlp/deploy_ray_inframework.py --nemo_checkpoint /opt/checkpoints/model.nemo --model_id llama --num_replicas 16 --num_gpus 16"
```

2) Submit the job (you can override SBATCH directives on the command line):

```bash
sbatch --nodes=2 --account <ACCOUNT> --partition <PARTITION> \
  --job-name nemo-ray --time 01:00:00 \
  /opt/Export-Deploy/scripts/deploy/utils/ray.sub
```

The script will:
- Start a Ray head on node 0 and one Ray worker per remaining node
- Wait until all nodes register their resources
- Launch the `COMMAND` on the head node (driver) once the cluster is healthy

3) Attaching and monitoring:
- Logs: `$SLURM_SUBMIT_DIR/<jobid>-logs/` contains `ray-head.log`, `ray-worker-<n>.log`, and (if set) synced Ray logs.
- Interactive shell: the job creates `<jobid>-attach.sh`. For head: `bash <jobid>-attach.sh`. For worker i: `bash <jobid>-attach.sh i`.
- Ray status: once attached to the head container, run `ray status`.

4) Query the deployment (from within the head container):

```bash
python /opt/Export-Deploy/scripts/deploy/nlp/query_ray_deployment.py \
  --model_id llama --host 0.0.0.0 --port 1024
```

#### Notes

- Set `--num_gpus` in the deploy command to the total GPUs across all nodes; adjust `--num_replicas` and model parallel sizes per your topology.
- If your cluster uses GRES, `ray.sub` auto-detects and sets `--gres=gpu:<GPUS_PER_NODE>`; ensure `GPUS_PER_NODE` matches the node’s GPU count.
- You can leave `--cuda_visible_devices` unset for multi-node runs; per-node visibility is managed by Ray workers.