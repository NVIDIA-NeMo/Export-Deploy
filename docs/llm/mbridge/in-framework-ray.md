# Deploy Megatron-Bridge LLMs with Ray Serve

This section demonstrates how to deploy Megatron-Bridge LLM models using Ray Serve. Ray deployment support provides scalable and flexible deployment for NeMo models, offering features such as automatic scaling, load balancing, and multi-replica deployment with support for advanced parallelism strategies.

**Note:** Single-node examples are shown below. For multi-node clusters managed by SLURM, you can deploy across nodes using the `ray.sub` helper described in the section "Multi-node on SLURM using ray.sub".

## Quick Example

1. Follow the steps on the [Generate A Megatron-Bridge Checkpoint page](gen_mbridge_ckpt.md) to generate a Megatron-Bridge Llama checkpoint.

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

3. Deploy the NeMo model to Ray:

   ```shell
   python /opt/Export-Deploy/scripts/deploy/nlp/deploy_ray_inframework.py \
      --megatron_checkpoint /opt/checkpoints/hf_llama31_8B_mbridge \
      --model_format megatron \
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

Deploying Megatron-Bridge models with Ray Serve closely follows the same process as deploying NeMo 2.0 models. The primary differences are:

- Use the `--megatron_checkpoint` argument to specify your Megatron-Bridge checkpoint file.
- Set `--model_format megatron` to indicate the model type.

All other deployment steps, parameters, and Ray Serve features remain the same as for NeMo 2.0 models. For a comprehensive walkthrough of advanced options, scaling, and troubleshooting, refer to the [Deploy NeMo 2.0 LLMs with Ray Serve](../nemo_2/in-framework-ray.md) documentation.