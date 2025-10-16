# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging
import multiprocessing

from nemo_deploy.deploy_ray import DeployRay

LOGGER = logging.getLogger("NeMo")


def get_available_cpus():
    """Get the total number of available CPUs in the system."""
    return multiprocessing.cpu_count()


def parse_args():
    """Parse command-line arguments for the Ray deployment script."""
    parser = argparse.ArgumentParser(description="Deploy a Megatron model using Ray")
    parser.add_argument(
        "--nemo-checkpoint",
        type=str,
        default=None,
        help="Path to the .nemo checkpoint file",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="Number of GPUs to use per node",
    )
    parser.add_argument(
        "--tensor-model-parallel-size",
        type=int,
        default=1,
        help="Size of the tensor model parallelism",
    )
    parser.add_argument(
        "--pipeline-model-parallel-size",
        type=int,
        default=1,
        help="Size of the pipeline model parallelism",
    )
    parser.add_argument(
        "-nlfps",
        "--num-layers-in-first-pipeline-stage",
        default=None,
        type=int,
        help="Number of layers in the first pipeline stage",
    )
    parser.add_argument(
        "-nllps",
        "--num-layers-in-last-pipeline-stage",
        default=None,
        type=int,
        help="Number of layers in the last pipeline stage",
    )
    parser.add_argument(
        "--expert-model-parallel-size",
        type=int,
        default=1,
        help="Size of the expert model parallelism",
    )
    parser.add_argument(
        "--context-parallel-size",
        type=int,
        default=1,
        help="Size of the context parallelism",
    )
    parser.add_argument(
        "-eps",
        "--account-for-embedding-in-pipeline-split",
        default=False,
        action="store_true",
        help="Account for embedding in the pipeline split",
    )
    parser.add_argument(
        "-lps",
        "--account-for-loss-in-pipeline-split",
        default=False,
        action="store_true",
        help="Account for loss in the pipeline split",
    )
    parser.add_argument(
        "--model-id",
        type=str,
        default="nemo-model",
        help="Identifier for the model in the API responses",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host address to bind the Ray Serve server to",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=1024,
        help="Port number to use for the Ray Serve server",
    )
    parser.add_argument(
        "--num-cpus",
        type=int,
        default=None,
        help="Number of CPUs to allocate for the Ray cluster. If None, will use all available CPUs.",
    )
    parser.add_argument(
        "--num-cpus-per-replica",
        type=float,
        default=8,
        help="Number of CPUs per model replica",
    )
    parser.add_argument(
        "--include-dashboard",
        action="store_true",
        help="Whether to include the Ray dashboard",
    )
    parser.add_argument(
        "--cuda-visible-devices",
        type=str,
        default=None,
        help="Comma-separated list of CUDA visible devices",
    )
    parser.add_argument(
        "--enable-cuda-graphs",
        action="store_true",
        help="Whether to enable CUDA graphs for faster inference",
    )
    parser.add_argument(
        "--enable-flash-decode",
        action="store_true",
        help="Whether to enable Flash Attention decode",
    )
    parser.add_argument(
        "--num-replicas",
        type=int,
        default=1,
        help="Number of replicas for the deployment",
    )
    parser.add_argument(
        "--legacy-ckpt",
        action="store_true",
        help="Whether to use legacy checkpoint format",
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=32,
        help="Maximum batch size for inference",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="Random seed for reproducible inference",
    )
    parser.add_argument(
        "--micro-batch-size",
        type=int,
        default=None,
        help="Micro batch size for model execution",
    )
    return parser.parse_args()


def main():
    """Main function to deploy a Megatron model using Ray."""
    args = parse_args()
    # Initialize Ray deployment with updated DeployRay class
    runtime_env = {}
    if args.cuda_visible_devices is not None:
        runtime_env["env_vars"] = {
            "CUDA_VISIBLE_DEVICES": args.cuda_visible_devices,
        }

    ray_deployer = DeployRay(
        num_cpus=args.num_cpus,
        num_gpus=args.num_gpus,
        include_dashboard=args.include_dashboard,
        host=args.host,
        port=args.port,
        runtime_env=runtime_env,
    )
    if not args.nemo_checkpoint:
        raise ValueError("--nemo-checkpoint must be provided")

    model_config_kwargs = {
        "account_for_embedding_in_pipeline_split": args.account_for_embedding_in_pipeline_split,
        "account_for_loss_in_pipeline_split": args.account_for_loss_in_pipeline_split,
    }

    if args.num_layers_in_first_pipeline_stage is not None:
        model_config_kwargs["num_layers_in_first_pipeline_stage"] = args.num_layers_in_first_pipeline_stage

    if args.num_layers_in_last_pipeline_stage is not None:
        model_config_kwargs["num_layers_in_last_pipeline_stage"] = args.num_layers_in_last_pipeline_stage

    # Deploy the inframework model using the updated API
    ray_deployer.deploy_inframework_model(
        nemo_checkpoint=args.nemo_checkpoint,
        num_gpus=args.num_gpus,
        tensor_model_parallel_size=args.tensor_model_parallel_size,
        pipeline_model_parallel_size=args.pipeline_model_parallel_size,
        expert_model_parallel_size=args.expert_model_parallel_size,
        context_parallel_size=args.context_parallel_size,
        model_id=args.model_id,
        num_cpus_per_replica=args.num_cpus_per_replica,
        num_replicas=args.num_replicas,
        enable_cuda_graphs=args.enable_cuda_graphs,
        enable_flash_decode=args.enable_flash_decode,
        legacy_ckpt=args.legacy_ckpt,
        max_batch_size=args.max_batch_size,
        random_seed=args.random_seed,
        micro_batch_size=args.micro_batch_size,
        **model_config_kwargs,
    )


if __name__ == "__main__":
    main()
