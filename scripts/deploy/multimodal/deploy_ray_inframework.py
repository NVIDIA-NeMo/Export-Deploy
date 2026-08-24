# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
    """Parse command-line arguments for the Ray multimodal deployment script."""
    parser = argparse.ArgumentParser(description="Deploy a Megatron multimodal model using Ray")
    parser.add_argument(
        "--megatron_checkpoint",
        type=str,
        required=True,
        help="Path to the Megatron checkpoint directory",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="Number of GPUs to use per node",
    )
    parser.add_argument(
        "--tensor_model_parallel_size",
        type=int,
        default=1,
        help="Size of the tensor model parallelism",
    )
    parser.add_argument(
        "--pipeline_model_parallel_size",
        type=int,
        default=1,
        help="Size of the pipeline model parallelism",
    )
    parser.add_argument(
        "--params_dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Data type for model parameters",
    )
    parser.add_argument(
        "--inference_batch_times_seqlen_threshold",
        type=int,
        default=1000,
        help="Inference batch times sequence length threshold",
    )
    parser.add_argument(
        "--inference_max_seq_length",
        type=int,
        default=8192,
        help="Maximum sequence length for inference",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="megatron-model",
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
        "--num_cpus",
        type=int,
        default=None,
        help="Number of CPUs to allocate for the Ray cluster. If None, will use all available CPUs.",
    )
    parser.add_argument(
        "--num_cpus_per_replica",
        type=float,
        default=8,
        help="Number of CPUs per model replica",
    )
    parser.add_argument(
        "--include_dashboard",
        action="store_true",
        help="Whether to include the Ray dashboard",
    )
    parser.add_argument(
        "--cuda_visible_devices",
        type=str,
        default=None,
        help="Comma-separated list of CUDA visible devices",
    )
    parser.add_argument(
        "--num_replicas",
        type=int,
        default=1,
        help="Number of replicas for the deployment",
    )
    return parser.parse_args()


def main():
    """Main function to deploy a Megatron multimodal model using Ray."""
    args = parse_args()

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Initialize Ray deployment
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

    # Convert dtype string to torch dtype
    import torch

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    params_dtype = dtype_map[args.params_dtype]

    model_config_kwargs = {
        "params_dtype": params_dtype,
        "inference_batch_times_seqlen_threshold": args.inference_batch_times_seqlen_threshold,
        "inference_max_seq_length": args.inference_max_seq_length,
    }

    # Deploy the multimodal model
    ray_deployer.deploy_vlm_inframework_model(
        megatron_checkpoint=args.megatron_checkpoint,
        num_gpus=args.num_gpus,
        tensor_model_parallel_size=args.tensor_model_parallel_size,
        pipeline_model_parallel_size=args.pipeline_model_parallel_size,
        model_id=args.model_id,
        num_cpus_per_replica=args.num_cpus_per_replica,
        num_replicas=args.num_replicas,
        **model_config_kwargs,
    )


if __name__ == "__main__":
    main()
