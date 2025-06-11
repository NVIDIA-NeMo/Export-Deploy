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
import signal
import sys
import json
import os
from pathlib import Path
from nemo_deploy.deploy_ray import DeployRay
from nemo_export.tensorrt_llm import TensorRTLLM
from nemo_export.tensorrt_llm_deployable_ray import TensorRTLLMRayDeployable

LOGGER = logging.getLogger("NeMo")


def get_available_cpus():
    """Get the total number of available CPUs in the system."""
    return multiprocessing.cpu_count()


def check_engine_config(engine_dir):
    """Check the engine configuration to verify max_input_len."""
    config_path = os.path.join(engine_dir, "config.json")
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                config = json.load(f)
            max_input_len = config.get("build_config", {}).get("max_input_len", "NOT_FOUND")
            max_batch_size = config.get("build_config", {}).get("max_batch_size", "NOT_FOUND")
            LOGGER.info(f"Engine config check - max_input_len: {max_input_len}, max_batch_size: {max_batch_size}")
            return max_input_len
        except Exception as e:
            LOGGER.error(f"Error reading engine config: {e}")
            return None
    else:
        LOGGER.warning(f"Engine config file not found at: {config_path}")
        return None


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Deploy a TensorRT-LLM model using Ray")
    
    # Model path arguments (at least one required)
    model_group = parser.add_mutually_exclusive_group(required=True)
    model_group.add_argument(
        "--trt_llm_path",
        type=str,
        default=None,
        help="Path to the TensorRT-LLM model directory with pre-built engines",
    )
    model_group.add_argument(
        "--nemo_checkpoint_path",
        type=str,
        default=None,
        help="Path to the NeMo checkpoint file to be exported to TensorRT-LLM",
    )
    model_group.add_argument(
        "--hf_model_path",
        type=str,
        default=None,
        help="Path to the HuggingFace model to be exported to TensorRT-LLM",
    )
    
    # Model configuration
    parser.add_argument(
        "--model_type",
        type=str,
        default="llama",
        help="Model type/architecture (e.g., 'llama', 'gpt')",
    )
    parser.add_argument(
        "--tensor_parallelism_size",
        type=int,
        default=1,
        help="Number of tensor parallelism",
    )
    parser.add_argument(
        "--pipeline_parallelism_size",
        type=int,
        default=1,
        help="Number of pipeline parallelism",
    )
    parser.add_argument(
        "--max_batch_size",
        type=int,
        default=8,
        help="Maximum number of requests to batch together",
    )
    parser.add_argument(
        "--max_input_len",
        type=int,
        default=2048,
        help="Maximum input sequence length in tokens (default: 2048)",
    )
    parser.add_argument(
        "--max_output_len",
        type=int,
        default=1024,
        help="Maximum output sequence length in tokens (default: 1024)",
    )
    parser.add_argument(
        "--use_python_runtime",
        action="store_true",
        help="Whether to use Python runtime (default: True)",
    )
    parser.add_argument(
        "--use_cpp_runtime",
        action="store_true",
        help="Whether to use C++ runtime (overrides use_python_runtime)",
    )
    parser.add_argument(
        "--enable_chunked_context",
        action="store_true",
        help="Whether to enable chunked context (C++ runtime only)",
    )
    parser.add_argument(
        "--max_tokens_in_paged_kv_cache",
        type=int,
        default=None,
        help="Maximum tokens in paged KV cache (C++ runtime only)",
    )
    parser.add_argument(
        "--multi_block_mode",
        action="store_true",
        help="Whether to enable multi-block mode",
    )
    parser.add_argument(
        "--lora_ckpt_list",
        type=str,
        nargs="*",
        default=None,
        help="List of LoRA checkpoint paths",
    )
    
    # API configuration
    parser.add_argument(
        "--model_id",
        type=str,
        default="tensorrt-llm-model",
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
    
    # Ray cluster configuration
    parser.add_argument(
        "--num_cpus",
        type=int,
        default=None,
        help="Number of CPUs to allocate for the Ray cluster. If None, will use all available CPUs.",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="Number of GPUs to allocate for the Ray cluster",
    )
    parser.add_argument(
        "--include_dashboard",
        action="store_true",
        help="Whether to include the Ray dashboard",
    )
    parser.add_argument(
        "--num_replicas",
        type=int,
        default=1,
        help="Number of model replicas to deploy",
    )
    parser.add_argument(
        "--num_gpus_per_replica",
        type=float,
        default=1,
        help="Number of GPUs per model replica",
    )
    parser.add_argument(
        "--num_cpus_per_replica",
        type=float,
        default=8,
        help="Number of CPUs per model replica",
    )
    parser.add_argument(
        "--cuda_visible_devices",
        type=str,
        default="0,1",
        help="Comma-separated list of CUDA visible devices",
    )
    
    return parser.parse_args()


def signal_handler(signum, frame, deployer):
    """Handle interrupt signals."""
    LOGGER.info("Received interrupt signal. Shutting down gracefully...")
    deployer.stop()
    sys.exit(0)


def main():
    args = parse_args()

    # If num_cpus is not specified, use all available CPUs
    if args.num_cpus is None:
        args.num_cpus = get_available_cpus()
        LOGGER.info(f"Using all available CPUs: {args.num_cpus}")

    # Handle runtime selection
    # Default to Python runtime unless C++ runtime is explicitly requested
    use_python_runtime = not args.use_cpp_runtime

    # Validate C++ runtime specific options
    if use_python_runtime and (args.enable_chunked_context or args.max_tokens_in_paged_kv_cache):
        LOGGER.error(
            "enable_chunked_context and max_tokens_in_paged_kv_cache options "
            "work only with the TensorRT-LLM C++ runtime. Please use --use_cpp_runtime."
        )
        sys.exit(1)

    try:
        if not args.nemo_checkpoint_path and not args.hf_model_path and not args.trt_llm_path:
            raise ValueError("Either nemo_checkpoint_path or hf_model_path or trt_llm_path must be provided for deployment")
        if not args.trt_llm_path:
            args.trt_llm_path = "/tmp/trt_llm_model_dir/"
            LOGGER.info(
                    "/tmp/trt_llm_model_dir/ path will be used as the TensorRT LLM folder. "
                    "Please set the --triton_model_repository parameter if you'd like to use a path that already "
                    "includes the TensorRT LLM model files."
            )
            Path(args.trt_llm_path).mkdir(parents=True, exist_ok=True)
            
            # Prepare TensorRTLLM constructor arguments
            trtllm_kwargs = {
                "model_dir": args.trt_llm_path,
                "lora_ckpt_list": args.lora_ckpt_list,
                "load_model": False,
                "use_python_runtime": use_python_runtime,
                "multi_block_mode": args.multi_block_mode,
            }
            
            # Add C++ runtime specific options if using C++ runtime
            if not use_python_runtime:
                trtllm_kwargs["enable_chunked_context"] = args.enable_chunked_context
                trtllm_kwargs["max_tokens_in_paged_kv_cache"] = args.max_tokens_in_paged_kv_cache
            
            trtllmConverter = TensorRTLLM(**trtllm_kwargs)
            
            if args.nemo_checkpoint_path:
                LOGGER.info("Exporting Nemo checkpoint to TensorRT-LLM")
                try:
                    trtllmConverter.export(
                        nemo_checkpoint_path=args.nemo_checkpoint_path,
                        model_type=args.model_type,
                        tensor_parallelism_size=args.tensor_parallelism_size,
                        pipeline_parallelism_size=args.pipeline_parallelism_size,
                        max_input_len=args.max_input_len,
                        max_output_len=args.max_output_len,
                        max_batch_size=args.max_batch_size,
                        delete_existing_files=True,
                        max_seq_len=args.max_input_len + args.max_output_len
                    )
                except Exception as e:
                    LOGGER.error(f"Error exporting Nemo checkpoint to TensorRT-LLM: {str(e)}")
                    raise RuntimeError(f"Error exporting Nemo checkpoint to TensorRT-LLM: {str(e)}")
            elif args.hf_model_path:
                LOGGER.info("Exporting HF model to TensorRT-LLM")
                try:
                    trtllmConverter.export_hf_model(
                        hf_model_path=args.hf_model_path,
                        max_batch_size=args.max_batch_size,
                        tensor_parallelism_size=args.tensor_parallelism_size,
                        max_input_len=args.max_input_len,
                        max_output_len=args.max_output_len,
                        delete_existing_files=True,
                        max_seq_len=args.max_input_len + args.max_output_len
                    )
                except Exception as e:
                    LOGGER.error(f"Error exporting HF model to TensorRT-LLM: {str(e)}")
                    raise RuntimeError(f"Error exporting HF model to TensorRT-LLM: {str(e)}")
            del trtllmConverter
    except Exception as e:
        LOGGER.error(f"Error during TRTLLM model export: {str(e)}")
        sys.exit(1)

    # Check the engine configuration after export
    engine_dir = os.path.join(args.trt_llm_path, "engines")
    if os.path.exists(engine_dir):
        LOGGER.info("Checking engine configuration after export...")
        actual_max_input_len = check_engine_config(engine_dir)
        if actual_max_input_len != args.max_input_len:
            LOGGER.warning(
                f"Engine max_input_len ({actual_max_input_len}) does not match "
                f"expected value ({args.max_input_len}). This may cause runtime errors."
            )
        else:
            LOGGER.info(f"Engine configuration verified: max_input_len = {actual_max_input_len}")
    else:
        LOGGER.warning(f"Engine directory not found at: {engine_dir}")

    # Initialize Ray deployment
    ray_deployer = DeployRay(
        num_cpus=args.num_cpus,
        num_gpus=args.num_gpus,
        include_dashboard=args.include_dashboard,
        runtime_env={
            "env_vars": {
                "CUDA_VISIBLE_DEVICES": args.cuda_visible_devices,
            }
        },
    )

    # Set up signal handlers
    signal.signal(
        signal.SIGINT, lambda signum, frame: signal_handler(signum, frame, ray_deployer)
    )
    signal.signal(
        signal.SIGTERM,
        lambda signum, frame: signal_handler(signum, frame, ray_deployer),
    )
    # Start Ray Serve
    ray_deployer.start(host=args.host, port=args.port)

    # Prepare deployment parameters
    deployment_kwargs = {
            "trt_llm_path": args.trt_llm_path,
            "model_id": args.model_id,
            "use_python_runtime": use_python_runtime,
            "multi_block_mode": args.multi_block_mode,
            "lora_ckpt_list": args.lora_ckpt_list,
    }

    # Add C++ runtime specific options if using C++ runtime
    if not use_python_runtime:
        deployment_kwargs["enable_chunked_context"] = args.enable_chunked_context
        deployment_kwargs["max_tokens_in_paged_kv_cache"] = args.max_tokens_in_paged_kv_cache

    # Create the TensorRT-LLM model deployment
    app = TensorRTLLMRayDeployable.options(
        num_replicas=args.num_replicas,
        ray_actor_options={
            "num_gpus": args.num_gpus_per_replica,
            "num_cpus": args.num_cpus_per_replica,
        },
    ).bind(**deployment_kwargs)

    # Deploy the model
    ray_deployer.run(app, args.model_id)

    LOGGER.info(f"TensorRT-LLM model deployed successfully at {args.host}:{args.port}")
    LOGGER.info("Press Ctrl+C to stop the deployment")

    # Keep the script running
    while True:
        signal.pause()


if __name__ == "__main__":
    main() 