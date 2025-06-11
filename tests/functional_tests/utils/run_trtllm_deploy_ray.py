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
import json
import logging
import multiprocessing
import os
import sys
import time
from pathlib import Path

import torch

# TensorRT-LLM Ray deployment imports
run_ray_tests = True
try:
    from nemo_deploy.deploy_ray import DeployRay
    from nemo_export.tensorrt_llm import TensorRTLLM
    from nemo_export.tensorrt_llm_deployable_ray import TensorRTLLMRayDeployable
    from ray import serve
except Exception as e:
    print(f"TensorRT-LLM Ray dependencies not available: {e}")
    run_ray_tests = False

LOGGER = logging.getLogger("NeMo")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


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


def test_deployment_handle_direct(deployment_handle, model_id, endpoint_type="completions", max_retries=10, retry_delay=2):
    """Test deployment handle directly without HTTP requests."""
    print(f"Testing {endpoint_type} endpoint via deployment handle...")
    
    for attempt in range(max_retries):
        try:
            if endpoint_type == "completions":
                payload = {
                    "model": model_id,
                    "prompt": "The capital of France is",
                    "max_tokens": 10,
                    "temperature": 0.1,
                }
                response = deployment_handle.completions.remote(payload).result()
                    
            elif endpoint_type == "chat":
                payload = {
                    "model": model_id,
                    "messages": "Hello, how are you?",
                    "max_tokens": 10,
                    "temperature": 0.1,
                    "apply_chat_template": False
                }
                response = deployment_handle.chat_completions.remote(payload).result()
                    
            elif endpoint_type == "models":
                response = deployment_handle.list_models.remote().result()
                    
            elif endpoint_type == "health":
                response = deployment_handle.health_check.remote().result()
            else:
                raise ValueError(f"Unknown endpoint type: {endpoint_type}")
            
            if response and (isinstance(response, dict) or hasattr(response, '__dict__')):
                print(f"✓ {endpoint_type.capitalize()} endpoint is responsive")
                return True, response
            else:
                print(f"✗ {endpoint_type.capitalize()} endpoint returned invalid response: {response}")
                if attempt < max_retries - 1:
                    print(f"Retrying in {retry_delay} seconds... (attempt {attempt + 1}/{max_retries})")
                    time.sleep(retry_delay)
                
        except Exception as e:
            print(f"✗ Error calling {endpoint_type} endpoint: {e}")
            if attempt < max_retries - 1:
                print(f"Retrying in {retry_delay} seconds... (attempt {attempt + 1}/{max_retries})")
                time.sleep(retry_delay)
    
    print(f"✗ {endpoint_type.capitalize()} endpoint failed after {max_retries} attempts")
    return False, None


def run_comprehensive_deployment_tests(deployment_handle, model_id):
    """Run comprehensive tests on deployment handle directly."""
    print("\n" + "="*60)
    print("RUNNING COMPREHENSIVE TRTLLM DEPLOYMENT HANDLE TESTS")
    print("="*60)
    
    test_results = {}
    endpoints = ["health", "models", "completions", "chat"]
    
    for endpoint in endpoints:
        print(f"\n--- Testing {endpoint} endpoint ---")
        success, response = test_deployment_handle_direct(deployment_handle, model_id, endpoint)
        test_results[endpoint] = {
            "success": success,
            "response": response
        }
        
        if success and response:
            print(f"✓ {endpoint.capitalize()} endpoint test PASSED")
            if endpoint in ["completions", "chat"]:
                if isinstance(response, dict) and "choices" in response:
                    choice_content = response['choices'][0].get('text', response['choices'][0].get('message', {}))
                    print(f"  Sample response: {choice_content}")
                else:
                    print(f"  Response type: {type(response)}")
        else:
            print(f"✗ {endpoint.capitalize()} endpoint test FAILED")
    
    return test_results


def convert_nemo_to_trtllm(
    nemo_checkpoint_path,
    trt_llm_path,
    model_type="llama",
    tensor_parallelism_size=1,
    pipeline_parallelism_size=1,
    max_input_len=2048,
    max_output_len=1024,
    max_batch_size=8,
    use_python_runtime=True,
    multi_block_mode=False,
    enable_chunked_context=False,
    max_tokens_in_paged_kv_cache=None,
    lora_ckpt_list=None,
    debug=True
):
    """Convert NeMo checkpoint to TensorRT-LLM model."""
    
    if debug:
        print(f"\nConverting NeMo checkpoint to TensorRT-LLM...")
        print(f"Source: {nemo_checkpoint_path}")
        print(f"Target: {trt_llm_path}")
        print(f"Model type: {model_type}")
        print(f"Tensor parallelism: {tensor_parallelism_size}")
        print(f"Pipeline parallelism: {pipeline_parallelism_size}")
        print(f"Max input length: {max_input_len}")
        print(f"Max output length: {max_output_len}")
        print(f"Max batch size: {max_batch_size}")
        print(f"Python runtime: {use_python_runtime}")
    
    try:
        # Create TensorRT-LLM directory
        Path(trt_llm_path).mkdir(parents=True, exist_ok=True)
        
        # Prepare TensorRTLLM constructor arguments
        trtllm_kwargs = {
            "model_dir": trt_llm_path,
            "lora_ckpt_list": lora_ckpt_list,
            "load_model": False,
            "use_python_runtime": use_python_runtime,
            "multi_block_mode": multi_block_mode,
        }
        
        # Add C++ runtime specific options if using C++ runtime
        if not use_python_runtime:
            trtllm_kwargs["enable_chunked_context"] = enable_chunked_context
            trtllm_kwargs["max_tokens_in_paged_kv_cache"] = max_tokens_in_paged_kv_cache
        
        trtllm_converter = TensorRTLLM(**trtllm_kwargs)
        
        # Export NeMo checkpoint to TensorRT-LLM
        if debug:
            print("Starting NeMo to TensorRT-LLM conversion...")
        
        trtllm_converter.export(
            nemo_checkpoint_path=nemo_checkpoint_path,
            model_type=model_type,
            tensor_parallelism_size=tensor_parallelism_size,
            pipeline_parallelism_size=pipeline_parallelism_size,
            max_input_len=max_input_len,
            max_output_len=max_output_len,
            max_batch_size=max_batch_size,
            delete_existing_files=True,
            max_seq_len=max_input_len + max_output_len
        )
        
        del trtllm_converter
        
        # Check the engine configuration after export
        engine_dir = os.path.join(trt_llm_path, "engines")
        if os.path.exists(engine_dir):
            if debug:
                print("Verifying engine configuration...")
            actual_max_input_len = check_engine_config(engine_dir)
            if actual_max_input_len != max_input_len:
                LOGGER.warning(
                    f"Engine max_input_len ({actual_max_input_len}) does not match "
                    f"expected value ({max_input_len}). This may cause runtime errors."
                )
            else:
                if debug:
                    print(f"✓ Engine configuration verified: max_input_len = {actual_max_input_len}")
        else:
            LOGGER.warning(f"Engine directory not found at: {engine_dir}")
        
        if debug:
            print("✓ NeMo to TensorRT-LLM conversion completed successfully")
        
        return True
        
    except Exception as e:
        LOGGER.error(f"Error converting NeMo checkpoint to TensorRT-LLM: {str(e)}")
        return False


def run_trtllm_ray_inference(
    model_name,
    nemo_checkpoint_path,
    trt_llm_path=None,
    model_type="llama",
    tensor_parallelism_size=1,
    pipeline_parallelism_size=1,
    max_input_len=2048,
    max_output_len=1024,
    max_batch_size=8,
    use_python_runtime=True,
    multi_block_mode=False,
    enable_chunked_context=False,
    max_tokens_in_paged_kv_cache=None,
    lora_ckpt_list=None,
    num_gpus=1,
    num_replicas=1,
    host="0.0.0.0",
    port=8000,
    num_cpus=None,
    num_cpus_per_replica=8,
    num_gpus_per_replica=1,
    include_dashboard=False,
    cuda_visible_devices="0,1",
    test_endpoints=True,
    deployment_timeout=300,
    debug=True,
):
    """Deploy a TensorRT-LLM model (converted from NeMo) on Ray cluster and test all endpoints."""
    
    if not run_ray_tests:
        print("TensorRT-LLM Ray dependencies not available. Skipping TensorRT-LLM Ray tests.")
        return None
    
    if not Path(nemo_checkpoint_path).exists():
        raise Exception(f"NeMo checkpoint {nemo_checkpoint_path} could not be found.")
    
    if num_gpus > torch.cuda.device_count():
        print(
            f"Model: {model_name} with {num_gpus} gpus won't be tested since available # of gpus = {torch.cuda.device_count()}"
        )
        return None
    
    # Set default TensorRT-LLM path if not provided
    if trt_llm_path is None:
        trt_llm_path = f"/tmp/trt_llm_{model_name}_model_dir/"
    
    if debug:
        print("")
        print("=" * 80)
        print("NEW TRTLLM RAY DEPLOYMENT TEST")
        print("=" * 80)
        print(f"Model: {model_name}")
        print(f"NeMo Checkpoint: {nemo_checkpoint_path}")
        print(f"TensorRT-LLM Path: {trt_llm_path}")
        print(f"GPUs: {num_gpus}")
        print(f"Replicas: {num_replicas}")
        print(f"Host: {host}:{port}")

    # If num_cpus is not specified, use all available CPUs
    if num_cpus is None:
        num_cpus = get_available_cpus()
        if debug:
            print(f"Using all available CPUs: {num_cpus}")

    # Validate C++ runtime specific options
    if use_python_runtime and (enable_chunked_context or max_tokens_in_paged_kv_cache):
        raise Exception(
            "enable_chunked_context and max_tokens_in_paged_kv_cache options "
            "work only with the TensorRT-LLM C++ runtime. Please set use_python_runtime=False."
        )

    # Convert NeMo checkpoint to TensorRT-LLM
    conversion_success = convert_nemo_to_trtllm(
        nemo_checkpoint_path=nemo_checkpoint_path,
        trt_llm_path=trt_llm_path,
        model_type=model_type,
        tensor_parallelism_size=tensor_parallelism_size,
        pipeline_parallelism_size=pipeline_parallelism_size,
        max_input_len=max_input_len,
        max_output_len=max_output_len,
        max_batch_size=max_batch_size,
        use_python_runtime=use_python_runtime,
        multi_block_mode=multi_block_mode,
        enable_chunked_context=enable_chunked_context,
        max_tokens_in_paged_kv_cache=max_tokens_in_paged_kv_cache,
        lora_ckpt_list=lora_ckpt_list,
        debug=debug
    )
    
    if not conversion_success:
        print("✗ Failed to convert NeMo checkpoint to TensorRT-LLM")
        return None

    # Initialize Ray deployment
    ray_deployer = DeployRay(
        num_cpus=num_cpus,
        num_gpus=num_gpus,
        include_dashboard=include_dashboard,
        runtime_env={
            "env_vars": {
                "CUDA_VISIBLE_DEVICES": cuda_visible_devices,
            }
        }
    )

    deployment_success = False
    test_results = {}
    deployment_handle = None
    
    try:
        # Start Ray Serve
        if debug:
            print(f"Starting Ray Serve on {host}:{port}...")
        ray_deployer.start(host=host, port=port)
        
        # Create the TensorRT-LLM model deployment
        if debug:
            print("Creating TensorRT-LLM Ray deployable...")
        
        # Prepare deployment parameters
        deployment_kwargs = {
            "trt_llm_path": trt_llm_path,
            "model_id": model_name,
            "use_python_runtime": use_python_runtime,
            "multi_block_mode": multi_block_mode,
            "lora_ckpt_list": lora_ckpt_list,
        }

        # Add C++ runtime specific options if using C++ runtime
        if not use_python_runtime:
            deployment_kwargs["enable_chunked_context"] = enable_chunked_context
            deployment_kwargs["max_tokens_in_paged_kv_cache"] = max_tokens_in_paged_kv_cache

        app = TensorRTLLMRayDeployable.options(
            num_replicas=num_replicas,
            ray_actor_options={
                "num_gpus": num_gpus_per_replica,
                "num_cpus": num_cpus_per_replica,
            },
        ).bind(**deployment_kwargs)

        # Deploy the model and get handle
        if debug:
            print("Deploying TensorRT-LLM model...")
        
        # Deploy using serve.run and get the deployment handle
        serve.run(app, name=model_name)
        
        # Get the app handle (not deployment handle) - this is the correct approach
        deployment_handle = serve.get_app_handle(model_name)
        deployment_success = True

        if debug:
            print(f"✓ TensorRT-LLM model deployed successfully at {host}:{port}")
            print(f"✓ App handle obtained: {deployment_handle}")

        # Wait a moment for deployment to be fully ready
        if debug:
            print("Waiting for deployment to be fully ready...")
        time.sleep(5)
        
        # Test endpoints if requested using deployment handle
        if test_endpoints:
            test_results = run_comprehensive_deployment_tests(deployment_handle, model_name)
        
        return test_results

    except Exception as e:
        print(f"✗ Error during TensorRT-LLM deployment: {str(e)}")
        return None
    finally:
        if deployment_success:
            print("Shutting down TensorRT-LLM Ray deployment...")
            ray_deployer.stop()


def get_args():
    """Parse command-line arguments for the TensorRT-LLM Ray deployment test script."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Deploy TensorRT-LLM models (converted from NeMo) to Ray cluster and test endpoints",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Name of the model to deploy",
    )
    parser.add_argument(
        "--nemo_checkpoint_path",
        type=str,
        required=True,
        help="Path to the .nemo checkpoint file to convert to TensorRT-LLM",
    )
    parser.add_argument(
        "--trt_llm_path",
        type=str,
        default=None,
        help="Path to store the converted TensorRT-LLM model (default: /tmp/trt_llm_{model_name}_model_dir/)",
    )
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
        help="Size of the tensor model parallelism",
    )
    parser.add_argument(
        "--pipeline_parallelism_size",
        type=int,
        default=1,
        help="Size of the pipeline model parallelism",
    )
    parser.add_argument(
        "--max_input_len",
        type=int,
        default=2048,
        help="Maximum input sequence length in tokens",
    )
    parser.add_argument(
        "--max_output_len",
        type=int,
        default=1024,
        help="Maximum output sequence length in tokens",
    )
    parser.add_argument(
        "--max_batch_size",
        type=int,
        default=8,
        help="Maximum number of requests to batch together",
    )
    parser.add_argument(
        "--use_python_runtime",
        action="store_true",
        help="Whether to use Python runtime (default: True unless --use_cpp_runtime is set)",
    )
    parser.add_argument(
        "--use_cpp_runtime",
        action="store_true",
        help="Whether to use C++ runtime (overrides use_python_runtime)",
    )
    parser.add_argument(
        "--multi_block_mode",
        action="store_true",
        help="Whether to enable multi-block mode",
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
        "--lora_ckpt_list",
        type=str,
        nargs="*",
        default=None,
        help="List of LoRA checkpoint paths",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="Number of GPUs to use for deployment",
    )
    parser.add_argument(
        "--num_replicas",
        type=int,
        default=1,
        help="Number of replicas for the deployment",
    )
    parser.add_argument(
        "--num_gpus_per_replica",
        type=float,
        default=1,
        help="Number of GPUs per model replica",
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
        default=8000,
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
        default="0,1",
        help="Comma-separated list of CUDA visible devices",
    )
    parser.add_argument(
        "--test_endpoints",
        type=str,
        default="True",
        help="Whether to test all endpoints after deployment (True/False)",
    )
    parser.add_argument(
        "--deployment_timeout",
        type=int,
        default=300,
        help="Timeout in seconds for deployment readiness",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug output",
    )

    return parser.parse_args()


def run_trtllm_ray_deployment_tests(args):
    """Run TensorRT-LLM Ray deployment tests with the provided arguments."""
    
    # Convert string arguments to boolean
    if args.test_endpoints == "True":
        args.test_endpoints = True
    else:
        args.test_endpoints = False

    # Handle runtime selection
    # Default to Python runtime unless C++ runtime is explicitly requested
    use_python_runtime = not args.use_cpp_runtime

    print("\n" + "="*80)
    print("TRTLLM RAY DEPLOYMENT TEST SUITE")
    print("="*80)
    print(f"Model: {args.model_name}")
    print(f"NeMo Checkpoint: {args.nemo_checkpoint_path}")
    print(f"TensorRT-LLM Path: {args.trt_llm_path}")
    print(f"Configuration: {args.num_replicas} replicas, {args.num_gpus} GPUs")
    print(f"Runtime: {'Python' if use_python_runtime else 'C++'}")
    print("="*80)

    try:
        test_results = run_trtllm_ray_inference(
            model_name=args.model_name,
            nemo_checkpoint_path=args.nemo_checkpoint_path,
            trt_llm_path=args.trt_llm_path,
            model_type=args.model_type,
            tensor_parallelism_size=args.tensor_parallelism_size,
            pipeline_parallelism_size=args.pipeline_parallelism_size,
            max_input_len=args.max_input_len,
            max_output_len=args.max_output_len,
            max_batch_size=args.max_batch_size,
            use_python_runtime=use_python_runtime,
            multi_block_mode=args.multi_block_mode,
            enable_chunked_context=args.enable_chunked_context,
            max_tokens_in_paged_kv_cache=args.max_tokens_in_paged_kv_cache,
            lora_ckpt_list=args.lora_ckpt_list,
            num_gpus=args.num_gpus,
            num_replicas=args.num_replicas,
            num_gpus_per_replica=args.num_gpus_per_replica,
            host=args.host,
            port=args.port,
            num_cpus=args.num_cpus,
            num_cpus_per_replica=args.num_cpus_per_replica,
            include_dashboard=args.include_dashboard,
            cuda_visible_devices=args.cuda_visible_devices,
            test_endpoints=args.test_endpoints,
            deployment_timeout=args.deployment_timeout,
            debug=args.debug,
        )

        # Print test summary
        print("\n" + "="*80)
        print("TEST SUMMARY")
        print("="*80)
        
        if test_results is None:
            print("✗ DEPLOYMENT FAILED")
            test_result = "FAIL"
        elif not args.test_endpoints:
            print("✓ DEPLOYMENT SUCCESSFUL (endpoint testing skipped)")
            test_result = "PASS"
        else:
            passed_endpoints = sum(1 for result in test_results.values() if result["success"])
            total_endpoints = len(test_results)
            
            print(f"Endpoint Tests: {passed_endpoints}/{total_endpoints} passed")
            
            for endpoint, result in test_results.items():
                status = "✓ PASS" if result["success"] else "✗ FAIL"
                print(f"  {endpoint.capitalize()}: {status}")
            
            if passed_endpoints == total_endpoints:
                test_result = "PASS"
                print("\n✓ ALL TESTS PASSED")
            else:
                test_result = "FAIL"
                print(f"\n✗ {total_endpoints - passed_endpoints} TESTS FAILED")

        print("="*80)
        print(f"FINAL RESULT: {test_result}")
        print("="*80)

        if test_result == "FAIL":
            raise Exception("One or more tests failed")

    except Exception as e:
        print(f"\n✗ TEST SUITE FAILED: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    args = get_args()
    run_trtllm_ray_deployment_tests(args) 