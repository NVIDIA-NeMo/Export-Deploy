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
import sys
import time
from pathlib import Path

import torch

# Ray deployment imports
run_ray_tests = True
try:
    from nemo_deploy.deploy_ray import DeployRay
    from nemo_deploy.nlp.megatronllm_deployable_ray import MegatronRayDeployable
    from ray import serve
except Exception as e:
    print(f"Ray dependencies not available: {e}")
    run_ray_tests = False

LOGGER = logging.getLogger("NeMo")
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


def get_available_cpus():
    """Get the total number of available CPUs in the system."""
    return multiprocessing.cpu_count()


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
    print("RUNNING COMPREHENSIVE DEPLOYMENT HANDLE TESTS")
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


def run_ray_inference(
    model_name,
    checkpoint_path,
    num_gpus=1,
    num_nodes=1,
    num_replicas=1,
    tensor_model_parallel_size=1,
    pipeline_model_parallel_size=1,
    expert_model_parallel_size=1,
    context_parallel_size=1,
    host="0.0.0.0",
    port=8000,
    num_cpus=None,
    num_cpus_per_replica=8,
    include_dashboard=False,
    cuda_visible_devices="0,1",
    enable_cuda_graphs=False,
    enable_flash_decode=False,
    legacy_ckpt=False,
    test_endpoints=True,
    deployment_timeout=300,
    debug=True,
):
    """Deploy a NeMo model on Ray cluster and test all endpoints."""
    
    if not run_ray_tests:
        print("Ray dependencies not available. Skipping Ray tests.")
        return None
    
    if not Path(checkpoint_path).exists():
        raise Exception(f"Checkpoint {checkpoint_path} could not be found.")
    
    if num_gpus > torch.cuda.device_count():
        print(
            f"Model: {model_name} with {num_gpus} gpus won't be tested since available # of gpus = {torch.cuda.device_count()}"
        )
        return None
    
    if debug:
        print("")
        print("=" * 80)
        print("NEW RAY DEPLOYMENT TEST")
        print("=" * 80)
        print(f"Model: {model_name}")
        print(f"Checkpoint: {checkpoint_path}")
        print(f"GPUs: {num_gpus}")
        print(f"Nodes: {num_nodes}")
        print(f"Replicas: {num_replicas}")
        print(f"Host: {host}:{port}")

    # If num_cpus is not specified, use all available CPUs
    if num_cpus is None:
        num_cpus = get_available_cpus()
        if debug:
            print(f"Using all available CPUs: {num_cpus}")

    # Calculate total GPUs and validate configuration
    total_gpus = num_gpus * num_nodes
    gpus_per_replica = total_gpus // num_replicas
    
    parallelism_per_replica = (
        tensor_model_parallel_size * 
        pipeline_model_parallel_size * 
        context_parallel_size
    )
    
    if parallelism_per_replica != gpus_per_replica:
        raise Exception(
            f"Parallelism per replica ({parallelism_per_replica}) must equal "
            f"GPUs per replica ({gpus_per_replica})"
        )

    if debug:
        print(f"Configuration: {num_replicas} replicas, {gpus_per_replica} GPUs per replica")

    # Initialize Ray deployment
    ray_deployer = DeployRay(
        num_cpus=num_cpus,
        num_gpus=total_gpus,
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
        
        # Create the Multi-Rank Megatron model deployment
        if debug:
            print("Creating Megatron Ray deployable...")
        app = MegatronRayDeployable.options(
            num_replicas=num_replicas,
            ray_actor_options={
                "num_cpus": num_cpus_per_replica
            }
        ).bind(
            nemo_checkpoint_filepath=checkpoint_path,
            num_gpus=gpus_per_replica,
            num_nodes=num_nodes,
            tensor_model_parallel_size=tensor_model_parallel_size,
            pipeline_model_parallel_size=pipeline_model_parallel_size,
            expert_model_parallel_size=expert_model_parallel_size,
            context_parallel_size=context_parallel_size,
            model_id=model_name,
            enable_cuda_graphs=enable_cuda_graphs,
            enable_flash_decode=enable_flash_decode,
            legacy_ckpt=legacy_ckpt
        )

        # Deploy the model and get handle
        if debug:
            print("Deploying model...")
        
        # Deploy using serve.run and get the deployment handle
        serve.run(app, name=model_name)
        
        # Get the app handle (not deployment handle) - this is the correct approach
        deployment_handle = serve.get_app_handle(model_name)
        deployment_success = True

        if debug:
            print(f"✓ Model deployed successfully at {host}:{port}")
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
        print(f"✗ Error during deployment: {str(e)}")
        return None
    finally:
        if deployment_success:
            print("Shutting down Ray deployment...")
            ray_deployer.stop()


def get_args():
    """Parse command-line arguments for the Ray deployment test script."""
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Deploy NeMo models to Ray cluster and test endpoints",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Name of the model to deploy",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the .nemo checkpoint file",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=1,
        help="Number of GPUs to use per node",
    )
    parser.add_argument(
        "--num_nodes",
        type=int,
        default=1,
        help="Number of nodes to use for deployment",
    )
    parser.add_argument(
        "--num_replicas",
        type=int,
        default=1,
        help="Number of replicas for the deployment",
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
        "--expert_model_parallel_size",
        type=int,
        default=1,
        help="Size of the expert model parallelism",
    )
    parser.add_argument(
        "--context_parallel_size",
        type=int,
        default=1,
        help="Size of the context parallelism",
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
        "--enable_cuda_graphs",
        action="store_true",
        help="Whether to enable CUDA graphs for faster inference",
    )
    parser.add_argument(
        "--enable_flash_decode",
        action="store_true",
        help="Whether to enable Flash Attention decode",
    )
    parser.add_argument(
        "--legacy_ckpt",
        action="store_true",
        help="Whether to use legacy checkpoint format",
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


def run_ray_deployment_tests(args):
    """Run Ray deployment tests with the provided arguments."""
    
    # Convert string arguments to boolean
    if args.test_endpoints == "True":
        args.test_endpoints = True
    else:
        args.test_endpoints = False

    print("\n" + "="*80)
    print("NEMO RAY DEPLOYMENT TEST SUITE")
    print("="*80)
    print(f"Model: {args.model_name}")
    print(f"Checkpoint: {args.checkpoint_path}")
    print(f"Configuration: {args.num_replicas} replicas, {args.num_gpus} GPUs per node")
    print("="*80)

    try:
        test_results = run_ray_inference(
            model_name=args.model_name,
            checkpoint_path=args.checkpoint_path,
            num_gpus=args.num_gpus,
            num_nodes=args.num_nodes,
            num_replicas=args.num_replicas,
            tensor_model_parallel_size=args.tensor_model_parallel_size,
            pipeline_model_parallel_size=args.pipeline_model_parallel_size,
            expert_model_parallel_size=args.expert_model_parallel_size,
            context_parallel_size=args.context_parallel_size,
            host=args.host,
            port=args.port,
            num_cpus=args.num_cpus,
            num_cpus_per_replica=args.num_cpus_per_replica,
            include_dashboard=args.include_dashboard,
            cuda_visible_devices=args.cuda_visible_devices,
            enable_cuda_graphs=args.enable_cuda_graphs,
            enable_flash_decode=args.enable_flash_decode,
            legacy_ckpt=args.legacy_ckpt,
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
    run_ray_deployment_tests(args)
