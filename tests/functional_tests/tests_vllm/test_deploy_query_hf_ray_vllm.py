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

import logging
import subprocess
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from tests.functional_tests.utils.ray_test_utils import (
    query_ray_deployment,
    terminate_deployment_process,
    wait_for_deployment_ready,
)


class TestDeployRayHFVLLM:
    def setup_method(self):
        """Setup for each test method."""
        self.deploy_proc = None

    def teardown_method(self):
        """Cleanup after each test method."""
        if self.deploy_proc is not None:
            terminate_deployment_process(self.deploy_proc)
            # Avoid double termination in case test used finally to clean up
            self.deploy_proc = None

    def test_deploy_ray_hf_vllm_backend(self):
        """Test deploying HuggingFace model with vLLM backend using Ray."""
        hf_model_path = "meta-llama/Llama-3.2-1B"

        try:
            # Run Ray deployment for HF model with vLLM backend
            self.deploy_proc = subprocess.Popen(
                [
                    "coverage",
                    "run",
                    "--data-file=/workspace/.coverage",
                    "--source=/workspace/",
                    "--parallel-mode",
                    "scripts/deploy/llm/deploy_ray_hf.py",
                    "--model_path",
                    hf_model_path,
                    "--task",
                    "text-generation",
                    "--model_id",
                    "hf-llm-vllm",
                    "--num_gpus",
                    str(1),
                    "--host",
                    "0.0.0.0",
                    "--port",
                    str(8002),
                    "--trust_remote_code",
                    "--cuda_visible_devices",
                    "0",
                    "--use_vllm_backend",
                ]
            )
            print("HF Deployment with vLLM backend started. Waiting for it to be ready...")

            # Wait for deployment to be ready
            if not wait_for_deployment_ready(host="0.0.0.0", port=8002, max_wait_time=300):
                assert False, "Deployment failed to become ready within timeout"

            time.sleep(20)

            # Test basic completion endpoint
            output = query_ray_deployment(
                host="0.0.0.0",
                port=8002,
                model_id="hf-llm-vllm",
                prompt="What is the color of a banana?",
                max_tokens=20,
            )

            print(f"Basic completion response: {output}")

            # Check if deployment was successful
            assert output != "", "First prediction is empty"

            # Test chat completion endpoint
            output_chat = query_ray_deployment(
                host="0.0.0.0",
                port=8002,
                model_id="hf-llm-vllm",
                prompt=[{"role": "user", "content": "Hello, how are you?"}],
                max_tokens=20,
                use_chat=True,
            )
            print(f"Chat completion response: {output_chat}")

            # Check if deployment was successful
            assert output_chat != "", "Second prediction (chat) is empty"

            # Test with different temperature
            output_temp = query_ray_deployment(
                host="0.0.0.0",
                port=8002,
                model_id="hf-llm-vllm",
                prompt="Tell me a short story about a cat.",
                max_tokens=30,
                temperature=0.9,
            )
            print(f"High temperature response: {output_temp}")

            # Check if deployment was successful
            assert output_temp != "", "High temperature prediction is empty"

        finally:
            # Ensure the deployment is terminated as soon as queries complete or on failure
            if self.deploy_proc is not None:
                terminate_deployment_process(self.deploy_proc)
                self.deploy_proc = None

    def test_deploy_ray_hf_vllm_backend_with_parameters(self):
        """Test deploying HuggingFace model with vLLM backend and custom parameters."""
        hf_model_path = "meta-llama/Llama-3.2-1B"

        try:
            # Run Ray deployment for HF model with vLLM backend and custom parameters
            self.deploy_proc = subprocess.Popen(
                [
                    "coverage",
                    "run",
                    "--data-file=/workspace/.coverage",
                    "--source=/workspace/",
                    "--parallel-mode",
                    "scripts/deploy/llm/deploy_ray_hf.py",
                    "--model_path",
                    hf_model_path,
                    "--task",
                    "text-generation",
                    "--model_id",
                    "hf-llm-vllm-params",
                    "--num_gpus",
                    str(1),
                    "--host",
                    "0.0.0.0",
                    "--port",
                    str(8003),
                    "--trust_remote_code",
                    "--cuda_visible_devices",
                    "0",
                    "--use_vllm_backend",
                    "--num_replicas",
                    str(1),
                    "--num_gpus_per_replica",
                    str(1),
                    "--num_cpus_per_replica",
                    str(4),
                    "--max_ongoing_requests",
                    str(5),
                ]
            )
            print("HF Deployment with vLLM backend and custom parameters started. Waiting for it to be ready...")

            # Wait for deployment to be ready
            if not wait_for_deployment_ready(host="0.0.0.0", port=8003, max_wait_time=300):
                assert False, "Deployment failed to become ready within timeout"

            time.sleep(20)

            # Test multiple requests to verify the deployment handles them correctly
            prompts = [
                "What is 2+2?",
                "Name a fruit that is red.",
                "What is the capital of France?",
            ]

            for i, prompt in enumerate(prompts):
                output = query_ray_deployment(
                    host="0.0.0.0",
                    port=8003,
                    model_id="hf-llm-vllm-params",
                    prompt=prompt,
                    max_tokens=15,
                    temperature=0.7,
                )
                print(f"Request {i + 1} response: {output}")
                assert output != "", f"Prediction {i + 1} is empty"

        finally:
            # Ensure the deployment is terminated as soon as queries complete or on failure
            if self.deploy_proc is not None:
                terminate_deployment_process(self.deploy_proc)
                self.deploy_proc = None
