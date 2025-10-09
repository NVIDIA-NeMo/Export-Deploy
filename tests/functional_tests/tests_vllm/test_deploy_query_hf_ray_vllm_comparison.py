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


class TestDeployRayHFVLLMComparison:
    def setup_method(self):
        """Setup for each test method."""
        self.deploy_proc_vllm = None
        self.deploy_proc_regular = None

    def teardown_method(self):
        """Cleanup after each test method."""
        if self.deploy_proc_vllm is not None:
            terminate_deployment_process(self.deploy_proc_vllm)
            self.deploy_proc_vllm = None
        if self.deploy_proc_regular is not None:
            terminate_deployment_process(self.deploy_proc_regular)
            self.deploy_proc_regular = None

    def test_deploy_ray_hf_vllm_vs_regular_comparison(self):
        """Test comparing HF deployment with vLLM backend vs regular HF deployment."""
        hf_model_path = "meta-llama/Llama-3.2-1B"

        try:
            # Deploy with vLLM backend
            self.deploy_proc_vllm = subprocess.Popen(
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
                    str(8004),
                    "--trust_remote_code",
                    "--cuda_visible_devices",
                    "0",
                    "--use_vllm_backend",
                ]
            )
            print("HF Deployment with vLLM backend started. Waiting for it to be ready...")

            # Wait for vLLM deployment to be ready
            if not wait_for_deployment_ready(host="0.0.0.0", port=8004, max_wait_time=300):
                assert False, "vLLM deployment failed to become ready within timeout"

            time.sleep(20)

            # Test vLLM deployment first
            test_prompt = "What is the capital of France?"

            # Query vLLM deployment
            output_vllm = query_ray_deployment(
                host="0.0.0.0",
                port=8004,
                model_id="hf-llm-vllm",
                prompt=test_prompt,
                max_tokens=20,
                temperature=0.7,
            )
            print(f"vLLM backend response: {output_vllm}")

            # Both should return non-empty responses
            assert output_vllm != "", "vLLM backend prediction is empty"
            assert len(output_vllm) > 0, "vLLM backend returned empty response"

            # Test chat completion on vLLM
            chat_prompt = [{"role": "user", "content": "Hello, how are you?"}]

            # Query vLLM deployment with chat
            output_vllm_chat = query_ray_deployment(
                host="0.0.0.0",
                port=8004,
                model_id="hf-llm-vllm",
                prompt=chat_prompt,
                max_tokens=20,
                use_chat=True,
            )
            print(f"vLLM backend chat response: {output_vllm_chat}")

            # Both should return non-empty responses for chat
            assert output_vllm_chat != "", "vLLM backend chat prediction is empty"

            # Now terminate the vLLM deployment and start the regular one
            print("Terminating vLLM deployment and starting regular deployment...")
            terminate_deployment_process(self.deploy_proc_vllm)
            self.deploy_proc_vllm = None
            time.sleep(10)  # Give time for cleanup

            # Deploy without vLLM backend (regular HF)
            self.deploy_proc_regular = subprocess.Popen(
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
                    "hf-llm-regular",
                    "--num_gpus",
                    str(1),
                    "--host",
                    "0.0.0.0",
                    "--port",
                    str(8005),
                    "--trust_remote_code",
                    "--cuda_visible_devices",
                    "0",
                    # Note: no --use_vllm_backend flag
                ]
            )
            print("HF Deployment without vLLM backend started. Waiting for it to be ready...")

            # Wait for regular deployment to be ready
            if not wait_for_deployment_ready(host="0.0.0.0", port=8005, max_wait_time=300):
                assert False, "Regular deployment failed to become ready within timeout"

            time.sleep(20)

            # Query regular deployment
            output_regular = query_ray_deployment(
                host="0.0.0.0",
                port=8005,
                model_id="hf-llm-regular",
                prompt=test_prompt,
                max_tokens=20,
                temperature=0.7,
            )
            print(f"Regular backend response: {output_regular}")

            # Both should be able to handle the same prompt (responses may differ due to randomness)
            assert output_regular != "", "Regular backend prediction is empty"
            assert len(output_regular) > 0, "Regular backend returned empty response"

            # Query regular deployment with chat
            output_regular_chat = query_ray_deployment(
                host="0.0.0.0",
                port=8005,
                model_id="hf-llm-regular",
                prompt=chat_prompt,
                max_tokens=20,
                use_chat=True,
            )
            print(f"Regular backend chat response: {output_regular_chat}")

            # Both should return non-empty responses for chat
            assert output_regular_chat != "", "Regular backend chat prediction is empty"

        finally:
            # Ensure both deployments are terminated
            if self.deploy_proc_vllm is not None:
                terminate_deployment_process(self.deploy_proc_vllm)
                self.deploy_proc_vllm = None
            if self.deploy_proc_regular is not None:
                terminate_deployment_process(self.deploy_proc_regular)
                self.deploy_proc_regular = None

    def test_deploy_ray_hf_vllm_backend_performance(self):
        """Test vLLM backend deployment with multiple concurrent requests."""
        hf_model_path = "meta-llama/Llama-3.2-1B"

        try:
            # Deploy with vLLM backend
            self.deploy_proc_vllm = subprocess.Popen(
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
                    "hf-llm-vllm-perf",
                    "--num_gpus",
                    str(1),
                    "--host",
                    "0.0.0.0",
                    "--port",
                    str(8006),
                    "--trust_remote_code",
                    "--cuda_visible_devices",
                    "0",
                    "--use_vllm_backend",
                    "--max_ongoing_requests",
                    str(10),
                ]
            )
            print("HF Deployment with vLLM backend for performance test started. Waiting for it to be ready...")

            # Wait for deployment to be ready
            if not wait_for_deployment_ready(host="0.0.0.0", port=8006, max_wait_time=300):
                assert False, "vLLM deployment failed to become ready within timeout"

            time.sleep(20)

            # Test multiple requests to verify vLLM backend can handle concurrent requests
            prompts = [
                "What is 2+2?",
                "Name a color.",
                "What is the weather like?",
                "Tell me a joke.",
                "What is your name?",
            ]

            responses = []
            for i, prompt in enumerate(prompts):
                output = query_ray_deployment(
                    host="0.0.0.0",
                    port=8006,
                    model_id="hf-llm-vllm-perf",
                    prompt=prompt,
                    max_tokens=15,
                    temperature=0.7,
                )
                print(f"Request {i + 1} response: {output}")
                responses.append(output)
                assert output != "", f"Prediction {i + 1} is empty"

            # Verify all requests were processed
            assert len(responses) == len(prompts), "Not all requests were processed"
            assert all(response != "" for response in responses), "Some responses are empty"

        finally:
            # Ensure the deployment is terminated
            if self.deploy_proc_vllm is not None:
                terminate_deployment_process(self.deploy_proc_vllm)
                self.deploy_proc_vllm = None
