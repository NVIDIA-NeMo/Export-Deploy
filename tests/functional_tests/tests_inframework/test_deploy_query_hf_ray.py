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


class TestDeployRayHF:
    def setup_method(self):
        """Setup for each test method."""
        self.deploy_proc = None

    def teardown_method(self):
        """Cleanup after each test method."""
        if self.deploy_proc is not None:
            terminate_deployment_process(self.deploy_proc)
            # Avoid double termination in case test used finally to clean up
            self.deploy_proc = None

    def test_deploy_ray_hf(self):
        hf_model_path = "meta-llama/Llama-3.2-1B"

        try:
            # Run Ray deployment for HF model
            self.deploy_proc = subprocess.Popen(
                [
                    "coverage",
                    "run",
                    "--data-file=/workspace/.coverage",
                    "--source=/workspace/",
                    "--parallel-mode",
                    "scripts/deploy/nlp/deploy_ray_hf.py",
                    "--model_path",
                    hf_model_path,
                    "--task",
                    "text-generation",
                    "--model_id",
                    "hf-llm",
                    "--num_gpus",
                    str(1),
                    "--host",
                    "0.0.0.0",
                    "--port",
                    str(8001),
                    "--trust_remote_code",
                    "--cuda_visible_devices",
                    "0",
                ]
            )
            print("HF Deployment started. Waiting for it to be ready...")

            # Wait for deployment to be ready
            if not wait_for_deployment_ready(host="0.0.0.0", port=8001, max_wait_time=300):
                assert False, "Deployment failed to become ready within timeout"

            time.sleep(20)

            output = query_ray_deployment(
                host="0.0.0.0",
                port=8001,
                model_id="hf-llm",
                prompt="What is the color of a banana?",
                max_tokens=20,
            )

            print(output)

            # Check if deployment was successful
            assert output != "", "First prediction is empty"

            # Send a second request using the chat endpoint
            output_chat = query_ray_deployment(
                host="0.0.0.0",
                port=8001,
                model_id="hf-llm",
                prompt=[{"role": "user", "content": "Hello, how are you?"}],
                max_tokens=20,
                use_chat=True,
            )
            print(output_chat)
            # Check if deployment was successful
            assert output_chat != "", "Second prediction (chat) is empty"
        finally:
            # Ensure the deployment is terminated as soon as queries complete or on failure
            if self.deploy_proc is not None:
                terminate_deployment_process(self.deploy_proc)
                self.deploy_proc = None
