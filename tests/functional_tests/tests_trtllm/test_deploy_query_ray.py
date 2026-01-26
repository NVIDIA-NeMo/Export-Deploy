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


class TestDeployRayTRTLLM:
    def setup_method(self):
        """Setup for each test method."""
        self.deploy_proc = None

    def teardown_method(self):
        """Cleanup after each test method."""
        if self.deploy_proc is not None:
            terminate_deployment_process(self.deploy_proc)
            # Avoid double termination in case test used finally to clean up
            self.deploy_proc = None

    def test_deploy_ray_trtllm(self):
        nemo_checkpoint_path = "/home/TestData/llm/models/llama32_1b_nemo2"
        host = "0.0.0.0"
        port = 8002
        model_id = "trt-llm"

        try:
            # Run Ray deployment using the TRT-LLM deploy script
            self.deploy_proc = subprocess.Popen(
                [
                    "coverage",
                    "run",
                    "--data-file=/workspace/.coverage",
                    "--source=/workspace/",
                    "--parallel-mode",
                    "scripts/deploy/nlp/deploy_ray_trtllm.py",
                    "--nemo_checkpoint_path",
                    nemo_checkpoint_path,
                    "--model_type",
                    "llama",
                    "--tensor_parallelism_size",
                    str(1),
                    "--max_input_len",
                    str(2048),
                    "--max_output_len",
                    str(1024),
                    "--max_batch_size",
                    str(8),
                    "--num_gpus",
                    str(1),
                    "--num_replicas",
                    str(1),
                    "--host",
                    host,
                    "--port",
                    str(port),
                    "--model_id",
                    model_id,
                    "--cuda_visible_devices",
                    "0,1",
                ]
            )
            print("TRT-LLM Deployment started. Waiting for it to be ready...")

            # Wait for deployment to be ready
            if not wait_for_deployment_ready(host=host, port=port, max_wait_time=300):
                assert False, "Deployment failed to become ready within timeout"

            time.sleep(20)

            output = query_ray_deployment(
                host=host,
                port=port,
                model_id=model_id,
                prompt="What is the color of a banana?",
                max_tokens=20,
            )

            print(output)

            # Check if deployment was successful
            assert output != "", "First prediction is empty"

            # Send a second request using the chat endpoint
            output_chat = query_ray_deployment(
                host=host,
                port=port,
                model_id=model_id,
                prompt="Hello, how are you?",
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
