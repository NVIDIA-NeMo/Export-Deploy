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
import signal
import subprocess
import time

from scripts.deploy.llm.nemo2.query_triton import query_llm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestDeployNemo2Triton:
    def setup_method(self):
        """Setup for each test method."""
        self.deploy_proc = None

    def teardown_method(self):
        """Cleanup after each test method."""
        if self.deploy_proc is not None:
            logger.info("Terminating deployment process...")
            try:
                self.deploy_proc.send_signal(signal.SIGINT)
                try:
                    self.deploy_proc.wait(timeout=10)
                    logger.info("Deployment terminated gracefully")
                except subprocess.TimeoutExpired:
                    logger.warning("Graceful shutdown timed out, forcing termination...")
                    self.deploy_proc.kill()
                    self.deploy_proc.wait()
                    logger.info("Deployment force terminated")
            except Exception as e:
                logger.error(f"Error terminating deployment: {e}")
                try:
                    self.deploy_proc.kill()
                except Exception:
                    pass
            self.deploy_proc = None

    def test_deploy_nemo2_triton(self):
        nemo_checkpoint_path = "/home/TestData/llm/models/llama32_1b_nemo2"

        try:
            # Run Triton deployment with torchrun for distributed setup
            self.deploy_proc = subprocess.Popen(
                [
                    "torchrun",
                    "--nproc_per_node=1",
                    "--no-python",
                    "coverage",
                    "run",
                    "--data-file=/workspace/.coverage",
                    "--source=/workspace/",
                    "--parallel-mode",
                    "scripts/deploy/llm/nemo2/deploy_triton.py",
                    "--nemo-checkpoint",
                    nemo_checkpoint_path,
                    "--triton-model-name",
                    "llama",
                    "--tensor-parallelism-size",
                    str(1),
                    "--num-gpus",
                    str(1),
                    "--triton-port",
                    str(8000),
                    "--server-port",
                    str(8080),
                    "--max-batch-size",
                    str(8),
                    "--enable-flash-decode",
                    "--enable-cuda-graphs",
                    "--inference-max-seq-length",
                    str(4096),
                    "--micro-batch-size",
                    str(10),
                    "debug-mode",
                ]
            )
            logger.info("Deployment started. Waiting for it to be ready...")

            # Wait for deployment to be ready - give it time to initialize
            # PyTriton typically takes longer to start than Ray
            time.sleep(120)

            # Query the deployment - first request
            outputs = query_llm(
                url="0.0.0.0",
                model_name="llama",
                prompts=["What is the color of a banana?"],
                max_output_len=20,
                top_k=1,
                top_p=0.0,
                temperature=1.0,
                init_timeout=60.0,
            )

            print(outputs)

            # Check if deployment was successful
            assert len(outputs) != 0, "First prediction is empty"

            # Send a second request to ensure service is stable
            outputs_2 = query_llm(
                url="0.0.0.0",
                model_name="llama",
                prompts=["Hello, how are you?"],
                max_output_len=20,
                top_k=1,
                top_p=0.0,
                temperature=1.0,
                init_timeout=60.0,
            )

            print(outputs_2)

            # Check if deployment was successful
            assert len(outputs_2) != 0, "Second prediction is empty"

        finally:
            # Ensure the deployment is terminated as soon as queries complete or on failure
            if self.deploy_proc is not None:
                logger.info("Terminating deployment process in finally block...")
                try:
                    self.deploy_proc.send_signal(signal.SIGINT)
                    self.deploy_proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    logger.warning("Forcing termination...")
                    self.deploy_proc.kill()
                    self.deploy_proc.wait()
                except Exception as e:
                    logger.error(f"Error during cleanup: {e}")
                self.deploy_proc = None
