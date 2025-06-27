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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestDeployRay:
    def test_deploy_ray(self):
        # Run deployment
        subprocess.run(
            [
                "coverage",
                "run",
                "-a",
                "--data-file=/workspace/.coverage",
                "--source=/workspace",
                "tests/functional_tests/utils/run_trtllm_deploy_ray.py",
                "--model_name",
                "test_model",
                "--nemo_checkpoint_path",
                "/home/TestData/llm/models/llama32_1b_nemo2",
                "--model_type",
                "llama",
                "--tensor_parallelism_size",
                "1",
                "--max_input_len",
                "2048",
                "--max_output_len",
                "1024",
                "--max_batch_size",
                "8",
                "--num_gpus",
                "1",
                "--num_replicas",
                "1",
                "--debug",
            ],
            check=True,
        )
