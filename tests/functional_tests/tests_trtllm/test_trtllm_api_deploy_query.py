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


class TestTRTLLMAPIDeployQuerySubprocess:
    """Test class for TensorRT-LLM API deploy and query functionality using subprocess."""

    def test_trtllm_api_deploy_query(self):
        """Test TensorRT-LLM API deploy and query with Llama-3.2-1B model using subprocess."""
        subprocess.run(
            [
                "coverage",
                "run",
                "-a",
                "--data-file=/workspace/.coverage",
                "--source=/workspace",
                "tests/functional_tests/utils/run_trtllm_api_deploy_query.py",
                "--hf_model_path",
                "/home/TestData/hf/Llama-3.2-1B",
                "--tensor_parallel_size",
                "2",
            ],
            check=True,
        )

        logger.info("TRTLLM API deploy and query test completed successfully")
