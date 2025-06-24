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
import shutil
import signal
import subprocess
import tempfile

from scripts.deploy.nlp.query_inframework import query_llm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestDeployPyTriton:
    @classmethod
    def setup_class(cls):
        # Create output directories
        cls.testdir = tempfile.mkdtemp()
        logger.info(f"Test directory: {cls.testdir}")

        # HF to NeMo2
        subprocess.run(
            [
                "coverage",
                "run",
                "--data-file=/workspace/.coverage",
                "--source=/workspace/",
                "--parallel-mode",
                "tests/functional_tests/utils/test_hf_import.py",
                "--hf_model",
                "meta-llama/Llama-3.2-1B",
                "--output_path",
                f"{cls.testdir}/nemo2_ckpt",
                "--config",
                "Llama32Config1B",
            ],
            check=True,
        )

    @classmethod
    def teardown_class(cls):
        logger.info(f"Removing test directory: {cls.testdir}")
        shutil.rmtree(cls.testdir)

    def test_deploy_pytriton(self):
        # Run deployment
        deploy_proc = subprocess.Popen(
            [
                "torchrun",
                "--nproc_per_node=2",
                "--no-python",
                "coverage",
                "run",
                "--data-file=/workspace/.coverage",
                "--source=/workspace/",
                "--parallel-mode",
                "scripts/deploy/nlp/deploy_inframework_triton.py",
                "--nemo_checkpoint",
                f"{self.testdir}/nemo2_ckpt",
                "--triton_model_name",
                "llama",
                "--tensor_parallelism_size",
                str(2),
            ]
        )

        outputs = query_llm(
            url="0.0.0.0",
            model_name="llama",
            prompts=["What is the color of a banana?"],
            max_output_len=20,
        )

        print(outputs)

        # Check if deployment was successful
        assert len(outputs) != 0, "Prediction empty"

        deploy_proc.send_signal(signal.SIGINT)
