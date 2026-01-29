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
import os
import shutil
import subprocess
import tempfile

import pytest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@pytest.fixture
def tmp_dir():
    tmp_dir = tempfile.mkdtemp()
    yield tmp_dir
    try:
        shutil.rmtree(tmp_dir)
    except FileNotFoundError as e:
        logger.warning(f"Error removing temporary directory {tmp_dir}: {e}")


@pytest.mark.skip(reason="Temporarily disabled")
class TestONNXTRTExport:
    def test_export_onnx_trt_embedding(self):
        subprocess.run(
            [
                "coverage",
                "run",
                "--data-file=/workspace/.coverage",
                "--source=/workspace/",
                "--parallel-mode",
                "tests/functional_tests/utils/run_onnx_trt_embedding_export.py",
                "--hf_model_path",
                "/home/TestData/llm/models/llama-3.2-nv-embedqa-1b-v2",
                "--normalize",
            ],
            check=True,
            env={
                **os.environ.copy(),
                "HF_DATASETS_CACHE": "/tmp/hf_datasets_cache",
            },
        )

    def test_export_onnx_trt_embedding_int8(self):
        subprocess.run(
            [
                "coverage",
                "run",
                "--data-file=/workspace/.coverage",
                "--source=/workspace/",
                "--parallel-mode",
                "tests/functional_tests/utils/run_onnx_trt_embedding_export.py",
                "--hf_model_path",
                "/home/TestData/llm/models/llama-3.2-nv-embedqa-1b-v2",
                "--quant_cfg",
                "int8_sq",
                "--calibration_dataset",
                "tests/functional_tests/data/calibration_dataset.json",
                "--calibration_batch_size",
                "2",
                "--calibration_dataset_size",
                "6",
            ],
            check=True,
            env={
                **os.environ.copy(),
                "HF_DATASETS_CACHE": "/tmp/hf_datasets_cache",
            },
        )

    def test_export_onnx_trt_reranking(self):
        subprocess.run(
            [
                "coverage",
                "run",
                "--data-file=/workspace/.coverage",
                "--source=/workspace/",
                "--parallel-mode",
                "tests/functional_tests/utils/run_onnx_trt_reranking_export.py",
                "--hf_model_path",
                "/home/TestData/llm/models/llama-3.2-nv-reranker-1b",
            ],
            check=True,
            env={
                **os.environ.copy(),
                "HF_DATASETS_CACHE": "/tmp/hf_datasets_cache",
            },
        )
