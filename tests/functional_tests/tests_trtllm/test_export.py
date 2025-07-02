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
import subprocess
import tempfile
from pathlib import Path

import pytest

from nemo_export.tensorrt_llm import TensorRTLLM

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


class TestTRTLLMExport:
    @pytest.mark.pleasefixme
    @pytest.mark.parametrize("tensor_parallelism_size", [2, 1])
    def test_nemo2_convert_to_export(self, tensor_parallelism_size):
        """
        Test safe tensor exporter. This tests the whole nemo export until engine building.
        """

        trt_llm_exporter = TensorRTLLM(model_dir="/tmp/safe_tensor_test_2/")
        trt_llm_exporter.export(
            nemo_checkpoint_path="/home/TestData/llm/models/llama32_1b_nemo2",
            model_type="llama",
            delete_existing_files=True,
            tensor_parallelism_size=tensor_parallelism_size,
            pipeline_parallelism_size=1,
            max_input_len=1024,
            max_output_len=256,
            max_batch_size=3,
            use_parallel_embedding=False,
            paged_kv_cache=True,
            remove_input_padding=True,
            use_paged_context_fmha=False,
            dtype=None,
            load_model=True,
            use_lora_plugin=None,
            lora_target_modules=None,
            max_lora_rank=64,
            max_num_tokens=None,
            opt_num_tokens=None,
            max_seq_len=512,
            multiple_profiles=False,
            gpt_attention_plugin="auto",
            gemm_plugin="auto",
            reduce_fusion=True,
            fp8_quantized=None,
            fp8_kvcache=None,
            build_rank=None,
        )

        output = trt_llm_exporter.forward(
            input_texts=["Tell me the capitol of France "],
            max_output_len=16,
            top_k=1,
            top_p=0.0,
            temperature=0.1,
            stop_words_list=None,
            bad_words_list=None,
            no_repeat_ngram_size=None,
            lora_uids=None,
            output_log_probs=False,
            output_context_logits=False,
            output_generation_logits=False,
        )

        print(output)

        assert Path("/tmp/safe_tensor_test_2/trtllm_engine/").exists(), "Safe tensors were not generated."
        assert Path("/tmp/safe_tensor_test_2/trtllm_engine/rank0.engine").exists(), (
            "Safe tensors for rank0 were not generated."
        )
        assert Path("/tmp/safe_tensor_test_2/trtllm_engine/config.json").exists(), "config.yaml was not generated."

        shutil.rmtree("/tmp/safe_tensor_test_2/")

    def test_export_hf(self, tmp_dir):
        subprocess.run(
            [
                "coverage",
                "run",
                "--data-file=/workspace/.coverage",
                "--source=/workspace/",
                "--parallel-mode",
                "tests/functional_tests/utils/run_nemo_export.py",
                "--model_name",
                "test",
                "--model_dir",
                tmp_dir,
                "--use_huggingface",
                "True",
                "--checkpoint_dir",
                "/home/TestData/llm/models/llama3.2-1B-hf/",
                "--min_tps",
                "1",
                "--test_deployment",
                "True",
                "--debug",
            ],
            check=True,
        )

    def test_export_nemo2(self, tmp_dir):
        subprocess.run(
            [
                "coverage",
                "run",
                "--data-file=/workspace/.coverage",
                "--source=/workspace/",
                "--parallel-mode",
                "tests/functional_tests/utils/run_nemo_export.py",
                "--model_name",
                "test",
                "--model_dir",
                tmp_dir,
                "--model_type",
                "llama",
                "--checkpoint_dir",
                "/home/TestData/llm/models/llama32_1b_nemo2",
                "--min_tps",
                "1",
                "--test_deployment",
                "True",
                "--debug",
            ]
        )

    def test_export_qnemo(self, tmp_dir):
        subprocess.run(
            [
                "coverage",
                "run",
                "--data-file=/workspace/.coverage",
                "--source=/workspace/",
                "--parallel-mode",
                "tests/functional_tests/utils/create_hf_model.py",
                "--model_name_or_path",
                "/home/TestData/hf/Llama-2-7b-hf",
                "--output_dir",
                f"{tmp_dir}/llama_tiny_hf",
                "--config_updates",
                '{"num_hidden_layers": 2, "hidden_size": 512, "intermediate_size": 384, "num_attention_heads": 8, "num_key_value_heads": 8}',
            ],
            check=True,
        )

        subprocess.run(
            [
                "coverage",
                "run",
                "--data-file=/workspace/.coverage",
                "--source=/workspace/",
                "--parallel-mode",
                "tests/functional_tests/utils/test_hf_import.py",
                "--hf_model",
                f"{tmp_dir}/llama_tiny_hf",
                "--output_path",
                f"{tmp_dir}/nemo2_ckpt",
            ],
            check=True,
        )

        subprocess.run(
            [
                "coverage",
                "run",
                "--data-file=/workspace/.coverage",
                "--source=/workspace/",
                "--parallel-mode",
                "tests/functional_tests/utils/create_ptq_ckpt.py",
                "--nemo_checkpoint",
                f"{tmp_dir}/nemo2_ckpt",
                "--algorithm",
                "int8_sq",
                "--calibration_dataset",
                "tests/functional_tests/data/calibration_dataset.json",
                "--calibration_batch_size",
                "2",
                "--calibration_dataset_size",
                "6",
                "--export_format",
                "trtllm",
                "--export_path",
                f"{tmp_dir}/nemo2_ptq",
                "--generate_sample",
            ],
            check=True,
        )

        subprocess.run(
            [
                "coverage",
                "run",
                "--data-file=/workspace/.coverage",
                "--source=/workspace/",
                "--parallel-mode",
                "tests/functional_tests/utils/run_nemo_export.py",
                "--model_name",
                "test",
                "--model_dir",
                f"{tmp_dir}/trt_llm_model_dir/",
                "--checkpoint_dir",
                f"{tmp_dir}/nemo2_ptq",
                "--min_tps",
                "1",
                "--test_deployment",
                "True",
                "--debug",
            ],
            check=True,
        )

    def test_export_onnx(self):
        subprocess.run(
            [
                "coverage",
                "run",
                "--data-file=/workspace/.coverage",
                "--source=/workspace/",
                "--parallel-mode",
                "tests/functional_tests/utils/test_export_onnx.py",
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
        )

    def test_export_onnx_int8(self):
        subprocess.run(
            [
                "coverage",
                "run",
                "--data-file=/workspace/.coverage",
                "--source=/workspace/",
                "--parallel-mode",
                "tests/functional_tests/utils/test_export_onnx.py",
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
        )
