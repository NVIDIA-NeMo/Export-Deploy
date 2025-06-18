import subprocess
import json
import tempfile
import shutil

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestVLLMExportLlama:
    @classmethod
    def setup_class(cls):
        # Create output directories
        cls.testdir = tempfile.mkdtemp()
        logger.info(f"Test directory: {cls.testdir}")

        # Update HF model
        subprocess.run(
            [
                "python",
                "tests/functional_tests/utils/create_hf_model.py",
                "--model_name_or_path",
                "/home/TestData/nlp/megatron_llama/llama-ci-hf",
                "--output_dir",
                f"{cls.testdir}/llama_head64",
                "--config_updates",
                json.dumps(
                    {
                        "hidden_size": 512,
                        "num_attention_heads": 4,
                        "num_key_value_heads": 4,
                        "intermediate_size": 1024,
                        "head_dim": 128,
                        "num_hidden_layers": 2,
                        "torch_dtype": "float16",
                    }
                ),
            ],
            check=True,
        )

        # HF to NeMo2
        subprocess.run(
            [
                "python",
                "tests/functional_tests/utils/test_hf_import.py",
                "--hf_model",
                f"{cls.testdir}/llama_head64",
                "--output_path",
                f"{cls.testdir}/nemo2_ckpt",
            ],
            check=True,
        )

    @classmethod
    def teardown_class(cls):
        logger.info(f"Removing test directory: {cls.testdir}")
        shutil.rmtree(cls.testdir)

    def test_vllm_export_llama(self):
        subprocess.run(
            [
                "python",
                "tests/functional_tests/utils/run_nemo_export.py",
                "--min_tps",
                "1",
                "--max_tps",
                "1",
                "--use_vllm",
                "True",
                "--max_output_len",
                "128",
                "--test_deployment",
                "True",
                "--model_name",
                "nemo2_ckpt",
                "--model_dir",
                f"{self.testdir}/vllm_from_nemo2",
                "--checkpoint_dir",
                f"{self.testdir}/nemo2_ckpt",
            ],
            check=True,
        )
