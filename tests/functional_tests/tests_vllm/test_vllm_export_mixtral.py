import json
import logging
import shutil
import subprocess
import tempfile

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestVLLMExportMixtral:
    @classmethod
    def setup_class(cls):
        # Create output directories
        cls.testdir = tempfile.mkdtemp()
        logger.info(f"Test directory: {cls.testdir}")

        # Update HF model
        subprocess.run(
            [
                "coverage",
                "run",
                "--data-file=/workspace/.coverage",
                "--source=/workspace/",
                "--parallel-mode",
                "tests/functional_tests/utils/create_hf_model.py",
                "--model_name_or_path",
                "/home/TestData/hf/Mixtral-8x7B-Instruct-v0.1",
                "--output_dir",
                f"{cls.testdir}/mixtral_tiny_hf",
                "--config_updates",
                json.dumps(
                    {
                        "num_hidden_layers": 2,
                        "hidden_size": 128,
                        "intermediate_size": 448,
                        "num_attention_heads": 4,
                        "num_key_value_heads": 2,
                        "head_dim": 32,
                        "num_local_experts": 4,
                    }
                ),
            ],
            check=True,
        )

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
                f"{cls.testdir}/mixtral_tiny_hf",
                "--model",
                "MixtralModel",
                "--config",
                "MixtralConfig8x7B",
                "--output_path",
                f"{cls.testdir}/mixtral_tiny_nemo2",
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
                "coverage",
                "run",
                "--data-file=/workspace/.coverage",
                "--source=/workspace/",
                "--parallel-mode",
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
                f"{self.testdir}/mixtral_tiny_nemo2",
            ],
            check=True,
        )
