import logging
import shutil
import subprocess
import tempfile

import pytest

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestVLLMExportLlama:
    @classmethod
    def setup_class(cls):
        # Create output directories
        cls.testdir = tempfile.mkdtemp()
        logger.info(f"Test directory: {cls.testdir}")

    @classmethod
    def teardown_class(cls):
        logger.info(f"Removing test directory: {cls.testdir}")
        shutil.rmtree(cls.testdir)

    def test_finetune_llava_next_InternVIT(self):
        subprocess.run(
            [
                "python",
                "tests/functional_tests/utils/test_llava_next_InternVIT.py",
                "--devices",
                "1",
                "--max-steps",
                "5",
                "--experiment-dir",
                self.testdir,
            ],
            check=True,
        )

    @pytest.mark.parametrize(
        "model",
        [
            "OpenGVLab/InternViT-300M-448px-V2_5",
            "openai/clip-vit-large-patch14",
            "google/siglip-base-patch16-224",
        ],
    )
    def test_import_hf_models(self, model):
        """Test importing different HuggingFace models using the import_hf.py script."""
        logger.info(f"Testing import of model: {model}")

        result = subprocess.run(
            ["python", "scripts/vlm/import_hf.py", "--input_name_or_path", model],
            capture_output=True,
            text=True,
            cwd="/workspace",
            check=True,
        )

        # Check if the command executed successfully
        assert result.returncode == 0, f"Failed to import {model}. stdout: {result.stdout}, stderr: {result.stderr}"
        logger.info(f"Successfully imported {model}")
