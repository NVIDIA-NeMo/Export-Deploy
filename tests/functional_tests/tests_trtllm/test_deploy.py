import logging
import subprocess

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestTRTLLMDeploy:
    def test_trtllm_deploy_nemo2(self):
        subprocess.run(
            [
                "coverage",
                "run",
                "--data-file=/workspace/.coverage",
                "--source=/workspace/",
                "--parallel-mode",
                "tests/functional_tests/utils/run_nemo_deploy.py",
                "--model_name",
                "test_model",
                "--checkpoint_dir",
                "/home/TestData/llm/models/llama32_1b_nemo2",
                "--backend",
                "TensorRT-LLM",
                "--min_gpus",
                "1",
                "--max_gpus",
                "2",
                "--run_accuracy",
                "True",
                "--test_data_path",
                "tests/functional_tests/data/lambada.json",
                "--test_deployment",
                "True",
                "--debug",
            ],
            check=True,
        )
