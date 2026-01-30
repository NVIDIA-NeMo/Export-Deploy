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


import argparse
import unittest
from unittest.mock import MagicMock, patch

from nemo_deploy.deploy_ray import DeployRay

# Import the functions from the deploy script
from scripts.deploy.nlp.deploy_ray_inframework import (
    json_type,
)


class TestDeployRay(unittest.TestCase):
    def setUp(self):
        # Ensure tests run even when Ray is not installed
        self._have_ray_patcher = patch("nemo_deploy.deploy_ray.HAVE_RAY", True)
        self._have_ray_patcher.start()

    def tearDown(self):
        self._have_ray_patcher.stop()

    @patch("nemo_deploy.deploy_ray.ray")
    def test_init_with_existing_cluster(self, mock_ray):
        # Test initialization connecting to existing cluster
        DeployRay(address="auto", num_cpus=2, num_gpus=1)
        mock_ray.init.assert_called_once_with(address="auto", ignore_reinit_error=True, runtime_env=None)

    @patch("nemo_deploy.deploy_ray.ray")
    def test_init_with_runtime_env(self, mock_ray):
        # Test initialization with custom runtime environment
        runtime_env = {"pip": ["numpy", "pandas"]}
        DeployRay(runtime_env=runtime_env)
        mock_ray.init.assert_called_once_with(address="auto", ignore_reinit_error=True, runtime_env=runtime_env)

    @patch("nemo_deploy.deploy_ray.ray")
    def test_init_with_new_cluster(self, mock_ray):
        # Test initialization creating a new cluster when connection fails
        mock_ray.init.side_effect = [ConnectionError, None]

        DeployRay(num_cpus=4, num_gpus=2, include_dashboard=True)

        assert mock_ray.init.call_count == 2
        mock_ray.init.assert_called_with(
            num_cpus=4,
            num_gpus=2,
            include_dashboard=True,
            ignore_reinit_error=True,
            runtime_env=None,
        )

    @patch("nemo_deploy.deploy_ray.get_available_cpus", return_value=12)
    @patch("nemo_deploy.deploy_ray.ray")
    def test_init_uses_available_cpus_on_local_start(self, mock_ray, mock_get_cpus):
        # When num_cpus is None and connection fails, it should use get_available_cpus()
        mock_ray.init.side_effect = [ConnectionError, None]
        DeployRay(num_cpus=None, num_gpus=1, include_dashboard=False)
        # Second call is for local init with computed CPUs
        mock_ray.init.assert_called_with(
            num_cpus=12,
            num_gpus=1,
            include_dashboard=False,
            ignore_reinit_error=True,
            runtime_env=None,
        )

    @patch("nemo_deploy.deploy_ray.ray")
    @patch("nemo_deploy.deploy_ray.serve")
    def test_start_with_port(self, mock_serve, mock_ray):
        # Test starting Ray Serve with specified port via _start()
        deploy = DeployRay(host="localhost", port=8080)
        deploy._start()

        mock_serve.start.assert_called_once_with(
            http_options={
                "host": "localhost",
                "port": 8080,
            }
        )

    @patch("nemo_deploy.deploy_ray.ray")
    @patch("nemo_deploy.deploy_ray.serve")
    @patch("nemo_deploy.deploy_ray.find_available_port")
    def test_start_without_port(self, mock_find_port, mock_serve, mock_ray):
        # Test starting Ray Serve with auto-detected port via _start()
        mock_find_port.return_value = 9090

        deploy = DeployRay(host="0.0.0.0")
        deploy._start()

        mock_find_port.assert_called_once_with(8000, "0.0.0.0")
        mock_serve.start.assert_called_once_with(
            http_options={
                "host": "0.0.0.0",
                "port": 9090,
            }
        )

    @patch("nemo_deploy.deploy_ray.ray")
    @patch("nemo_deploy.deploy_ray.serve")
    @patch("nemo_deploy.deploy_ray.HFRayDeployable")
    @patch("nemo_deploy.deploy_ray.signal.signal")
    @patch.object(DeployRay, "_start")
    def test_deploy_hf_runs(self, mock_start, mock_signal, mock_hf_deployable, mock_serve, mock_ray):
        # Test running a HuggingFace model triggers serve.run
        deploy = DeployRay()

        mock_app = MagicMock()
        mock_options = MagicMock()
        mock_options.bind.return_value = mock_app
        mock_hf_deployable.options.return_value = mock_options

        deploy.deploy_huggingface_model(
            hf_model_id_path="test-hf-model",
            model_id="test_model",
            test_mode=True,
        )

        mock_start.assert_called_once()
        mock_serve.run.assert_called_once_with(mock_app, name="test_model")
        mock_hf_deployable.options.assert_called_once()

    @patch("nemo_deploy.deploy_ray.ray")
    @patch("nemo_deploy.deploy_ray.serve")
    @patch("nemo_deploy.deploy_ray.HFRayDeployable")
    @patch("nemo_deploy.deploy_ray.signal.signal")
    @patch.object(DeployRay, "_start")
    def test_deploy_hf_with_vllm_backend(self, mock_start, mock_signal, mock_hf_deployable, mock_serve, mock_ray):
        # Test running a HuggingFace model with vLLM backend
        deploy = DeployRay()

        mock_app = MagicMock()
        mock_options = MagicMock()
        mock_options.bind.return_value = mock_app
        mock_hf_deployable.options.return_value = mock_options

        deploy.deploy_huggingface_model(
            hf_model_id_path="test-hf-model",
            model_id="test_model",
            use_vllm_backend=True,
            test_mode=True,
        )

        mock_start.assert_called_once()
        mock_serve.run.assert_called_once_with(mock_app, name="test_model")
        mock_hf_deployable.options.assert_called_once()

        # Verify the bind call includes use_vllm_backend=True
        _, bind_kwargs = mock_options.bind.call_args
        assert bind_kwargs["use_vllm_backend"] is True

    @patch("nemo_deploy.deploy_ray.ray")
    @patch("nemo_deploy.deploy_ray.serve")
    @patch("nemo_deploy.deploy_ray.MegatronRayDeployable")
    @patch("nemo_deploy.deploy_ray.signal.signal")
    @patch.object(DeployRay, "_start")
    def test_deploy_inframework_runs(self, mock_start, mock_signal, mock_megatron, mock_serve, mock_ray):
        # Parallelism per replica (2) equals GPUs per replica (2): valid
        deploy = DeployRay()

        mock_app = MagicMock()
        mock_options = MagicMock()
        mock_options.bind.return_value = mock_app
        mock_megatron.options.return_value = mock_options

        deploy.deploy_inframework_model(
            megatron_checkpoint="/path/to/model.megatron",
            num_gpus=4,
            tensor_model_parallel_size=2,
            pipeline_model_parallel_size=1,
            context_parallel_size=1,
            num_replicas=2,
            num_cpus_per_replica=4,
            model_id="megatron-model",
            test_mode=True,
        )

        mock_start.assert_called_once()
        mock_serve.run.assert_called_once_with(mock_app, name="megatron-model")
        mock_megatron.options.assert_called_once()
        # Ensure actor options include provided CPUs
        _, kwargs = mock_megatron.options.call_args
        assert kwargs["ray_actor_options"]["num_cpus"] == 4

    @patch("nemo_deploy.deploy_ray.ray")
    @patch("nemo_deploy.deploy_ray.serve")
    @patch("nemo_deploy.deploy_ray.MegatronMultimodalRayDeployable")
    @patch("nemo_deploy.deploy_ray.signal.signal")
    @patch.object(DeployRay, "_start")
    def test_deploy_vlm_inframework_model_runs(self, mock_start, mock_signal, mock_multimodal, mock_serve, mock_ray):
        # Test deploying VLM (multimodal) inframework model
        deploy = DeployRay()

        mock_app = MagicMock()
        mock_options = MagicMock()
        mock_options.bind.return_value = mock_app
        mock_multimodal.options.return_value = mock_options

        deploy.deploy_vlm_inframework_model(
            megatron_checkpoint="/path/to/multimodal-model.megatron",
            num_gpus=2,
            tensor_model_parallel_size=2,
            pipeline_model_parallel_size=1,
            model_id="multimodal-model",
            num_cpus_per_replica=8,
            num_replicas=1,
            test_mode=True,
        )

        mock_start.assert_called_once()
        mock_serve.run.assert_called_once_with(mock_app, name="multimodal-model")
        mock_multimodal.options.assert_called_once()
        # Ensure actor options include provided CPUs
        _, kwargs = mock_multimodal.options.call_args
        assert kwargs["ray_actor_options"]["num_cpus"] == 8
        assert kwargs["num_replicas"] == 1

    @patch("nemo_deploy.deploy_ray.ray")
    @patch("nemo_deploy.deploy_ray.serve")
    @patch("nemo_deploy.deploy_ray.MegatronMultimodalRayDeployable")
    @patch("nemo_deploy.deploy_ray.signal.signal")
    @patch.object(DeployRay, "_start")
    def test_deploy_vlm_inframework_model_with_custom_params(
        self, mock_start, mock_signal, mock_multimodal, mock_serve, mock_ray
    ):
        # Test deploying VLM model with custom inference parameters
        deploy = DeployRay()

        mock_app = MagicMock()
        mock_options = MagicMock()
        mock_options.bind.return_value = mock_app
        mock_multimodal.options.return_value = mock_options

        deploy.deploy_vlm_inframework_model(
            megatron_checkpoint="/path/to/multimodal-model.megatron",
            num_gpus=4,
            tensor_model_parallel_size=2,
            pipeline_model_parallel_size=2,
            model_id="custom-vlm-model",
            params_dtype="bfloat16",
            inference_batch_times_seqlen_threshold=2000,
            inference_max_seq_length=4096,
            test_mode=True,
        )

        mock_start.assert_called_once()
        mock_serve.run.assert_called_once_with(mock_app, name="custom-vlm-model")

        # Verify custom parameters were passed in bind call
        _, bind_kwargs = mock_options.bind.call_args
        assert bind_kwargs["megatron_checkpoint_filepath"] == "/path/to/multimodal-model.megatron"
        assert bind_kwargs["tensor_model_parallel_size"] == 2
        assert bind_kwargs["pipeline_model_parallel_size"] == 2
        assert bind_kwargs["model_id"] == "custom-vlm-model"
        assert bind_kwargs["params_dtype"] == "bfloat16"
        assert bind_kwargs["inference_batch_times_seqlen_threshold"] == 2000
        assert bind_kwargs["inference_max_seq_length"] == 4096

    @patch("nemo_deploy.deploy_ray.ray")
    @patch("nemo_deploy.deploy_ray.serve")
    @patch("nemo_deploy.deploy_ray.TensorRTLLMRayDeployable")
    @patch("nemo_deploy.deploy_ray.signal.signal")
    @patch.object(DeployRay, "_start")
    def test_deploy_tensorrt_llm_python_runtime_runs(self, mock_start, mock_signal, mock_trt, mock_serve, mock_ray):
        deploy = DeployRay()

        mock_app = MagicMock()
        mock_options = MagicMock()
        mock_options.bind.return_value = mock_app
        mock_trt.options.return_value = mock_options

        deploy.deploy_tensorrt_llm_model(
            trt_llm_path="/path/to/trt-llm",
            model_id="trt-model",
            num_replicas=2,
            num_cpus_per_replica=3,
            num_gpus_per_replica=1,
            max_ongoing_requests=5,
            test_mode=True,
        )

        mock_start.assert_called_once()
        mock_serve.run.assert_called_once_with(mock_app, name="trt-model")
        mock_trt.options.assert_called_once()
        _, kwargs = mock_trt.options.call_args
        assert kwargs["ray_actor_options"]["num_cpus"] == 3
        assert kwargs["ray_actor_options"]["num_gpus"] == 1
        assert kwargs["num_replicas"] == 2
        assert kwargs["max_ongoing_requests"] == 5

    @patch("nemo_deploy.deploy_ray.ray")
    @patch("nemo_deploy.deploy_ray.serve")
    def test_deploy_tensorrt_llm_invalid_cpp_options_error(self, mock_serve, mock_ray):
        deploy = DeployRay.__new__(DeployRay)  # Avoid __init__
        deploy.host = "0.0.0.0"
        deploy.port = None

        with self.assertRaises(ValueError):
            DeployRay.deploy_tensorrt_llm_model(
                deploy,
                trt_llm_path="/path/to/trt-llm",
                enable_chunked_context=True,
                use_python_runtime=True,
                test_mode=True,
            )
        mock_serve.run.assert_not_called()

    @patch("nemo_deploy.deploy_ray.ray")
    @patch("nemo_deploy.deploy_ray.serve")
    @patch("nemo_deploy.deploy_ray.TensorRTLLMRayDeployable")
    @patch("nemo_deploy.deploy_ray.signal.signal")
    @patch.object(DeployRay, "_start")
    def test_deploy_tensorrt_llm_cpp_runtime_accepts_cpp_options(
        self, mock_start, mock_signal, mock_trt, mock_serve, mock_ray
    ):
        deploy = DeployRay()

        mock_app = MagicMock()
        mock_options = MagicMock()
        mock_options.bind.return_value = mock_app
        mock_trt.options.return_value = mock_options

        deploy.deploy_tensorrt_llm_model(
            trt_llm_path="/path/to/trt-llm",
            model_id="trt-model",
            use_python_runtime=False,
            enable_chunked_context=True,
            max_tokens_in_paged_kv_cache=12345,
            test_mode=True,
        )

        mock_start.assert_called_once()
        mock_serve.run.assert_called_once_with(mock_app, name="trt-model")
        # Ensure C++ runtime specific args were bound
        _, bind_kwargs = mock_options.bind.call_args
        assert bind_kwargs["enable_chunked_context"] is True
        assert bind_kwargs["max_tokens_in_paged_kv_cache"] == 12345

    @patch("nemo_deploy.deploy_ray.ray")
    @patch("nemo_deploy.deploy_ray.serve")
    def test_stop(self, mock_serve, mock_ray):
        # Test stopping Ray Serve and Ray via _stop()
        deploy = DeployRay()
        deploy._stop()

        mock_serve.shutdown.assert_called_once()
        mock_ray.shutdown.assert_called_once()

    @patch("nemo_deploy.deploy_ray.ray")
    @patch("nemo_deploy.deploy_ray.serve")
    @patch("nemo_deploy.deploy_ray.LOGGER")
    def test_stop_with_errors(self, mock_logger, mock_serve, mock_ray):
        # Test handling errors during stop
        mock_serve.shutdown.side_effect = Exception("Serve shutdown error")
        mock_ray.shutdown.side_effect = Exception("Ray shutdown error")

        deploy = DeployRay()
        deploy._stop()

        # Verify we log warnings but don't crash
        assert mock_logger.warning.call_count == 2
        mock_logger.warning.assert_any_call("Error during serve.shutdown(): Serve shutdown error")
        mock_logger.warning.assert_any_call("Error during ray.shutdown(): Ray shutdown error")


class TestDeployRayInFrameworkScriptJsonType(unittest.TestCase):
    """Test suite for deploy_ray_inframework.py script's json_type function."""

    def test_json_type_valid_json(self):
        """Test json_type with valid JSON strings."""
        # Test valid dictionary
        result = json_type('{"key": "value", "number": 42}')
        self.assertEqual(result, {"key": "value", "number": 42})

        # Test valid list
        result = json_type("[1, 2, 3]")
        self.assertEqual(result, [1, 2, 3])

        # Test nested structure
        result = json_type('{"pip": ["numpy", "pandas"], "env_vars": {"PATH": "/usr/bin"}}')
        expected = {"pip": ["numpy", "pandas"], "env_vars": {"PATH": "/usr/bin"}}
        self.assertEqual(result, expected)

    def test_json_type_invalid_json(self):
        """Test json_type with invalid JSON strings."""
        with self.assertRaises(argparse.ArgumentTypeError) as context:
            json_type("not a valid json")
        self.assertIn("Invalid JSON", str(context.exception))

        with self.assertRaises(argparse.ArgumentTypeError) as context:
            json_type('{"incomplete": ')
        self.assertIn("Invalid JSON", str(context.exception))

    def test_json_type_empty_json(self):
        """Test json_type with empty JSON objects."""
        result = json_type("{}")
        self.assertEqual(result, {})

        result = json_type("[]")
        self.assertEqual(result, [])


if __name__ == "__main__":
    unittest.main()
