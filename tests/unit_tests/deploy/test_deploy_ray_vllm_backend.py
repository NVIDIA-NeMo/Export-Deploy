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


import unittest
from unittest.mock import MagicMock, patch

from nemo_deploy.deploy_ray import DeployRay


class TestDeployRayVLLMBackend(unittest.TestCase):
    """Test cases for the use_vllm_backend parameter in deploy_huggingface_model method."""

    def setUp(self):
        # Ensure tests run even when Ray is not installed
        self._have_ray_patcher = patch("nemo_deploy.deploy_ray.HAVE_RAY", True)
        self._have_ray_patcher.start()

    def tearDown(self):
        self._have_ray_patcher.stop()

    @patch("nemo_deploy.deploy_ray.ray")
    @patch("nemo_deploy.deploy_ray.serve")
    @patch("nemo_deploy.deploy_ray.HFRayDeployable")
    @patch("nemo_deploy.deploy_ray.signal.signal")
    @patch.object(DeployRay, "_start")
    def test_deploy_hf_with_vllm_backend_true(self, mock_start, mock_signal, mock_hf_deployable, mock_serve, mock_ray):
        """Test deploy_huggingface_model with use_vllm_backend=True."""
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
    @patch("nemo_deploy.deploy_ray.HFRayDeployable")
    @patch("nemo_deploy.deploy_ray.signal.signal")
    @patch.object(DeployRay, "_start")
    def test_deploy_hf_with_vllm_backend_false(self, mock_start, mock_signal, mock_hf_deployable, mock_serve, mock_ray):
        """Test deploy_huggingface_model with use_vllm_backend=False."""
        deploy = DeployRay()

        mock_app = MagicMock()
        mock_options = MagicMock()
        mock_options.bind.return_value = mock_app
        mock_hf_deployable.options.return_value = mock_options

        deploy.deploy_huggingface_model(
            hf_model_id_path="test-hf-model",
            model_id="test_model",
            use_vllm_backend=False,
            test_mode=True,
        )

        mock_start.assert_called_once()
        mock_serve.run.assert_called_once_with(mock_app, name="test_model")
        mock_hf_deployable.options.assert_called_once()

        # Verify the bind call includes use_vllm_backend=False
        _, bind_kwargs = mock_options.bind.call_args
        assert bind_kwargs["use_vllm_backend"] is False

    @patch("nemo_deploy.deploy_ray.ray")
    @patch("nemo_deploy.deploy_ray.serve")
    @patch("nemo_deploy.deploy_ray.HFRayDeployable")
    @patch("nemo_deploy.deploy_ray.signal.signal")
    @patch.object(DeployRay, "_start")
    def test_deploy_hf_with_vllm_backend_default(
        self, mock_start, mock_signal, mock_hf_deployable, mock_serve, mock_ray
    ):
        """Test deploy_huggingface_model with default use_vllm_backend (should be False)."""
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

        # Verify the bind call includes use_vllm_backend=False (default)
        _, bind_kwargs = mock_options.bind.call_args
        assert bind_kwargs["use_vllm_backend"] is False

    @patch("nemo_deploy.deploy_ray.ray")
    @patch("nemo_deploy.deploy_ray.serve")
    @patch("nemo_deploy.deploy_ray.HFRayDeployable")
    @patch("nemo_deploy.deploy_ray.signal.signal")
    @patch.object(DeployRay, "_start")
    def test_deploy_hf_with_vllm_backend_and_other_params(
        self, mock_start, mock_signal, mock_hf_deployable, mock_serve, mock_ray
    ):
        """Test deploy_huggingface_model with use_vllm_backend and other parameters."""
        deploy = DeployRay()

        mock_app = MagicMock()
        mock_options = MagicMock()
        mock_options.bind.return_value = mock_app
        mock_hf_deployable.options.return_value = mock_options

        deploy.deploy_huggingface_model(
            hf_model_id_path="test-hf-model",
            task="text-generation",
            trust_remote_code=True,
            device_map="auto",
            max_memory="16GB",
            model_id="test_model",
            num_replicas=2,
            num_cpus_per_replica=4,
            num_gpus_per_replica=1,
            max_ongoing_requests=5,
            use_vllm_backend=True,
            test_mode=True,
        )

        mock_start.assert_called_once()
        mock_serve.run.assert_called_once_with(mock_app, name="test_model")
        mock_hf_deployable.options.assert_called_once()

        # Verify all parameters are passed correctly
        _, bind_kwargs = mock_options.bind.call_args
        assert bind_kwargs["hf_model_id_path"] == "test-hf-model"
        assert bind_kwargs["task"] == "text-generation"
        assert bind_kwargs["trust_remote_code"] is True
        assert bind_kwargs["device_map"] == "auto"
        assert bind_kwargs["max_memory"] == "16GB"
        assert bind_kwargs["model_id"] == "test_model"
        assert bind_kwargs["use_vllm_backend"] is True

        # Verify actor options
        _, options_kwargs = mock_hf_deployable.options.call_args
        assert options_kwargs["ray_actor_options"]["num_cpus"] == 4
        assert options_kwargs["ray_actor_options"]["num_gpus"] == 1
        assert options_kwargs["num_replicas"] == 2
        assert options_kwargs["max_ongoing_requests"] == 5

    @patch("nemo_deploy.deploy_ray.ray")
    @patch("nemo_deploy.deploy_ray.serve")
    @patch("nemo_deploy.deploy_ray.HFRayDeployable")
    @patch("nemo_deploy.deploy_ray.signal.signal")
    @patch.object(DeployRay, "_start")
    def test_deploy_hf_vllm_backend_with_error_handling(
        self, mock_start, mock_signal, mock_hf_deployable, mock_serve, mock_ray
    ):
        """Test deploy_huggingface_model with use_vllm_backend=True and error handling."""
        deploy = DeployRay()

        # Simulate an error during deployment
        mock_serve.run.side_effect = Exception("Deployment failed")

        with self.assertRaises(SystemExit):
            deploy.deploy_huggingface_model(
                hf_model_id_path="test-hf-model",
                model_id="test_model",
                use_vllm_backend=True,
                test_mode=True,
            )

        # Verify _stop is called on error
        mock_serve.shutdown.assert_called_once()
        mock_ray.shutdown.assert_called_once()


if __name__ == "__main__":
    unittest.main()
