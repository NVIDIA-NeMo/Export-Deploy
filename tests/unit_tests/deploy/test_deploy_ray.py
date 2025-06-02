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

import pytest

from nemo_deploy.deploy_ray import DeployRay


class TestDeployRay(unittest.TestCase):
    @patch("nemo_deploy.deploy_ray.ray")
    def test_init_with_existing_cluster(self, mock_ray):
        # Test initialization connecting to existing cluster
        DeployRay(address="auto", num_cpus=2, num_gpus=1)
        mock_ray.init.assert_called_once_with(
            address="auto", ignore_reinit_error=True, runtime_env=None
        )

    @patch("nemo_deploy.deploy_ray.ray")
    def test_init_with_runtime_env(self, mock_ray):
        # Test initialization with custom runtime environment
        runtime_env = {"pip": ["numpy", "pandas"]}
        DeployRay(runtime_env=runtime_env)
        mock_ray.init.assert_called_once_with(
            address="auto", ignore_reinit_error=True, runtime_env=runtime_env
        )

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

    @patch("nemo_deploy.deploy_ray.ray")
    @patch("nemo_deploy.deploy_ray.serve")
    def test_start_with_port(self, mock_serve, mock_ray):
        # Test starting Ray Serve with specified port
        deploy = DeployRay()
        deploy.start(host="localhost", port=8080)

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
        # Test starting Ray Serve with auto-detected port
        mock_find_port.return_value = 9090

        deploy = DeployRay()
        deploy.start(host="0.0.0.0")

        mock_find_port.assert_called_once_with(8000, "0.0.0.0")
        mock_serve.start.assert_called_once_with(
            http_options={
                "host": "0.0.0.0",
                "port": 9090,
            }
        )

    @patch("nemo_deploy.deploy_ray.ray")
    @patch("nemo_deploy.deploy_ray.serve")
    def test_run(self, mock_serve, mock_ray):
        # Test running a model
        deploy = DeployRay()
        mock_app = MagicMock()

        deploy.run(mock_app, "test_model")

        mock_serve.run.assert_called_once_with(mock_app, name="test_model")

    @patch("nemo_deploy.deploy_ray.ray")
    @patch("nemo_deploy.deploy_ray.serve")
    def test_stop(self, mock_serve, mock_ray):
        # Test stopping Ray Serve and Ray
        deploy = DeployRay()
        deploy.stop()

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
        deploy.stop()

        # Verify we log warnings but don't crash
        assert mock_logger.warning.call_count == 2
        mock_logger.warning.assert_any_call(
            "Error during serve.shutdown(): Serve shutdown error"
        )
        mock_logger.warning.assert_any_call(
            "Error during ray.shutdown(): Ray shutdown error"
        )


if __name__ == "__main__":
    unittest.main()
