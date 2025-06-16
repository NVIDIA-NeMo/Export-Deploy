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

import asyncio
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
import ray
from ray import serve

from nemo_deploy.deploy_ray import DeployRay
from nemo_deploy.nlp.megatronllm_deployable_ray import (
    MegatronRayDeployable,
    ModelWorker,
)

# Fixtures for Ray cluster setup and model mocking


@pytest.fixture(scope="session")
def ray_cluster():
    """Setup a real Ray cluster for testing."""
    # Initialize Ray for testing
    if not ray.is_initialized():
        ray.init(num_cpus=4, num_gpus=0, include_dashboard=False, ignore_reinit_error=True)

    yield

    # Cleanup
    try:
        if ray.is_initialized():
            serve.shutdown()
            ray.shutdown()
    except Exception:
        pass  # Ignore shutdown errors


@pytest.fixture
def deploy_ray_instance(ray_cluster):
    """Create a DeployRay instance for testing with real Ray."""
    deploy_ray = DeployRay(
        address="auto",  # Connect to existing cluster
        num_cpus=2,
        num_gpus=0,  # Use CPU-only for testing
        include_dashboard=False,
        ignore_reinit_error=True,
    )

    yield deploy_ray

    # Cleanup
    try:
        deploy_ray.stop()
    except Exception:
        pass


@pytest.fixture
def mock_nemo_checkpoint():
    """Create a temporary mock .nemo checkpoint file."""
    with tempfile.NamedTemporaryFile(suffix=".nemo", delete=False) as f:
        checkpoint_path = f.name
        f.write(b"mock checkpoint data")

    yield checkpoint_path

    # Cleanup
    try:
        os.unlink(checkpoint_path)
    except Exception:
        pass


@pytest.fixture
def mock_megatron_model():
    """Mock the MegatronLLMDeployableNemo2 model to avoid loading real models."""
    with patch("nemo_deploy.nlp.megatronllm_deployable_ray.MegatronLLMDeployableNemo2") as mock:
        mock_instance = MagicMock()

        # Mock the ray_infer_fn method
        mock_instance.ray_infer_fn.return_value = {
            "sentences": ["Generated response 1", "Generated response 2"],
            "log_probs": [
                [0.1, 0.2],
                [0.3, 0.4],
            ],  # Use regular lists instead of numpy arrays
        }

        # Mock generate_other_ranks for non-rank-0 workers
        mock_instance.generate_other_ranks = MagicMock()

        mock.return_value = mock_instance
        yield mock


@pytest.fixture
def mock_environment_setup():
    """Mock environment variables and system calls."""
    with (
        patch.dict(os.environ, {}, clear=False),
        patch("torch.cuda.device_count") as mock_device_count,
        patch("time.sleep") as mock_sleep,
    ):
        mock_device_count.return_value = 1
        mock_sleep.return_value = None

        yield {"device_count": mock_device_count, "sleep": mock_sleep}


@pytest.fixture
def mock_model_worker(mock_megatron_model):
    """Mock ModelWorker class while preserving Ray remote functionality."""
    _ = ModelWorker

    # Create a new class that inherits from the original but mocks the model loading
    @ray.remote(num_gpus=0)  # Use CPU for testing
    class MockModelWorker:
        def __init__(self, *args, **kwargs):
            # Store the arguments for verification
            self.args = args
            self.kwargs = kwargs
            # Create a mock model instead of loading real one
            self.model = mock_megatron_model.return_value
            # Initialize successfully
            print(f"MockModelWorker initialized with args: {args[:2] if args else []}")  # Limit logging

        def infer(self, inputs):
            """Mock inference method that returns consistent results."""
            try:
                # Ensure we always return a valid response structure
                result = {
                    "sentences": ["Generated response 1", "Generated response 2"],
                    "log_probs": [
                        [0.1, 0.2],
                        [0.3, 0.4],
                    ],  # Use regular lists instead of numpy arrays
                }
                print(f"MockModelWorker.infer called with keys: {list(inputs.keys()) if inputs else []}")
                return result
            except Exception as e:
                # Log but don't raise exceptions that might cause serialization issues
                print(f"Mock inference error: {e}")
                return {
                    "sentences": ["Error response"],
                    "log_probs": [[0.0]],
                }

    # Patch the original ModelWorker with our mock
    with patch("nemo_deploy.nlp.megatronllm_deployable_ray.ModelWorker", MockModelWorker):
        yield MockModelWorker


@pytest.fixture
def cleanup_serve():
    """Cleanup fixture to ensure serve applications are shut down after each test."""
    yield
    # Cleanup after each test
    try:
        serve.shutdown()
    except Exception:
        pass  # Ignore errors during cleanup


# Helper function to run async functions in tests
def run_async(coro):
    """Helper function to run async coroutines in tests."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


class TestMegatronRayDeployable:
    """Test suite for MegatronRayDeployable class with real Ray integration."""

    def test_deploy_ray_initialization(self, deploy_ray_instance):
        """Test that DeployRay initializes correctly with real Ray."""
        assert deploy_ray_instance is not None
        assert ray.is_initialized()

    def test_deploy_ray_start_and_stop(self, deploy_ray_instance):
        """Test starting and stopping Ray Serve."""
        # Start Ray Serve
        deploy_ray_instance.start(port=8000)

        # Verify serve is running - check if status function works
        status = serve.status()
        assert status is not None  # Just verify we can get status

        # Stop should work without errors
        deploy_ray_instance.stop()

    def test_megatron_ray_deployable_initialization_single_gpu(
        self,
        mock_nemo_checkpoint,
        mock_model_worker,
        mock_environment_setup,
        ray_cluster,
        cleanup_serve,
    ):
        """Test basic initialization of MegatronRayDeployable with single GPU."""
        deployment_handle = MegatronRayDeployable.bind(
            nemo_checkpoint_filepath=mock_nemo_checkpoint,
            num_gpus=1,
            num_nodes=1,
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            context_parallel_size=1,
            model_id="test-model",
        )

        # Deploy the deployment
        serve.run(deployment_handle, name="test-model-deployment")

        # Get a handle to interact with the deployment
        deployable = serve.get_app_handle("test-model-deployment")

        # Test that we can call the endpoints
        health_response = deployable.health_check.remote().result()
        assert health_response["status"] == "healthy"

        models_response = deployable.list_models.remote().result()
        assert models_response["object"] == "list"
        assert len(models_response["data"]) == 1
        assert models_response["data"][0]["id"] == "test-model"

    def test_megatron_ray_deployable_initialization_multi_gpu(
        self,
        mock_nemo_checkpoint,
        mock_model_worker,
        mock_environment_setup,
        ray_cluster,
        cleanup_serve,
    ):
        """Test initialization with multiple GPUs."""
        deployment_handle = MegatronRayDeployable.bind(
            nemo_checkpoint_filepath=mock_nemo_checkpoint,
            num_gpus=2,
            num_nodes=1,
            tensor_model_parallel_size=2,
            pipeline_model_parallel_size=1,
            context_parallel_size=1,
            model_id="test-multi-gpu-model",
        )

        # Deploy the deployment
        serve.run(deployment_handle, name="test-multi-gpu-deployment")

        # Get a handle to interact with the deployment
        deployable = serve.get_app_handle("test-multi-gpu-deployment")

        # Test basic functionality
        health_response = deployable.health_check.remote().result()
        assert health_response["status"] == "healthy"

    def test_megatron_ray_deployable_invalid_parallelism(
        self,
        mock_nemo_checkpoint,
        mock_model_worker,
        mock_environment_setup,
        ray_cluster,
        cleanup_serve,
    ):
        """Test initialization with invalid parallelism configuration."""
        deployment_handle = MegatronRayDeployable.bind(
            nemo_checkpoint_filepath=mock_nemo_checkpoint,
            num_gpus=1,
            num_nodes=1,
            tensor_model_parallel_size=2,  # This would require 2 GPUs
            pipeline_model_parallel_size=1,
            context_parallel_size=1,
        )

        # This should fail when we try to deploy it
        with pytest.raises(Exception):  # Ray will wrap the ValueError
            serve.run(deployment_handle, name="test-invalid-parallelism")

    def test_list_models_endpoint(
        self,
        mock_nemo_checkpoint,
        mock_model_worker,
        mock_environment_setup,
        ray_cluster,
        cleanup_serve,
    ):
        """Test list models endpoint."""
        deployment_handle = MegatronRayDeployable.bind(
            nemo_checkpoint_filepath=mock_nemo_checkpoint,
            num_gpus=1,
            num_nodes=1,
            model_id="test-list-models",
        )

        serve.run(deployment_handle, name="test-list-models-deployment")
        deployable = serve.get_app_handle("test-list-models-deployment")

        response = deployable.list_models.remote().result()

        assert response["object"] == "list"
        assert len(response["data"]) == 1
        assert response["data"][0]["id"] == "test-list-models"
        assert response["data"][0]["object"] == "model"
        assert "created" in response["data"][0]

    def test_health_check_endpoint(
        self,
        mock_nemo_checkpoint,
        mock_model_worker,
        mock_environment_setup,
        ray_cluster,
        cleanup_serve,
    ):
        """Test health check endpoint."""
        deployment_handle = MegatronRayDeployable.bind(
            nemo_checkpoint_filepath=mock_nemo_checkpoint,
            num_gpus=1,
            num_nodes=1,
            model_id="test-health-model",
        )

        serve.run(deployment_handle, name="test-health-deployment")
        deployable = serve.get_app_handle("test-health-deployment")

        response = deployable.health_check.remote().result()

        assert response["status"] == "healthy"

    def test_initialization_with_cuda_graphs(
        self,
        mock_nemo_checkpoint,
        mock_model_worker,
        mock_environment_setup,
        ray_cluster,
        cleanup_serve,
    ):
        """Test initialization with CUDA graphs enabled."""
        deployment_handle = MegatronRayDeployable.bind(
            nemo_checkpoint_filepath=mock_nemo_checkpoint,
            num_gpus=1,
            num_nodes=1,
            enable_cuda_graphs=True,
            model_id="test-cuda-graphs-model",
        )

        serve.run(deployment_handle, name="test-cuda-graphs-deployment")
        deployable = serve.get_app_handle("test-cuda-graphs-deployment")

        # Verify the deployable was created successfully
        health_response = deployable.health_check.remote().result()
        assert health_response["status"] == "healthy"

    def test_initialization_with_flash_decode(
        self,
        mock_nemo_checkpoint,
        mock_model_worker,
        mock_environment_setup,
        ray_cluster,
        cleanup_serve,
    ):
        """Test initialization with Flash Decode enabled."""
        deployment_handle = MegatronRayDeployable.bind(
            nemo_checkpoint_filepath=mock_nemo_checkpoint,
            num_gpus=1,
            num_nodes=1,
            enable_flash_decode=True,
            model_id="test-flash-decode-model",
        )

        serve.run(deployment_handle, name="test-flash-decode-deployment")
        deployable = serve.get_app_handle("test-flash-decode-deployment")

        # Verify the deployable was created successfully
        health_response = deployable.health_check.remote().result()
        assert health_response["status"] == "healthy"

    def test_initialization_with_legacy_checkpoint(
        self,
        mock_nemo_checkpoint,
        mock_model_worker,
        mock_environment_setup,
        ray_cluster,
        cleanup_serve,
    ):
        """Test initialization with legacy checkpoint format."""
        deployment_handle = MegatronRayDeployable.bind(
            nemo_checkpoint_filepath=mock_nemo_checkpoint,
            num_gpus=1,
            num_nodes=1,
            legacy_ckpt=True,
            model_id="test-legacy-ckpt-model",
        )

        serve.run(deployment_handle, name="test-legacy-ckpt-deployment")
        deployable = serve.get_app_handle("test-legacy-ckpt-deployment")

        # Verify the deployable was created successfully
        health_response = deployable.health_check.remote().result()
        assert health_response["status"] == "healthy"

    def test_multi_node_initialization(
        self,
        mock_nemo_checkpoint,
        mock_model_worker,
        mock_environment_setup,
        ray_cluster,
        cleanup_serve,
    ):
        """Test initialization with multiple nodes."""
        deployment_handle = MegatronRayDeployable.bind(
            nemo_checkpoint_filepath=mock_nemo_checkpoint,
            num_gpus=2,
            num_nodes=2,
            tensor_model_parallel_size=4,
            pipeline_model_parallel_size=1,
            context_parallel_size=1,
            model_id="test-multi-node-model",
        )

        serve.run(deployment_handle, name="test-multi-node-deployment")
        deployable = serve.get_app_handle("test-multi-node-deployment")

        health_response = deployable.health_check.remote().result()
        assert health_response["status"] == "healthy"

    def test_pipeline_parallelism_initialization(
        self,
        mock_nemo_checkpoint,
        mock_model_worker,
        mock_environment_setup,
        ray_cluster,
        cleanup_serve,
    ):
        """Test initialization with pipeline parallelism."""
        deployment_handle = MegatronRayDeployable.bind(
            nemo_checkpoint_filepath=mock_nemo_checkpoint,
            num_gpus=4,
            num_nodes=1,
            tensor_model_parallel_size=2,
            pipeline_model_parallel_size=2,
            context_parallel_size=1,
            model_id="test-pipeline-parallel-model",
        )

        serve.run(deployment_handle, name="test-pipeline-parallel-deployment")
        deployable = serve.get_app_handle("test-pipeline-parallel-deployment")

        health_response = deployable.health_check.remote().result()
        assert health_response["status"] == "healthy"

    def test_context_parallelism_initialization(
        self,
        mock_nemo_checkpoint,
        mock_model_worker,
        mock_environment_setup,
        ray_cluster,
        cleanup_serve,
    ):
        """Test initialization with context parallelism."""
        deployment_handle = MegatronRayDeployable.bind(
            nemo_checkpoint_filepath=mock_nemo_checkpoint,
            num_gpus=2,
            num_nodes=1,
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            context_parallel_size=2,
            model_id="test-context-parallel-model",
        )

        serve.run(deployment_handle, name="test-context-parallel-deployment")
        deployable = serve.get_app_handle("test-context-parallel-deployment")

        health_response = deployable.health_check.remote().result()
        assert health_response["status"] == "healthy"
