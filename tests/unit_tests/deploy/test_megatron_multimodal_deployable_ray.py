# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
import base64
import os
import tempfile
from unittest.mock import MagicMock, patch

import pytest
import ray
import requests
from ray import serve

from nemo_deploy.deploy_ray import DeployRay
from nemo_deploy.multimodal.megatron_multimodal_deployable_ray import (
    MegatronMultimodalRayDeployable,
    ModelWorker,
)

# Fixtures for Ray cluster setup and model mocking


@pytest.fixture(scope="session")
def ray_cluster():
    """Setup a real Ray cluster for testing."""
    # Initialize Ray for testing
    if not ray.is_initialized():
        ray.init(num_cpus=8, num_gpus=1, include_dashboard=False, ignore_reinit_error=True)

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
        deploy_ray._stop()
    except Exception:
        pass


@pytest.fixture
def mock_megatron_checkpoint():
    """Create a temporary mock Megatron checkpoint directory."""
    with tempfile.NamedTemporaryFile(suffix=".megatron", delete=False) as f:
        checkpoint_path = f.name
        f.write(b"mock checkpoint data")

    yield checkpoint_path

    # Cleanup
    try:
        os.unlink(checkpoint_path)
    except Exception:
        pass


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
def mock_model_worker():
    """Mock ModelWorker class to avoid loading real models and GPU requirements."""
    _ = ModelWorker

    # Create a mock worker that returns results directly without loading models
    @ray.remote(num_gpus=0)  # Use CPU for testing
    class MockModelWorker:
        def __init__(self, *args, **kwargs):
            # Store arguments but don't load any model
            self.args = args
            self.kwargs = kwargs
            self.rank = kwargs.get("rank", 0)
            print(f"MockModelWorker initialized for rank {self.rank}")

        def infer(self, inputs):
            """Return mock inference results directly."""
            return {
                "sentences": [["Generated multimodal response"]],
            }

    # Patch the original ModelWorker with our mock
    with patch("nemo_deploy.multimodal.megatron_multimodal_deployable_ray.ModelWorker", MockModelWorker):
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


@pytest.fixture
def sample_image_base64():
    """Create a sample base64-encoded image for testing."""
    # Create a simple 1x1 pixel image
    image_data = base64.b64encode(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00").decode("utf-8")
    return f"data:image/png;base64,{image_data}"


@pytest.fixture
def sample_image_url():
    """Return a sample image URL for testing."""
    return "https://example.com/test-image.jpg"


# Helper function to run async functions in tests
def run_async(coro):
    """Helper function to run async coroutines in tests."""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


class TestMegatronMultimodalRayDeployable:
    """Test suite for MegatronMultimodalRayDeployable class with real Ray integration."""

    def test_deploy_ray_initialization(self, deploy_ray_instance):
        """Test that DeployRay initializes correctly with real Ray."""
        assert deploy_ray_instance is not None
        assert ray.is_initialized()

    def test_deploy_ray_start_and_stop(self, deploy_ray_instance):
        """Test starting and stopping Ray Serve."""
        # Start Ray Serve
        deploy_ray_instance._start()

        # Verify serve is running - check if status function works
        status = serve.status()
        assert status is not None  # Just verify we can get status

        # Stop should work without errors
        deploy_ray_instance._stop()

    def test_multimodal_ray_deployable_initialization_single_gpu(
        self,
        mock_megatron_checkpoint,
        mock_model_worker,
        mock_environment_setup,
        ray_cluster,
        cleanup_serve,
    ):
        """Test basic initialization of MegatronMultimodalRayDeployable with single GPU."""
        deployment_handle = MegatronMultimodalRayDeployable.bind(
            megatron_checkpoint_filepath=mock_megatron_checkpoint,
            num_gpus=1,
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            model_id="test-multimodal-model",
        )

        # Deploy the deployment
        serve.run(deployment_handle, name="test-multimodal-deployment")

        # Test that we can call the endpoints
        health_response = requests.get("http://127.0.0.1:8000/v1/health", timeout=10).json()
        assert health_response["status"] == "healthy"

        models_response = requests.get("http://127.0.0.1:8000/v1/models", timeout=10).json()
        assert models_response["object"] == "list"
        assert len(models_response["data"]) == 1
        assert models_response["data"][0]["id"] == "test-multimodal-model"

    def test_multimodal_ray_deployable_initialization_multi_gpu(
        self,
        mock_megatron_checkpoint,
        mock_model_worker,
        mock_environment_setup,
        ray_cluster,
        cleanup_serve,
    ):
        """Test initialization with multiple GPUs."""
        deployment_handle = MegatronMultimodalRayDeployable.bind(
            megatron_checkpoint_filepath=mock_megatron_checkpoint,
            num_gpus=2,
            tensor_model_parallel_size=2,
            pipeline_model_parallel_size=1,
            model_id="test-multi-gpu-multimodal-model",
        )

        # Deploy the deployment
        serve.run(deployment_handle, name="test-multi-gpu-multimodal-deployment")

        # Test basic functionality
        health_response = requests.get("http://127.0.0.1:8000/v1/health", timeout=10).json()
        assert health_response["status"] == "healthy"

    def test_model_worker_initialization_and_infer(
        self,
        mock_megatron_checkpoint,
        mock_model_worker,
    ):
        """Test ModelWorker by accessing underlying class without Ray remote decorator."""
        # Access the underlying class without the @ray.remote decorator
        from nemo_deploy.multimodal.megatron_multimodal_deployable_ray import ModelWorker

        # Get the actual class (not the remote wrapper)
        ModelWorkerClass = ModelWorker.__ray_metadata__.modified_class

        # Create instance directly with mock
        with patch(
            "nemo_deploy.multimodal.megatron_multimodal_deployable_ray.MegatronMultimodalDeployable"
        ) as mock_deployable:
            mock_instance = MagicMock()
            mock_instance.ray_infer_fn.return_value = {"sentences": [["Generated multimodal response"]]}
            mock_deployable.return_value = mock_instance

            # Create worker instance directly
            worker = ModelWorkerClass(
                megatron_checkpoint_filepath=mock_megatron_checkpoint,
                rank=0,
                world_size=1,
                tensor_model_parallel_size=1,
                pipeline_model_parallel_size=1,
                master_port="29500",
                master_addr="127.0.0.1",
                replica_id=0,
            )

            # Verify worker was created and has rank
            assert worker is not None
            assert worker.rank == 0

            # Test inference method directly
            inputs = {
                "prompts": ["Test prompt"],
                "images": [],
                "max_length": 50,
            }

            result = worker.infer(inputs)

            # Verify result structure
            assert "sentences" in result
            assert len(result["sentences"]) > 0
            assert result["sentences"][0] == ["Generated multimodal response"]

    def test_list_models_endpoint(
        self,
        mock_megatron_checkpoint,
        mock_model_worker,
        mock_environment_setup,
        ray_cluster,
        cleanup_serve,
    ):
        """Test list models endpoint."""
        deployment_handle = MegatronMultimodalRayDeployable.bind(
            megatron_checkpoint_filepath=mock_megatron_checkpoint,
            num_gpus=1,
            model_id="test-list-models-multimodal",
        )

        serve.run(deployment_handle, name="test-list-models-multimodal-deployment")

        response = requests.get("http://127.0.0.1:8000/v1/models", timeout=10).json()

        assert response["object"] == "list"
        assert len(response["data"]) == 1
        assert response["data"][0]["id"] == "test-list-models-multimodal"
        assert response["data"][0]["object"] == "model"
        assert "created" in response["data"][0]

    def test_health_check_endpoint(
        self,
        mock_megatron_checkpoint,
        mock_model_worker,
        mock_environment_setup,
        ray_cluster,
        cleanup_serve,
    ):
        """Test health check endpoint."""
        deployment_handle = MegatronMultimodalRayDeployable.bind(
            megatron_checkpoint_filepath=mock_megatron_checkpoint,
            num_gpus=1,
            model_id="test-health-multimodal-model",
        )

        serve.run(deployment_handle, name="test-health-multimodal-deployment")

        response = requests.get("http://127.0.0.1:8000/v1/health", timeout=10).json()

        assert response["status"] == "healthy"

    def test_pipeline_parallelism_initialization(
        self,
        mock_megatron_checkpoint,
        mock_model_worker,
        mock_environment_setup,
        ray_cluster,
        cleanup_serve,
    ):
        """Test initialization with pipeline parallelism."""
        deployment_handle = MegatronMultimodalRayDeployable.bind(
            megatron_checkpoint_filepath=mock_megatron_checkpoint,
            num_gpus=4,
            tensor_model_parallel_size=2,
            pipeline_model_parallel_size=2,
            model_id="test-pipeline-multimodal-model",
        )

        serve.run(deployment_handle, name="test-pipeline-multimodal-deployment")

        health_response = requests.get("http://127.0.0.1:8000/v1/health", timeout=10).json()
        assert health_response["status"] == "healthy"

    def test_multi_node_initialization(
        self,
        mock_megatron_checkpoint,
        mock_model_worker,
        mock_environment_setup,
        ray_cluster,
        cleanup_serve,
    ):
        """Test initialization with multiple nodes."""
        deployment_handle = MegatronMultimodalRayDeployable.bind(
            megatron_checkpoint_filepath=mock_megatron_checkpoint,
            num_gpus=4,
            tensor_model_parallel_size=4,
            pipeline_model_parallel_size=1,
            model_id="test-multi-node-multimodal-model",
        )

        serve.run(deployment_handle, name="test-multi-node-multimodal-deployment")

        health_response = requests.get("http://127.0.0.1:8000/v1/health", timeout=10).json()
        assert health_response["status"] == "healthy"

    def test_initialization_with_custom_inference_params(
        self,
        mock_megatron_checkpoint,
        mock_model_worker,
        mock_environment_setup,
        ray_cluster,
        cleanup_serve,
    ):
        """Test initialization with custom inference parameters."""
        deployment_handle = MegatronMultimodalRayDeployable.bind(
            megatron_checkpoint_filepath=mock_megatron_checkpoint,
            num_gpus=1,
            model_id="test-custom-init-model",
            params_dtype="bfloat16",
            inference_batch_times_seqlen_threshold=2000,
            inference_max_seq_length=4096,
        )

        serve.run(deployment_handle, name="test-custom-init-deployment")

        health_response = requests.get("http://127.0.0.1:8000/v1/health", timeout=10).json()
        assert health_response["status"] == "healthy"
