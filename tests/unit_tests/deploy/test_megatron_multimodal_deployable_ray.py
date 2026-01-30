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
def mock_multimodal_model():
    """Mock the MegatronMultimodalDeployable model to avoid loading real models."""
    with patch("nemo_deploy.multimodal.megatron_multimodal_deployable_ray.MegatronMultimodalDeployable") as mock:
        mock_instance = MagicMock()

        # Mock the ray_infer_fn method for multimodal inference
        mock_instance.ray_infer_fn.return_value = {
            "sentences": [["Generated multimodal response"]],
        }

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
def mock_model_worker(mock_multimodal_model):
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
            self.model = mock_multimodal_model.return_value
            # Initialize successfully
            print(f"MockModelWorker initialized with args: {args[:2] if args else []}")  # Limit logging

        def infer(self, inputs):
            """Mock inference method that returns consistent multimodal results."""
            try:
                # Ensure we always return a valid response structure
                result = {
                    "sentences": [["Generated multimodal response"]],
                }
                print(f"MockModelWorker.infer called with keys: {list(inputs.keys()) if inputs else []}")
                return result
            except Exception as e:
                # Log but don't raise exceptions that might cause serialization issues
                print(f"Mock inference error: {e}")
                return {
                    "sentences": [["Error response"]],
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

    def test_chat_completions_endpoint_with_base64_image(
        self,
        mock_megatron_checkpoint,
        mock_model_worker,
        mock_environment_setup,
        ray_cluster,
        cleanup_serve,
        sample_image_base64,
    ):
        """Test chat completions endpoint with base64 image."""
        deployment_handle = MegatronMultimodalRayDeployable.bind(
            megatron_checkpoint_filepath=mock_megatron_checkpoint,
            num_gpus=1,
            model_id="test-chat-multimodal-model",
        )

        serve.run(deployment_handle, name="test-chat-multimodal-deployment")

        request_data = {
            "model": "test-chat-multimodal-model",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image"},
                        {"type": "image_url", "image_url": {"url": sample_image_base64}},
                    ],
                }
            ],
            "max_tokens": 100,
        }

        response = requests.post(
            "http://127.0.0.1:8000/v1/chat/completions/",
            json=request_data,
            timeout=30,
        ).json()

        assert "id" in response
        assert response["object"] == "chat.completion"
        assert "created" in response
        assert "choices" in response
        assert len(response["choices"]) > 0
        assert "message" in response["choices"][0]
        assert "role" in response["choices"][0]["message"]
        assert response["choices"][0]["message"]["role"] == "assistant"
        assert "content" in response["choices"][0]["message"]

    def test_chat_completions_endpoint_with_url_image(
        self,
        mock_megatron_checkpoint,
        mock_model_worker,
        mock_environment_setup,
        ray_cluster,
        cleanup_serve,
        sample_image_url,
    ):
        """Test chat completions endpoint with image URL."""
        deployment_handle = MegatronMultimodalRayDeployable.bind(
            megatron_checkpoint_filepath=mock_megatron_checkpoint,
            num_gpus=1,
            model_id="test-chat-url-model",
        )

        serve.run(deployment_handle, name="test-chat-url-deployment")

        request_data = {
            "model": "test-chat-url-model",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What's in this image?"},
                        {"type": "image_url", "image_url": {"url": sample_image_url}},
                    ],
                }
            ],
            "max_tokens": 50,
        }

        response = requests.post(
            "http://127.0.0.1:8000/v1/chat/completions/",
            json=request_data,
            timeout=30,
        ).json()

        assert "choices" in response
        assert len(response["choices"]) > 0
        assert "message" in response["choices"][0]
        assert "content" in response["choices"][0]["message"]

    def test_completions_endpoint_with_image(
        self,
        mock_megatron_checkpoint,
        mock_model_worker,
        mock_environment_setup,
        ray_cluster,
        cleanup_serve,
        sample_image_base64,
    ):
        """Test completions endpoint with image."""
        deployment_handle = MegatronMultimodalRayDeployable.bind(
            megatron_checkpoint_filepath=mock_megatron_checkpoint,
            num_gpus=1,
            model_id="test-completions-multimodal-model",
        )

        serve.run(deployment_handle, name="test-completions-multimodal-deployment")

        request_data = {
            "model": "test-completions-multimodal-model",
            "prompt": "Describe this image in detail.",
            "image": sample_image_base64,
            "max_tokens": 100,
        }

        response = requests.post(
            "http://127.0.0.1:8000/v1/completions/",
            json=request_data,
            timeout=30,
        ).json()

        assert "id" in response
        assert response["object"] == "text_completion"
        assert "created" in response
        assert "choices" in response
        assert len(response["choices"]) > 0
        assert "text" in response["choices"][0]

    def test_completions_endpoint_without_image(
        self,
        mock_megatron_checkpoint,
        mock_model_worker,
        mock_environment_setup,
        ray_cluster,
        cleanup_serve,
    ):
        """Test completions endpoint without image (text-only)."""
        deployment_handle = MegatronMultimodalRayDeployable.bind(
            megatron_checkpoint_filepath=mock_megatron_checkpoint,
            num_gpus=1,
            model_id="test-text-only-model",
        )

        serve.run(deployment_handle, name="test-text-only-deployment")

        request_data = {
            "model": "test-text-only-model",
            "prompt": "Hello, world!",
            "max_tokens": 50,
        }

        response = requests.post(
            "http://127.0.0.1:8000/v1/completions/",
            json=request_data,
            timeout=30,
        ).json()

        assert "choices" in response
        assert len(response["choices"]) > 0
        assert "text" in response["choices"][0]

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

    def test_chat_completions_with_temperature(
        self,
        mock_megatron_checkpoint,
        mock_model_worker,
        mock_environment_setup,
        ray_cluster,
        cleanup_serve,
        sample_image_base64,
    ):
        """Test chat completions with custom temperature."""
        deployment_handle = MegatronMultimodalRayDeployable.bind(
            megatron_checkpoint_filepath=mock_megatron_checkpoint,
            num_gpus=1,
            model_id="test-temperature-model",
        )

        serve.run(deployment_handle, name="test-temperature-deployment")

        request_data = {
            "model": "test-temperature-model",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this image"},
                        {"type": "image_url", "image_url": {"url": sample_image_base64}},
                    ],
                }
            ],
            "max_tokens": 50,
            "temperature": 0.7,
            "top_k": 10,
            "top_p": 0.9,
        }

        response = requests.post(
            "http://127.0.0.1:8000/v1/chat/completions/",
            json=request_data,
            timeout=30,
        ).json()

        assert "choices" in response
        assert len(response["choices"]) > 0

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

    def test_chat_completions_with_multiple_images(
        self,
        mock_megatron_checkpoint,
        mock_model_worker,
        mock_environment_setup,
        ray_cluster,
        cleanup_serve,
        sample_image_base64,
        sample_image_url,
    ):
        """Test chat completions with multiple images."""
        deployment_handle = MegatronMultimodalRayDeployable.bind(
            megatron_checkpoint_filepath=mock_megatron_checkpoint,
            num_gpus=1,
            model_id="test-multi-image-model",
        )

        serve.run(deployment_handle, name="test-multi-image-deployment")

        request_data = {
            "model": "test-multi-image-model",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Compare these images"},
                        {"type": "image_url", "image_url": {"url": sample_image_base64}},
                        {"type": "image_url", "image_url": {"url": sample_image_url}},
                    ],
                }
            ],
            "max_tokens": 100,
        }

        response = requests.post(
            "http://127.0.0.1:8000/v1/chat/completions/",
            json=request_data,
            timeout=30,
        ).json()

        assert "choices" in response
        assert len(response["choices"]) > 0
        assert "message" in response["choices"][0]

    def test_completions_with_custom_params(
        self,
        mock_megatron_checkpoint,
        mock_model_worker,
        mock_environment_setup,
        ray_cluster,
        cleanup_serve,
        sample_image_base64,
    ):
        """Test completions with custom inference parameters."""
        deployment_handle = MegatronMultimodalRayDeployable.bind(
            megatron_checkpoint_filepath=mock_megatron_checkpoint,
            num_gpus=1,
            model_id="test-custom-params-model",
        )

        serve.run(deployment_handle, name="test-custom-params-deployment")

        request_data = {
            "model": "test-custom-params-model",
            "prompt": "Analyze this image",
            "image": sample_image_base64,
            "max_tokens": 200,
            "temperature": 0.5,
            "top_k": 50,
            "top_p": 0.95,
            "random_seed": 42,
            "max_batch_size": 8,
        }

        response = requests.post(
            "http://127.0.0.1:8000/v1/completions/",
            json=request_data,
            timeout=30,
        ).json()

        assert "choices" in response
        assert len(response["choices"]) > 0

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

    def test_chat_completions_openai_format(
        self,
        mock_megatron_checkpoint,
        mock_model_worker,
        mock_environment_setup,
        ray_cluster,
        cleanup_serve,
        sample_image_url,
    ):
        """Test chat completions with OpenAI-style image format (normalized internally)."""
        deployment_handle = MegatronMultimodalRayDeployable.bind(
            megatron_checkpoint_filepath=mock_megatron_checkpoint,
            num_gpus=1,
            model_id="test-openai-format-model",
        )

        serve.run(deployment_handle, name="test-openai-format-deployment")

        # Using OpenAI-style image_url format
        request_data = {
            "model": "test-openai-format-model",
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What do you see?"},
                        {"type": "image_url", "image_url": {"url": sample_image_url}},
                    ],
                }
            ],
            "max_tokens": 75,
        }

        response = requests.post(
            "http://127.0.0.1:8000/v1/chat/completions/",
            json=request_data,
            timeout=30,
        ).json()

        assert "choices" in response
        assert "message" in response["choices"][0]
        assert response["choices"][0]["message"]["role"] == "assistant"

    def test_completions_with_prompts_list(
        self,
        mock_megatron_checkpoint,
        mock_model_worker,
        mock_environment_setup,
        ray_cluster,
        cleanup_serve,
    ):
        """Test completions endpoint with prompts as a list."""
        deployment_handle = MegatronMultimodalRayDeployable.bind(
            megatron_checkpoint_filepath=mock_megatron_checkpoint,
            num_gpus=1,
            model_id="test-prompts-list-model",
        )

        serve.run(deployment_handle, name="test-prompts-list-deployment")

        request_data = {
            "model": "test-prompts-list-model",
            "prompts": ["Prompt 1", "Prompt 2"],
            "max_tokens": 50,
        }

        response = requests.post(
            "http://127.0.0.1:8000/v1/completions/",
            json=request_data,
            timeout=30,
        ).json()

        assert "choices" in response
        assert len(response["choices"]) > 0

    def test_completions_with_images_list(
        self,
        mock_megatron_checkpoint,
        mock_model_worker,
        mock_environment_setup,
        ray_cluster,
        cleanup_serve,
        sample_image_base64,
        sample_image_url,
    ):
        """Test completions endpoint with images as a list."""
        deployment_handle = MegatronMultimodalRayDeployable.bind(
            megatron_checkpoint_filepath=mock_megatron_checkpoint,
            num_gpus=1,
            model_id="test-images-list-model",
        )

        serve.run(deployment_handle, name="test-images-list-deployment")

        request_data = {
            "model": "test-images-list-model",
            "prompt": "Describe these images",
            "images": [sample_image_base64, sample_image_url],
            "max_tokens": 100,
        }

        response = requests.post(
            "http://127.0.0.1:8000/v1/completions/",
            json=request_data,
            timeout=30,
        ).json()

        assert "choices" in response
        assert len(response["choices"]) > 0
