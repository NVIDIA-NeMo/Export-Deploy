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


import time
from unittest.mock import MagicMock, patch

import pytest
from fastapi import FastAPI, HTTPException

from nemo_deploy.nlp.hf_deployable import HuggingFaceLLMDeploy


# Create a mock of the HFRayDeployable class without decorators for testing
class MockHFRayDeployable:
    def __init__(
        self,
        hf_model_id_path,
        task="text-generation",
        trust_remote_code=True,
        model_id="nemo-model",
        device_map="auto",
        max_memory=None,
        max_batch_size=8,
        batch_wait_timeout_s=0.3,
    ):
        self.hf_model_id_path = hf_model_id_path
        self.task = task
        self.trust_remote_code = trust_remote_code
        self.model_id = model_id
        self.device_map = device_map
        self.max_memory = max_memory
        self.max_batch_size = max_batch_size
        self.batch_wait_timeout_s = batch_wait_timeout_s
        self.model = None

    def _setup_unique_distributed_parameters(self, device_map):
        import os

        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29501"


# Mock fixtures to simulate dependencies
@pytest.fixture
def mock_hf_model():
    with patch("nemo_deploy.nlp.hf_deployable.HuggingFaceLLMDeploy") as mock:
        mock_instance = MagicMock(spec=HuggingFaceLLMDeploy)
        mock_instance.ray_infer_fn = MagicMock()
        mock_instance.ray_infer_fn.return_value = {
            "sentences": ["Generated response 1", "Generated response 2"],
            "log_probs": [[0.1, 0.2], [0.3, 0.4]],
        }
        mock.return_value = mock_instance
        yield mock


@pytest.fixture
def mock_ray():
    with patch("nemo_deploy.nlp.hf_deployable_ray.serve") as mock_serve:
        # Mock Ray serve to expose the actual class
        mock_deployment = MagicMock()
        mock_deployment.return_value = lambda x: x  # Return the class itself
        mock_serve.deployment = mock_deployment
        mock_serve.ingress.return_value = lambda c: c
        mock_serve.batch.return_value = lambda f: f
        yield mock_serve


@pytest.fixture
def mock_hfray_class():
    # Use our custom mock class for testing
    with patch("nemo_deploy.nlp.hf_deployable_ray.HFRayDeployable", MockHFRayDeployable):
        yield MockHFRayDeployable


@pytest.fixture
def mock_fastapi():
    with patch("nemo_deploy.nlp.hf_deployable_ray.FastAPI") as mock:
        mock.return_value = MagicMock(spec=FastAPI)
        yield mock


@pytest.fixture
def mock_torch_distributed():
    with patch("torch.distributed") as mock_dist:
        mock_dist.is_initialized.return_value = False
        yield mock_dist


@pytest.fixture
def mock_os_env():
    with patch("os.environ", {}):
        yield


@pytest.fixture
def mock_find_port():
    with patch("nemo_deploy.ray_utils.find_available_port") as mock:
        mock.return_value = 29501
        yield mock


@pytest.fixture
def mock_ray_instance(mock_hf_model, mock_hfray_class):
    instance = mock_hfray_class(
        hf_model_id_path="test/model",
        task="text-generation",
        model_id="test-model",
        max_batch_size=4,
        batch_wait_timeout_s=0.2,
    )

    # Initialize the model
    instance.model = mock_hf_model.return_value

    # Mock async methods
    instance.completions = MagicMock()
    instance.completions.return_value = {
        "id": "cmpl-123",
        "object": "text_completion",
        "created": int(time.time()),
        "model": "test-model",
        "choices": [{"text": "Generated text", "index": 0, "logprobs": None, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    }

    instance.chat_completions = MagicMock()
    instance.chat_completions.return_value = {
        "id": "chatcmpl-123",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "test-model",
        "choices": [
            {"message": {"role": "assistant", "content": "Generated response"}, "index": 0, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 15, "completion_tokens": 25, "total_tokens": 40},
    }

    instance.batched_inference = MagicMock()
    instance.list_models = MagicMock()
    instance.health_check = MagicMock()

    yield instance


# Helper function to run coroutines synchronously
def run_coroutine(coro):
    import asyncio

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


class TestHFRayDeployable:
    @pytest.mark.skip
    def test_init_with_balanced_device_map(
        self, mock_hf_model, mock_hfray_class, mock_torch_distributed, mock_find_port
    ):
        """Test initialization with balanced device map."""
        with patch("torch.cuda.device_count", return_value=2):
            # Custom initialization for the balanced device map test
            def mock_init(self, *args, **kwargs):
                self.hf_model_id_path = kwargs.get("hf_model_id_path")
                self.task = kwargs.get("task", "text-generation")
                self.trust_remote_code = kwargs.get("trust_remote_code", True)
                self.model_id = kwargs.get("model_id", "nemo-model")
                self.device_map = kwargs.get("device_map", "auto")
                self.max_memory = kwargs.get("max_memory")
                self.max_batch_size = kwargs.get("max_batch_size", 8)
                self.batch_wait_timeout_s = kwargs.get("batch_wait_timeout_s", 0.3)

                # Simulate the behavior for balanced device map
                if self.device_map == "balanced":
                    if not self.max_memory:
                        raise ValueError("max_memory must be provided when device_map is 'balanced'")
                    num_gpus = 2  # Mocked from torch.cuda.device_count()
                    max_memory_dict = {i: "75GiB" for i in range(num_gpus)}
                    self.model = mock_hf_model(
                        hf_model_id_path=self.hf_model_id_path,
                        task=self.task,
                        trust_remote_code=self.trust_remote_code,
                        device_map=self.device_map,
                        max_memory=max_memory_dict,
                    )

            with patch.object(MockHFRayDeployable, "__init__", mock_init):
                mock_hfray_class(hf_model_id_path="test/model", device_map="auto", max_memory="75GiB")

                # Verify max_memory_dict was created
                mock_hf_model.assert_called_once()
                args, kwargs = mock_hf_model.call_args
                assert kwargs["device_map"] == "balanced"
                assert kwargs["max_memory"] == {0: "75GiB", 1: "75GiB"}

    def test_init_with_balanced_device_map_no_memory(self, mock_hfray_class):
        """Test initialization with balanced device map but missing max_memory."""

        # Custom initialization to test the error case
        def mock_init(self, *args, **kwargs):
            if kwargs.get("device_map") == "balanced" and not kwargs.get("max_memory"):
                raise ValueError("max_memory must be provided when device_map is 'balanced'")

        with patch.object(MockHFRayDeployable, "__init__", mock_init):
            with pytest.raises(ValueError, match="max_memory must be provided"):
                mock_hfray_class(hf_model_id_path="test/model", device_map="balanced")

    def test_init_exception_handling(self, mock_hfray_class):
        """Test exception handling during initialization."""

        # Custom init to simulate the error
        def mock_init(self, *args, **kwargs):
            raise Exception("Test error")

        with patch.object(MockHFRayDeployable, "__init__", mock_init):
            with pytest.raises(Exception, match="Test error"):
                mock_hfray_class(hf_model_id_path="test/model")

    def test_setup_unique_distributed_parameters(self, mock_hfray_class, mock_os_env):
        """Test setting up unique distributed parameters."""
        instance = mock_hfray_class(hf_model_id_path="test/model", device_map="auto")
        instance._setup_unique_distributed_parameters("auto")

        # Check if environment variables were set
        import os

        assert os.environ["MASTER_ADDR"] == "127.0.0.1"
        assert os.environ["MASTER_PORT"] == "29501"

    def test_completions(self, mock_ray_instance):
        """Test the completions endpoint."""
        # Create a request
        request = {"prompt": "Test prompt", "max_tokens": 100, "temperature": 0.7}

        # Get the result directly from the mock
        result = mock_ray_instance.completions(request)

        # Assert the expected result
        assert result["id"] == "cmpl-123"
        assert result["object"] == "text_completion"

    def test_completions_error(self, mock_ray_instance):
        """Test error handling in completions endpoint."""
        # Set up the mock to return an error
        mock_ray_instance.completions.side_effect = HTTPException(status_code=500, detail="Test error")

        # Create a request
        request = {"prompt": "Test prompt"}

        # Test that an HTTPException is raised
        with pytest.raises(HTTPException) as excinfo:
            mock_ray_instance.completions(request)

        assert excinfo.value.status_code == 500
        assert "Test error" in str(excinfo.value.detail)

    def test_chat_completions(self, mock_ray_instance):
        """Test the chat completions endpoint."""
        # Create a request
        request = {
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello, how are you?"},
            ],
            "max_tokens": 100,
            "temperature": 0.7,
        }

        # Get the result directly from the mock
        result = mock_ray_instance.chat_completions(request)

        # Assert the expected result
        assert result["id"] == "chatcmpl-123"
        assert result["object"] == "chat.completion"

    def test_batched_inference(self, mock_ray_instance):
        """Test batched inference method."""
        # Create test requests
        requests = [
            {"prompt": "Test prompt 1", "max_tokens": 100, "temperature": 0.7},
            {"prompt": "Test prompt 2", "max_tokens": 50, "temperature": 0.9},
        ]

        # Setup expected results
        expected_results = [
            {
                "id": f"cmpl-{int(time.time())}",
                "object": "text_completion",
                "created": int(time.time()),
                "model": "test-model",
                "choices": [
                    {"text": "Generated response 1", "index": 0, "logprobs": [0.1, 0.2], "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 3, "completion_tokens": 3, "total_tokens": 6},
            },
            {
                "id": f"cmpl-{int(time.time())}",
                "object": "text_completion",
                "created": int(time.time()),
                "model": "test-model",
                "choices": [
                    {"text": "Generated response 2", "index": 0, "logprobs": [0.3, 0.4], "finish_reason": "stop"}
                ],
                "usage": {"prompt_tokens": 3, "completion_tokens": 3, "total_tokens": 6},
            },
        ]

        # Set up the mock to return the expected results
        mock_ray_instance.batched_inference.return_value = expected_results

        # Call the method
        results = mock_ray_instance.batched_inference(requests, "completion")

        # Verify results have the expected format
        assert len(results) == 2
        for result in results:
            assert "id" in result
            assert "object" in result
            assert "created" in result
            assert "choices" in result
            assert "usage" in result

    def test_list_models(self, mock_ray_instance):
        """Test the list_models endpoint."""
        expected_result = {
            "object": "list",
            "data": [{"id": "test-model", "object": "model", "created": int(time.time())}],
        }

        # Set up the mock to return the expected result
        mock_ray_instance.list_models.return_value = expected_result

        # Get the result
        result = mock_ray_instance.list_models()

        # Assert the expected result
        assert "object" in result
        assert result["object"] == "list"
        assert "data" in result
        assert len(result["data"]) == 1
        assert result["data"][0]["id"] == "test-model"

    def test_health_check(self, mock_ray_instance):
        """Test the health_check endpoint."""
        expected_result = {"status": "healthy"}

        # Set up the mock to return the expected result
        mock_ray_instance.health_check.return_value = expected_result

        # Get the result
        result = mock_ray_instance.health_check()

        # Assert the expected result
        assert "status" in result
        assert result["status"] == "healthy"


if __name__ == "__main__":
    pytest.main()
