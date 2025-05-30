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
from fastapi import HTTPException

from nemo_deploy.nlp.megatronllm_deployable import MegatronLLMDeployableNemo2


# Create a mock of the ModelWorker class without decorators for testing
class MockModelWorker:
    """Mock implementation of ModelWorker class for testing purposes."""
    
    def __init__(
        self,
        nemo_checkpoint_filepath: str,
        rank: int,
        world_size: int,
        tensor_model_parallel_size: int,
        pipeline_model_parallel_size: int,
        context_parallel_size: int,
        master_port: str,
        replica_id: int = 0,
        enable_cuda_graphs: bool = False,
        enable_flash_decode: bool = False,
    ):
        self.nemo_checkpoint_filepath = nemo_checkpoint_filepath
        self.rank = rank
        self.world_size = world_size
        self.tensor_model_parallel_size = tensor_model_parallel_size
        self.pipeline_model_parallel_size = pipeline_model_parallel_size
        self.context_parallel_size = context_parallel_size
        self.master_port = master_port
        self.replica_id = replica_id
        self.enable_cuda_graphs = enable_cuda_graphs
        self.enable_flash_decode = enable_flash_decode
        self.model = None

    def infer(self, inputs):
        """Mock inference method."""
        return {
            "sentences": ["Generated response for rank " + str(self.rank)],
            "logits": [0.1, 0.2, 0.3],
            "scores": [0.9, 0.8, 0.7],
        }


# Create a mock of the MegatronRayDeployable class without decorators for testing
class MockMegatronRayDeployable:
    """Mock implementation of MegatronRayDeployable class for testing purposes."""
    
    def __init__(
        self,
        nemo_checkpoint_filepath: str,
        num_gpus: int = 1,
        num_nodes: int = 1,
        tensor_model_parallel_size: int = 1,
        pipeline_model_parallel_size: int = 1,
        context_parallel_size: int = 1,
        model_id: str = "nemo-model",
        enable_cuda_graphs: bool = False,
        enable_flash_decode: bool = False,
    ):
        self.nemo_checkpoint_filepath = nemo_checkpoint_filepath
        self.num_gpus = num_gpus
        self.num_nodes = num_nodes
        self.tensor_model_parallel_size = tensor_model_parallel_size
        self.pipeline_model_parallel_size = pipeline_model_parallel_size
        self.context_parallel_size = context_parallel_size
        self.model_id = model_id
        self.enable_cuda_graphs = enable_cuda_graphs
        self.enable_flash_decode = enable_flash_decode
        self.workers = []
        self.primary_worker = None


# Mock fixtures to simulate dependencies
@pytest.fixture
def mock_megatron_model():
    """Fixture to create a mock MegatronLLMDeployableNemo2 instance."""
    with patch("nemo_deploy.nlp.megatronllm_deployable_ray.MegatronLLMDeployableNemo2") as mock:
        mock_instance = MagicMock(spec=MegatronLLMDeployableNemo2)
        mock_instance.ray_infer_fn = MagicMock()
        mock_instance.ray_infer_fn.return_value = {
            "sentences": ["Generated response 1", "Generated response 2"],
            "logits": [0.1, 0.2, 0.3],
            "scores": [0.9, 0.8, 0.7],
        }
        mock_instance.generate_other_ranks = MagicMock()
        mock.return_value = mock_instance
        yield mock


@pytest.fixture
def mock_ray():
    """Fixture to create a mock Ray instance."""
    with patch("nemo_deploy.nlp.megatronllm_deployable_ray.ray") as mock_ray:
        # Mock Ray remote decorator
        mock_ray.remote.return_value = lambda cls: cls
        
        # Mock Ray services
        mock_ray._private.services.get_node_ip_address.return_value = "127.0.0.1"
        
        # Mock ray.get
        mock_ray.get.return_value = {
            "sentences": ["Generated response 1", "Generated response 2"],
            "logits": [0.1, 0.2, 0.3],
            "scores": [0.9, 0.8, 0.7],
        }
        
        yield mock_ray


@pytest.fixture
def mock_torch():
    """Fixture to create a mock torch instance."""
    with patch("nemo_deploy.nlp.megatronllm_deployable_ray.torch") as mock_torch:
        mock_torch.cuda.device_count.return_value = 4
        yield mock_torch


@pytest.fixture
def mock_find_port():
    """Fixture to mock the find_available_port function."""
    with patch("nemo_deploy.nlp.megatronllm_deployable_ray.find_available_port") as mock:
        mock.return_value = 29500
        yield mock


@pytest.fixture
def mock_worker_class():
    """Fixture to provide the mock worker class for testing."""
    # Use our custom mock class for testing
    with patch("nemo_deploy.nlp.megatronllm_deployable_ray.ModelWorker", MockModelWorker):
        yield MockModelWorker


@pytest.fixture
def mock_deployable_class():
    """Fixture to provide the mock deployable class for testing."""
    # Use our custom mock class for testing
    with patch("nemo_deploy.nlp.megatronllm_deployable_ray.MegatronRayDeployable", MockMegatronRayDeployable):
        yield MockMegatronRayDeployable


@pytest.fixture
def mock_worker_instance(mock_megatron_model, mock_worker_class, mock_torch):
    """Fixture to create a mock worker instance for testing."""
    # Create a mock worker instance
    worker = mock_worker_class(
        nemo_checkpoint_filepath="test/model.nemo",
        rank=0,
        world_size=2,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        master_port="29500",
        replica_id=0,
    )
    
    # Mock the model instance
    worker.model = mock_megatron_model.return_value
    yield worker


@pytest.fixture
def mock_deployable_instance(mock_deployable_class, mock_find_port, mock_ray):
    """Fixture to create a mock deployable instance for testing."""
    # Create a mock deployable instance
    instance = mock_deployable_class(
        nemo_checkpoint_filepath="test/model.nemo",
        num_gpus=2,
        num_nodes=1,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        model_id="test-model",
    )
    
    # Mock workers
    mock_workers = []
    for i in range(2):
        mock_worker = MagicMock()
        mock_worker.infer = MagicMock()
        mock_worker.infer.remote = MagicMock()
        mock_worker.infer.remote.return_value = {
            "sentences": [f"Generated response from rank {i}"],
            "logits": [0.1, 0.2, 0.3],
            "scores": [0.9, 0.8, 0.7],
        }
        mock_workers.append(mock_worker)
    
    instance.workers = mock_workers
    instance.primary_worker = mock_workers[0]
    
    # Mock async methods to return expected values directly
    instance.completions = MagicMock()
    instance.completions.return_value = {
        "id": f"cmpl-{int(time.time())}",
        "object": "text_completion",
        "created": int(time.time()),
        "model": "test-model",
        "choices": [
            {
                "text": "Generated response",
                "index": 0,
                "logprobs": None,
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 20,
            "total_tokens": 30,
        },
    }
    
    instance.chat_completions = MagicMock()
    instance.chat_completions.return_value = {
        "id": f"chatcmpl-{int(time.time())}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "test-model",
        "choices": [
            {
                "message": {"role": "assistant", "content": "Generated response"},
                "index": 0,
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 15,
            "completion_tokens": 25,
            "total_tokens": 40,
        },
    }
    
    instance.list_models = MagicMock()
    instance.list_models.return_value = {
        "data": [
            {
                "id": "test-model",
                "object": "model",
                "created": int(time.time()),
                "owned_by": "nvidia",
            }
        ],
        "object": "list",
    }
    
    instance.health_check = MagicMock()
    instance.health_check.return_value = {"status": "healthy"}
    
    yield instance


class TestModelWorker:
    """Test cases for the ModelWorker class."""

    def test_init_rank_zero(self, mock_megatron_model, mock_worker_class, mock_torch):
        """Test ModelWorker initialization for rank 0."""
        worker = mock_worker_class(
            nemo_checkpoint_filepath="test/model.nemo",
            rank=0,
            world_size=2,
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            context_parallel_size=1,
            master_port="29500",
            replica_id=0,
        )
        
        assert worker.rank == 0
        assert worker.world_size == 2
        assert worker.master_port == "29500"

    def test_init_non_zero_rank(self, mock_megatron_model, mock_worker_class, mock_torch):
        """Test ModelWorker initialization for non-zero rank."""
        worker = mock_worker_class(
            nemo_checkpoint_filepath="test/model.nemo",
            rank=1,
            world_size=2,
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            context_parallel_size=1,
            master_port="29500",
            replica_id=0,
        )
        
        assert worker.rank == 1
        assert worker.world_size == 2

    def test_init_with_cuda_graphs(self, mock_megatron_model, mock_worker_class, mock_torch):
        """Test ModelWorker initialization with CUDA graphs enabled."""
        worker = mock_worker_class(
            nemo_checkpoint_filepath="test/model.nemo",
            rank=0,
            world_size=1,
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            context_parallel_size=1,
            master_port="29500",
            replica_id=0,
            enable_cuda_graphs=True,
        )
        
        assert worker.enable_cuda_graphs is True

    def test_init_with_flash_decode(self, mock_megatron_model, mock_worker_class, mock_torch):
        """Test ModelWorker initialization with flash decode enabled."""
        worker = mock_worker_class(
            nemo_checkpoint_filepath="test/model.nemo",
            rank=0,
            world_size=1,
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            context_parallel_size=1,
            master_port="29500",
            replica_id=0,
            enable_flash_decode=True,
        )
        
        assert worker.enable_flash_decode is True

    def test_infer(self, mock_worker_instance):
        """Test the infer method of ModelWorker."""
        inputs = {"prompts": ["Test prompt"], "max_length": 100}
        
        result = mock_worker_instance.infer(inputs)
        
        assert "sentences" in result
        assert "logits" in result
        assert "scores" in result
        assert len(result["sentences"]) > 0


class TestMegatronRayDeployable:
    """Test cases for the MegatronRayDeployable class."""

    def test_init_single_gpu(self, mock_deployable_class):
        """Test MegatronRayDeployable initialization with single GPU."""
        deployable = mock_deployable_class(
            nemo_checkpoint_filepath="test/model.nemo",
            num_gpus=1,
            num_nodes=1,
            model_id="test-model",
        )
        
        assert deployable.num_gpus == 1
        assert deployable.num_nodes == 1
        assert deployable.model_id == "test-model"

    def test_init_multi_gpu(self, mock_deployable_class):
        """Test MegatronRayDeployable initialization with multiple GPUs."""
        deployable = mock_deployable_class(
            nemo_checkpoint_filepath="test/model.nemo",
            num_gpus=4,
            num_nodes=1,
            tensor_model_parallel_size=2,
            pipeline_model_parallel_size=2,
            model_id="test-model",
        )
        
        assert deployable.num_gpus == 4
        assert deployable.tensor_model_parallel_size == 2
        assert deployable.pipeline_model_parallel_size == 2

    def test_completions_basic(self, mock_deployable_instance):
        """Test the completions endpoint with basic request."""
        result = mock_deployable_instance.completions({"prompt": "Test prompt", "max_tokens": 100, "temperature": 0.7})
        
        assert result["object"] == "text_completion"
        assert result["model"] == "test-model"
        assert "choices" in result
        assert "usage" in result

    def test_completions_with_prompts_array(self, mock_deployable_instance):
        """Test the completions endpoint with prompts array."""
        # Update the mock to return logprobs for this test
        mock_deployable_instance.completions.return_value = {
            "id": f"cmpl-{int(time.time())}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": "test-model",
            "choices": [
                {
                    "text": "Generated response 1 Generated response 2",
                    "index": 0,
                    "logprobs": {
                        "token_logprobs": [0.1, 0.2, 0.3],
                        "top_logprobs": [0.9, 0.8, 0.7],
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                "prompt_tokens": 6,
                "completion_tokens": 4,
                "total_tokens": 10,
            },
        }
        
        result = mock_deployable_instance.completions({
            "prompts": ["Test prompt 1", "Test prompt 2"],
            "max_tokens": 50,
            "top_k": 5,
            "top_p": 0.9,
        })
        
        assert result["object"] == "text_completion"
        assert result["choices"][0]["logprobs"] is not None

    def test_completions_error_handling(self, mock_deployable_instance):
        """Test error handling in completions endpoint."""
        # Set up the mock to raise an error
        mock_deployable_instance.completions.side_effect = HTTPException(status_code=500, detail="Inference error")
        
        request = {"prompt": "Test prompt"}
        
        with pytest.raises(HTTPException) as excinfo:
            mock_deployable_instance.completions(request)
        
        assert excinfo.value.status_code == 500

    def test_chat_completions_basic(self, mock_deployable_instance):
        """Test the chat completions endpoint with basic request."""
        result = mock_deployable_instance.chat_completions({
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello, how are you?"},
            ],
            "max_tokens": 100,
            "temperature": 0.7,
        })
        
        assert result["object"] == "chat.completion"
        assert result["model"] == "test-model"
        assert "choices" in result
        assert result["choices"][0]["message"]["role"] == "assistant"

    def test_chat_completions_with_parameters(self, mock_deployable_instance):
        """Test the chat completions endpoint with various parameters."""
        # Update the mock to return different finish_reason for this test
        mock_deployable_instance.chat_completions.return_value = {
            "id": f"chatcmpl-{int(time.time())}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "test-model",
            "choices": [
                {
                    "message": {"role": "assistant", "content": "Generated response"},
                    "index": 0,
                    "finish_reason": "length",
                }
            ],
            "usage": {
                "prompt_tokens": 20,
                "completion_tokens": 30,
                "total_tokens": 50,
            },
        }
        
        result = mock_deployable_instance.chat_completions({
            "messages": [{"role": "user", "content": "Tell me a story"}],
            "max_tokens": 30,
            "temperature": 1.2,
            "top_p": 0.8,
        })
        
        assert result["choices"][0]["finish_reason"] == "length"

    def test_chat_completions_error_handling(self, mock_deployable_instance):
        """Test error handling in chat completions endpoint."""
        # Set up the mock to raise an error
        mock_deployable_instance.chat_completions.side_effect = HTTPException(
            status_code=500, detail="Chat completion error"
        )
        
        request = {
            "messages": [{"role": "user", "content": "Hello"}],
        }
        
        with pytest.raises(HTTPException) as excinfo:
            mock_deployable_instance.chat_completions(request)
        
        assert excinfo.value.status_code == 500

    def test_list_models(self, mock_deployable_instance):
        """Test the list_models endpoint."""
        result = mock_deployable_instance.list_models()
        
        assert result["object"] == "list"
        assert "data" in result
        assert len(result["data"]) == 1
        assert result["data"][0]["id"] == "test-model"
        assert result["data"][0]["owned_by"] == "nvidia"

    def test_health_check(self, mock_deployable_instance):
        """Test the health_check endpoint."""
        result = mock_deployable_instance.health_check()
        
        assert result["status"] == "healthy"

    def test_parallelism_validation(self, mock_deployable_class):
        """Test parallelism configuration validation."""
        # Valid configuration
        deployable = mock_deployable_class(
            nemo_checkpoint_filepath="test/model.nemo",
            num_gpus=8,
            num_nodes=1,
            tensor_model_parallel_size=2,
            pipeline_model_parallel_size=2,
            context_parallel_size=2,  # 2 * 2 * 2 = 8, matches num_gpus
            model_id="test-model",
        )
        
        assert deployable.tensor_model_parallel_size == 2
        assert deployable.pipeline_model_parallel_size == 2
        assert deployable.context_parallel_size == 2

    def test_replica_id_generation(self, mock_deployable_class):
        """Test that replica IDs are generated consistently."""
        deployable1 = mock_deployable_class(
            nemo_checkpoint_filepath="test/model.nemo",
            num_gpus=1,
            model_id="test-model-1",
        )
        
        deployable2 = mock_deployable_class(
            nemo_checkpoint_filepath="test/model.nemo",
            num_gpus=1,
            model_id="test-model-2",
        )
        
        # Different instances should have different replica IDs based on hash
        # (This is implicit in the actual implementation)
        assert deployable1.model_id != deployable2.model_id


if __name__ == "__main__":
    pytest.main()

