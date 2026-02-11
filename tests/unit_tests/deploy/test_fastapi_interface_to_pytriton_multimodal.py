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

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient

from nemo_deploy.service.fastapi_interface_to_pytriton_multimodal import (
    BaseMultimodalRequest,
    ImageContent,
    MultimodalChatCompletionRequest,
    MultimodalCompletionRequest,
    TextContent,
    TritonSettings,
    app,
    convert_numpy,
    dict_to_str,
)


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def mock_triton_settings():
    """Mock TritonSettings to avoid environment variable dependencies."""
    with patch("nemo_deploy.service.fastapi_interface_to_pytriton_multimodal.triton_settings") as mock_settings:
        mock_settings.triton_service_ip = "127.0.0.1"
        mock_settings.triton_service_port = 8000
        yield mock_settings


class TestTritonSettings:
    def test_initialization_with_defaults(self):
        """Test TritonSettings initialization with default environment variables."""
        with patch.dict("os.environ", {}, clear=True):
            settings = TritonSettings()
            assert settings.triton_service_port == 8000
            assert settings.triton_service_ip == "0.0.0.0"

    def test_initialization_with_env_vars(self):
        """Test TritonSettings initialization with custom environment variables."""
        with patch.dict("os.environ", {"TRITON_PORT": "9000", "TRITON_HTTP_ADDRESS": "192.168.1.1"}):
            settings = TritonSettings()
            assert settings.triton_service_port == 9000
            assert settings.triton_service_ip == "192.168.1.1"

    def test_initialization_with_invalid_port(self):
        """Test TritonSettings initialization with invalid port logs error but doesn't raise."""
        with patch.dict("os.environ", {"TRITON_PORT": "invalid"}):
            with patch("nemo_deploy.service.fastapi_interface_to_pytriton_multimodal.logger") as mock_logger:
                TritonSettings()
                # The exception is caught and logged, not raised
                mock_logger.error.assert_called_once()
                # Verify error was called with the expected message prefix
                call_args = mock_logger.error.call_args[0]
                assert "An exception occurred trying to retrieve set args in TritonSettings class" in call_args[0]

    def test_properties(self):
        """Test TritonSettings properties."""
        with patch.dict("os.environ", {"TRITON_PORT": "8888", "TRITON_HTTP_ADDRESS": "localhost"}):
            settings = TritonSettings()
            assert settings.triton_service_port == 8888
            assert settings.triton_service_ip == "localhost"


class TestPydanticModels:
    def test_base_multimodal_request_defaults(self):
        """Test BaseMultimodalRequest with default values."""
        request = BaseMultimodalRequest(model="test-model")
        assert request.model == "test-model"
        assert request.max_tokens == 50
        assert request.temperature == 1.0
        assert request.top_p == 0.0
        assert request.top_k == 0
        assert request.random_seed is None
        assert request.max_batch_size == 4

    def test_base_multimodal_request_custom_values(self):
        """Test BaseMultimodalRequest with custom values."""
        request = BaseMultimodalRequest(
            model="custom-model",
            max_tokens=100,
            temperature=0.7,
            top_p=0.9,
            top_k=50,
            random_seed=42,
            max_batch_size=8,
        )
        assert request.model == "custom-model"
        assert request.max_tokens == 100
        assert request.temperature == 0.7
        assert request.top_p == 0.9
        assert request.top_k == 50
        assert request.random_seed == 42
        assert request.max_batch_size == 8

    def test_multimodal_completion_request(self):
        """Test MultimodalCompletionRequest."""
        request = MultimodalCompletionRequest(
            model="test-model", prompt="Test prompt", image="base64_image_data", apply_chat_template=True
        )
        assert request.model == "test-model"
        assert request.prompt == "Test prompt"
        assert request.image == "base64_image_data"
        assert request.apply_chat_template is True

    def test_multimodal_completion_request_without_image(self):
        """Test MultimodalCompletionRequest without image."""
        request = MultimodalCompletionRequest(model="test-model", prompt="Test prompt")
        assert request.image is None
        assert request.apply_chat_template is False

    def test_image_content(self):
        """Test ImageContent model."""
        content = ImageContent(image_url={"url": "http://example.com/image.jpg"})
        assert content.type == "image_url"
        assert content.image_url == {"url": "http://example.com/image.jpg"}

    def test_text_content(self):
        """Test TextContent model."""
        content = TextContent(text="Hello, world!")
        assert content.type == "text"
        assert content.text == "Hello, world!"

    def test_multimodal_chat_completion_request(self):
        """Test MultimodalChatCompletionRequest."""
        messages = [
            {"role": "user", "content": [{"type": "text", "text": "Hello"}, {"type": "image", "image": "base64_data"}]}
        ]
        request = MultimodalChatCompletionRequest(model="test-model", messages=messages)
        assert request.model == "test-model"
        assert request.messages == messages


class TestHealthEndpoints:
    def test_health_check(self, client):
        """Test /v1/health endpoint."""
        response = client.get("/v1/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_triton_health_success(self, client, mock_triton_settings):
        """Test /v1/triton_health endpoint with successful connection."""
        with patch("requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_get.return_value = mock_response

            response = client.get("/v1/triton_health")
            assert response.status_code == 200
            assert response.json() == {"status": "Triton server is reachable and ready"}

    def test_triton_health_not_ready(self, client, mock_triton_settings):
        """Test /v1/triton_health endpoint when Triton is not ready."""
        with patch("requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 503
            mock_get.return_value = mock_response

            response = client.get("/v1/triton_health")
            assert response.status_code == 503
            assert "not ready" in response.json()["detail"]

    def test_triton_health_unreachable(self, client, mock_triton_settings):
        """Test /v1/triton_health endpoint when Triton is unreachable."""
        with patch("requests.get") as mock_get:
            import requests

            mock_get.side_effect = requests.RequestException("Connection error")

            response = client.get("/v1/triton_health")
            assert response.status_code == 503
            assert "Cannot reach Triton server" in response.json()["detail"]


class TestHelperFunctions:
    def test_convert_numpy_array(self):
        """Test convert_numpy with numpy array."""
        arr = np.array([1, 2, 3])
        result = convert_numpy(arr)
        assert result == [1, 2, 3]
        assert isinstance(result, list)

    def test_convert_numpy_dict(self):
        """Test convert_numpy with dictionary containing numpy arrays."""
        data = {"array": np.array([1, 2, 3]), "value": 42}
        result = convert_numpy(data)
        assert result == {"array": [1, 2, 3], "value": 42}

    def test_convert_numpy_nested_dict(self):
        """Test convert_numpy with nested dictionary."""
        data = {"outer": {"inner": np.array([1, 2]), "num": 5}, "list": [np.array([3, 4])]}
        result = convert_numpy(data)
        assert result == {"outer": {"inner": [1, 2], "num": 5}, "list": [[3, 4]]}

    def test_convert_numpy_list(self):
        """Test convert_numpy with list containing numpy arrays."""
        data = [np.array([1, 2]), np.array([3, 4]), 5]
        result = convert_numpy(data)
        assert result == [[1, 2], [3, 4], 5]

    def test_convert_numpy_no_conversion(self):
        """Test convert_numpy with non-numpy data."""
        data = {"string": "text", "number": 42, "list": [1, 2, 3]}
        result = convert_numpy(data)
        assert result == data

    def test_dict_to_str(self):
        """Test dict_to_str function."""
        test_dict = {"key1": "value1", "key2": [1, 2, 3], "key3": {"nested": "dict"}}
        result = dict_to_str(test_dict)
        assert isinstance(result, str)
        assert json.loads(result) == test_dict

    def test_dict_to_str_empty(self):
        """Test dict_to_str with empty dictionary."""
        result = dict_to_str({})
        assert result == "{}"


class TestCompletionsEndpoint:
    def test_completions_single_prompt(self, client, mock_triton_settings):
        """Test /v1/completions/ endpoint with single prompt."""
        request_data = {"model": "test-model", "prompt": "Test prompt", "max_tokens": 50}

        mock_output = {
            "choices": [{"text": [["Generated response"]]}],
            "model": "test-model",
            "usage": {"prompt_tokens": 10, "completion_tokens": 20},
        }

        with patch("nemo_deploy.service.fastapi_interface_to_pytriton_multimodal.query_multimodal_async") as mock_query:
            mock_query.return_value = mock_output

            response = client.post("/v1/completions/", json=request_data)

            assert response.status_code == 200
            result = response.json()
            assert result["choices"][0]["text"] == "Generated response"
            assert result["model"] == "test-model"

            mock_query.assert_called_once()
            call_kwargs = mock_query.call_args[1]
            assert call_kwargs["prompts"] == ["Test prompt"]
            assert call_kwargs["images"] == []
            assert call_kwargs["max_length"] == 50
            assert call_kwargs["apply_chat_template"] is False

    def test_completions_with_image(self, client, mock_triton_settings):
        """Test /v1/completions/ endpoint with image."""
        request_data = {
            "model": "test-model",
            "prompt": "Describe this image",
            "image": "data:image;base64,base64_encoded_image_data",
            "temperature": 0.7,
        }

        mock_output = {"choices": [{"text": [["This is an image description"]]}], "model": "test-model"}

        with patch("nemo_deploy.service.fastapi_interface_to_pytriton_multimodal.query_multimodal_async") as mock_query:
            mock_query.return_value = mock_output

            response = client.post("/v1/completions/", json=request_data)

            assert response.status_code == 200
            result = response.json()
            assert result["choices"][0]["text"] == "This is an image description"

            mock_query.assert_called_once()
            call_kwargs = mock_query.call_args[1]
            assert call_kwargs["images"] == ["data:image;base64,base64_encoded_image_data"]
            assert call_kwargs["temperature"] == 0.7

    def test_completions_with_custom_params(self, client, mock_triton_settings):
        """Test /v1/completions/ endpoint with custom parameters."""
        request_data = {
            "model": "test-model",
            "prompt": "Test prompt",
            "max_tokens": 100,
            "temperature": 0.8,
            "top_k": 50,
            "top_p": 0.95,
            "random_seed": 42,
            "max_batch_size": 2,
        }

        mock_output = {"choices": [{"text": [["Generated text"]]}], "model": "test-model"}

        with patch("nemo_deploy.service.fastapi_interface_to_pytriton_multimodal.query_multimodal_async") as mock_query:
            mock_query.return_value = mock_output

            response = client.post("/v1/completions/", json=request_data)

            assert response.status_code == 200

            mock_query.assert_called_once()
            call_kwargs = mock_query.call_args[1]
            assert call_kwargs["max_length"] == 100
            assert call_kwargs["temperature"] == 0.8
            assert call_kwargs["top_k"] == 50
            assert call_kwargs["top_p"] == 0.95
            assert call_kwargs["random_seed"] == 42
            assert call_kwargs["max_batch_size"] == 2


class TestChatCompletionsEndpoint:
    def test_chat_completions_basic(self, client, mock_triton_settings):
        """Test /v1/chat/completions/ endpoint with basic message."""
        messages = [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}]
        request_data = {"model": "test-model", "messages": messages}

        mock_output = {"choices": [{"text": [["Hello! How can I help you?"]]}], "model": "test-model"}

        with patch("nemo_deploy.service.fastapi_interface_to_pytriton_multimodal.query_multimodal_async") as mock_query:
            mock_query.return_value = mock_output

            response = client.post("/v1/chat/completions/", json=request_data)

            assert response.status_code == 200
            result = response.json()
            assert result["object"] == "chat.completion"
            assert result["choices"][0]["message"]["role"] == "assistant"
            assert result["choices"][0]["message"]["content"] == "Hello! How can I help you?"
            assert "text" not in result["choices"][0]

            mock_query.assert_called_once()
            call_kwargs = mock_query.call_args[1]
            assert call_kwargs["apply_chat_template"] is True

    def test_chat_completions_with_image(self, client, mock_triton_settings):
        """Test /v1/chat/completions/ endpoint with image."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {"type": "image", "image": "data:image;base64,base64_image_data"},
                ],
            }
        ]
        request_data = {"model": "test-model", "messages": messages}

        mock_output = {"choices": [{"text": [["I see a cat"]]}], "model": "test-model"}

        with patch("nemo_deploy.service.fastapi_interface_to_pytriton_multimodal.query_multimodal_async") as mock_query:
            mock_query.return_value = mock_output

            response = client.post("/v1/chat/completions/", json=request_data)

            assert response.status_code == 200
            result = response.json()
            assert result["choices"][0]["message"]["content"] == "I see a cat"

            mock_query.assert_called_once()
            call_kwargs = mock_query.call_args[1]
            assert call_kwargs["images"] == ["data:image;base64,base64_image_data"]

    def test_chat_completions_multiple_images(self, client, mock_triton_settings):
        """Test /v1/chat/completions/ endpoint with multiple images."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Compare these images"},
                    {"type": "image", "image": "data:image;base64,base64_image_1"},
                    {"type": "image", "image": "data:image;base64,base64_image_2"},
                ],
            }
        ]
        request_data = {"model": "test-model", "messages": messages, "max_tokens": 200}

        mock_output = {"choices": [{"text": [["Comparison result"]]}], "model": "test-model"}

        with patch("nemo_deploy.service.fastapi_interface_to_pytriton_multimodal.query_multimodal_async") as mock_query:
            mock_query.return_value = mock_output

            response = client.post("/v1/chat/completions/", json=request_data)

            assert response.status_code == 200

            mock_query.assert_called_once()
            call_kwargs = mock_query.call_args[1]
            assert call_kwargs["images"] == ["data:image;base64,base64_image_1", "data:image;base64,base64_image_2"]
            assert call_kwargs["max_length"] == 200

    def test_chat_completions_with_image_url_format(self, client, mock_triton_settings):
        """Test /v1/chat/completions/ endpoint with OpenAI-style image_url format."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "What's in this image?"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}},
                ],
            }
        ]
        request_data = {"model": "test-model", "messages": messages}

        mock_output = {"choices": [{"text": [["I see a cat"]]}], "model": "test-model"}

        with patch("nemo_deploy.service.fastapi_interface_to_pytriton_multimodal.query_multimodal_async") as mock_query:
            mock_query.return_value = mock_output

            response = client.post("/v1/chat/completions/", json=request_data)

            assert response.status_code == 200
            result = response.json()
            assert result["choices"][0]["message"]["content"] == "I see a cat"

            mock_query.assert_called_once()
            call_kwargs = mock_query.call_args[1]
            assert call_kwargs["images"] == ["https://example.com/image.jpg"]

    def test_chat_completions_with_mixed_image_formats(self, client, mock_triton_settings):
        """Test /v1/chat/completions/ endpoint with mixed image and image_url formats."""
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Compare these images"},
                    {"type": "image", "image": "data:image;base64,base64_data"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/image.jpg"}},
                ],
            }
        ]
        request_data = {"model": "test-model", "messages": messages}

        mock_output = {"choices": [{"text": [["Comparison"]]}], "model": "test-model"}

        with patch("nemo_deploy.service.fastapi_interface_to_pytriton_multimodal.query_multimodal_async") as mock_query:
            mock_query.return_value = mock_output

            response = client.post("/v1/chat/completions/", json=request_data)

            assert response.status_code == 200

            mock_query.assert_called_once()
            call_kwargs = mock_query.call_args[1]
            assert call_kwargs["images"] == ["data:image;base64,base64_data", "https://example.com/image.jpg"]

    def test_chat_completions_with_params(self, client, mock_triton_settings):
        """Test /v1/chat/completions/ endpoint with custom parameters."""
        messages = [{"role": "user", "content": [{"type": "text", "text": "Hello"}]}]
        request_data = {
            "model": "test-model",
            "messages": messages,
            "max_tokens": 150,
            "temperature": 0.9,
            "top_k": 40,
            "top_p": 0.85,
            "random_seed": 123,
        }

        mock_output = {"choices": [{"text": [["Response"]]}], "model": "test-model"}

        with patch("nemo_deploy.service.fastapi_interface_to_pytriton_multimodal.query_multimodal_async") as mock_query:
            mock_query.return_value = mock_output

            response = client.post("/v1/chat/completions/", json=request_data)

            assert response.status_code == 200

            mock_query.assert_called_once()
            call_kwargs = mock_query.call_args[1]
            assert call_kwargs["temperature"] == 0.9
            assert call_kwargs["top_k"] == 40
            assert call_kwargs["top_p"] == 0.85
            assert call_kwargs["random_seed"] == 123


class TestQueryMultimodalAsync:
    def test_helper_fun(self):
        """Test _helper_fun function."""
        from nemo_deploy.service.fastapi_interface_to_pytriton_multimodal import _helper_fun

        with patch(
            "nemo_deploy.service.fastapi_interface_to_pytriton_multimodal.NemoQueryMultimodalPytorch"
        ) as mock_nq_class:
            mock_nq = MagicMock()
            mock_nq.query_multimodal.return_value = {"result": "success"}
            mock_nq_class.return_value = mock_nq

            result = _helper_fun(
                url="http://localhost:8000",
                model="test-model",
                prompts=["test prompt"],
                images=["data:image;base64,image_data"],
                temperature=0.7,
                top_k=10,
                top_p=0.9,
                max_length=100,
                random_seed=42,
                max_batch_size=4,
                apply_chat_template=False,
            )

            mock_nq_class.assert_called_once_with(url="http://localhost:8000", model_name="test-model")
            mock_nq.query_multimodal.assert_called_once_with(
                prompts=["test prompt"],
                images=["data:image;base64,image_data"],
                temperature=0.7,
                top_k=10,
                top_p=0.9,
                max_length=100,
                random_seed=42,
                max_batch_size=4,
                apply_chat_template=False,
                init_timeout=300,
            )
            assert result == {"result": "success"}

    def test_query_multimodal_async(self):
        """Test query_multimodal_async function."""
        import asyncio

        from nemo_deploy.service.fastapi_interface_to_pytriton_multimodal import query_multimodal_async

        with patch("nemo_deploy.service.fastapi_interface_to_pytriton_multimodal._helper_fun") as mock_helper:
            mock_helper.return_value = {"result": "async success"}

            # Run the async function using asyncio.run
            result = asyncio.run(
                query_multimodal_async(
                    url="http://localhost:8000",
                    model="test-model",
                    prompts=["test"],
                    images=[],
                    temperature=1.0,
                    top_k=1,
                    top_p=0.0,
                    max_length=50,
                    random_seed=None,
                    max_batch_size=4,
                    apply_chat_template=False,
                )
            )

            assert result == {"result": "async success"}


class TestEdgeCases:
    def test_completions_missing_model(self, client):
        """Test /v1/completions/ with missing model field."""
        request_data = {"prompt": "Test prompt"}
        response = client.post("/v1/completions/", json=request_data)
        assert response.status_code == 422

    def test_completions_missing_prompt(self, client):
        """Test /v1/completions/ with missing prompt field."""
        request_data = {"model": "test-model"}
        response = client.post("/v1/completions/", json=request_data)
        assert response.status_code == 422

    def test_chat_completions_missing_messages(self, client):
        """Test /v1/chat/completions/ with missing messages field."""
        request_data = {"model": "test-model"}
        response = client.post("/v1/chat/completions/", json=request_data)
        assert response.status_code == 422

    def test_chat_completions_invalid_messages(self, client):
        """Test /v1/chat/completions/ with invalid messages format."""
        request_data = {"model": "test-model", "messages": "not a list"}
        response = client.post("/v1/chat/completions/", json=request_data)
        assert response.status_code == 422

    def test_completions_invalid_temperature(self, client):
        """Test /v1/completions/ with invalid temperature."""
        request_data = {"model": "test-model", "prompt": "Test", "temperature": "invalid"}
        response = client.post("/v1/completions/", json=request_data)
        assert response.status_code == 422

    def test_convert_numpy_multidimensional(self):
        """Test convert_numpy with multidimensional arrays."""
        arr = np.array([[1, 2], [3, 4]])
        result = convert_numpy(arr)
        assert result == [[1, 2], [3, 4]]

    def test_convert_numpy_complex_structure(self):
        """Test convert_numpy with complex nested structure."""
        data = {
            "level1": {
                "level2": [np.array([1, 2]), {"level3": np.array([[3, 4], [5, 6]])}],
                "simple": "text",
            },
            "array": np.array([7, 8, 9]),
        }
        result = convert_numpy(data)
        expected = {
            "level1": {"level2": [[1, 2], {"level3": [[3, 4], [5, 6]]}], "simple": "text"},
            "array": [7, 8, 9],
        }
        assert result == expected
