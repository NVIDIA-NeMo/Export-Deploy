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
import os
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import requests
from fastapi.testclient import TestClient

from nemo_deploy.service.fastapi_interface_to_pytriton import (
    ChatCompletionRequest,
    CompletionRequest,
    TritonSettings,
    _helper_fun,
    app,
    convert_numpy,
    dict_to_str,
    query_llm_async,
)


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def mock_triton_settings():
    with patch("nemo_deploy.service.fastapi_interface_to_pytriton.TritonSettings") as mock:
        instance = mock.return_value
        instance.triton_service_port = 8000
        instance.triton_service_ip = "localhost"
        yield instance

class TestTritonSettings:
    def test_default_values(self):
        with patch.dict(os.environ, {}, clear=True):
            settings = TritonSettings()
            assert settings.triton_service_port == 8000
            assert settings.triton_service_ip == "0.0.0.0"

    def test_custom_values(self):
        with patch.dict(
            os.environ,
            {"TRITON_PORT": "9000", "TRITON_HTTP_ADDRESS": "127.0.0.1"},
            clear=True,
        ):
            settings = TritonSettings()
            assert settings.triton_service_port == 9000
            assert settings.triton_service_ip == "127.0.0.1"

    def test_triton_settings_exception_handling(self):
        """Test TritonSettings initialization when environment variables cause exceptions"""
        with patch.dict(os.environ, {"TRITON_PORT": "invalid_port"}, clear=True):
            with patch("nemo.utils.logging.error") as mock_logging:
                settings = TritonSettings()

                # The attributes won't be set due to the early return, so accessing properties will fail
                with pytest.raises(AttributeError):
                    _ = settings.triton_service_port

                with pytest.raises(AttributeError):
                    _ = settings.triton_service_ip

                # Verify that the error was logged
                mock_logging.assert_called_once()
                # Check that the error message contains the expected content
                args, kwargs = mock_logging.call_args
                assert "An exception occurred trying to retrieve set args in TritonSettings class" in args[0]


class TestCompletionRequest:
    def test_default_completions_values(self):
        request = CompletionRequest(model="test_model", prompt="test prompt")
        assert request.model == "test_model"
        assert request.prompt == "test prompt"
        # assert request.messages == [{}]
        assert request.max_tokens == 512
        assert request.temperature == 1.0
        assert request.top_p == 0.0
        assert request.top_k == 0
        assert request.logprobs is None
        assert request.echo is False

    def test_default_chat_values(self):
        request = ChatCompletionRequest(model="test_model", messages=[{"role": "user", "content": "test message"}])
        assert request.model == "test_model"
        assert request.messages == [{"role": "user", "content": "test message"}]
        assert request.max_tokens == 512
        assert request.temperature == 1.0
        assert request.top_p == 0.0
        assert request.top_k == 0

    def test_greedy_params(self):
        request = CompletionRequest(model="test_model", prompt="test prompt", temperature=0.0, top_p=0.0)
        assert request.top_k == 1


class TestHealthEndpoints:
    def test_health_check(self, client):
        response = client.get("/v1/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}

    def test_triton_health_check_not_ready(self, client):
        """Test triton health check when server returns non-200 status"""
        with patch("requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 503
            mock_get.return_value = mock_response

            response = client.get("/v1/triton_health")
            assert response.status_code == 503
            assert "Triton server is not ready" in response.json()["detail"]

    def test_triton_health_check_request_exception(self, client):
        """Test triton health check when request fails"""
        with patch("requests.get", side_effect=requests.RequestException("Connection failed")):
            response = client.get("/v1/triton_health")
            assert response.status_code == 503
            assert "Cannot reach Triton server" in response.json()["detail"]


class TestUtilityFunctions:
    def test_convert_numpy(self):
        # Test with numpy array
        arr = np.array([1, 2, 3])
        assert convert_numpy(arr) == [1, 2, 3]

        # Test with nested dictionary
        nested = {"a": np.array([1, 2]), "b": {"c": np.array([3, 4])}}
        assert convert_numpy(nested) == {"a": [1, 2], "b": {"c": [3, 4]}}

        # Test with list
        lst = [np.array([1, 2]), np.array([3, 4])]
        assert convert_numpy(lst) == [[1, 2], [3, 4]]

    def test_dict_to_str(self):
        test_dict = {"key": "value", "number": 42}
        result = dict_to_str(test_dict)
        assert isinstance(result, str)
        assert json.loads(result) == test_dict


class TestLLMQueryFunctions:
    def test_helper_fun(self):
        mock_nq = MagicMock()
        mock_nq.query_llm.return_value = {"test": "response"}

        with patch(
            "nemo_deploy.service.fastapi_interface_to_pytriton.NemoQueryLLMPyTorch",
            return_value=mock_nq,
        ):
            result = _helper_fun(
                url="http://test",
                model="test_model",
                prompts=["test prompt"],
                temperature=0.7,
                top_k=10,
                top_p=0.9,
                compute_logprob=True,
                max_length=100,
                apply_chat_template=False,
                echo=False,
                n_top_logprobs=0,
            )
            assert result == {"test": "response"}
            mock_nq.query_llm.assert_called_once()

    def test_query_llm_async(self):
        mock_result = {"test": "response"}
        with patch(
            "nemo_deploy.service.fastapi_interface_to_pytriton._helper_fun",
            return_value=mock_result,
        ):
            # Create an event loop and run the async function
            import asyncio

            loop = asyncio.get_event_loop()
            result = loop.run_until_complete(
                query_llm_async(
                    url="http://test",
                    model="test_model",
                    prompts=["test prompt"],
                    temperature=0.7,
                    top_k=10,
                    top_p=0.9,
                    compute_logprob=True,
                    max_length=100,
                    apply_chat_template=False,
                    echo=False,
                    n_top_logprobs=0,
                )
            )
            assert result == mock_result


class TestAPIEndpoints:
    def test_completions_v1(self, client):
        mock_output = {
            "choices": [
                {
                    "text": [["test response"]],
                    "logprobs": {
                        "token_logprobs": [[1.0, 2.0]],
                        "top_logprobs": [[{"a": 0.5}, {"b": 0.5}]],
                    },
                }
            ]
        }

        with patch(
            "nemo_deploy.service.fastapi_interface_to_pytriton.query_llm_async",
            return_value=mock_output,
        ):
            response = client.post(
                "/v1/completions/",
                json={"model": "test_model", "prompt": "test prompt", "logprobs": 1},
            )
            assert response.status_code == 200
            data = response.json()
            assert data["choices"][0]["text"] == "test response"
            assert "logprobs" in data["choices"][0]

    def test_chat_completions_v1(self, client):
        mock_output = {"choices": [{"text": [["test response"]]}]}

        with patch(
            "nemo_deploy.service.fastapi_interface_to_pytriton.query_llm_async",
            return_value=mock_output,
        ):
            response = client.post(
                "/v1/chat/completions/",
                json={
                    "model": "test_model",
                    "messages": [{"role": "user", "content": "test message"}],
                },
            )
            assert response.status_code == 200
            data = response.json()
            assert data["choices"][0]["message"]["role"] == "assistant"
            assert data["choices"][0]["message"]["content"] == "test response"


    def test_triton_health_success(self, client):
        with patch("requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_get.return_value = mock_response

            response = client.get("/v1/triton_health")
            assert response.status_code == 200
            assert response.json() == {"status": "Triton server is reachable and ready"}
