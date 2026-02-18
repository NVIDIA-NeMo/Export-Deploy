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

"""Tests for OpenAI API format compatibility in MegatronLLM Ray deployment."""

import asyncio
import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest


@pytest.fixture
def mock_ray_deployment():
    """Fixture to create a mock Ray deployment instance."""
    # Create a mock deployment that mimics MegatronRayDeployable's interface
    deployment = MagicMock()
    deployment.model_id = "megatron-model"
    deployment.workers = [MagicMock()]
    deployment.primary_worker = deployment.workers[0]

    # Mock the completions method to behave like the actual implementation
    async def mock_completions(request):
        # Import here to get access to the actual completions logic we want to test
        from nemo_deploy.llm.megatronllm_deployable_ray import MegatronRayDeployable

        # Get the actual class from the deployment decorator
        actual_class = MegatronRayDeployable.func_or_class
        # Call the actual completions method with self=deployment
        return await actual_class.completions(deployment, request)

    async def mock_chat_completions(request):
        from nemo_deploy.llm.megatronllm_deployable_ray import MegatronRayDeployable

        actual_class = MegatronRayDeployable.func_or_class
        return await actual_class.chat_completions(deployment, request)

    deployment.completions = mock_completions
    deployment.chat_completions = mock_chat_completions
    yield deployment


@pytest.mark.run_only_on("GPU")
def test_completions_output_format_basic(mock_ray_deployment):
    """Test that the completions endpoint returns OpenAI API compatible format."""
    request = {
        "prompt": "Question: An astronomer observes",
        "max_tokens": 40,
        "temperature": 1.0,
        "top_p": 0.0,
        "logprobs": 5,
    }

    # Mock the Ray worker response
    mock_results = {
        "sentences": ["Generated text response"],
        "log_probs": np.array([[-0.5, -1.0, -0.8]]),
        "top_logprobs": [
            json.dumps(
                [
                    {"token1": -0.5, "token2": -1.5},
                    {"token3": -1.0, "token4": -2.0},
                    {"token5": -0.8, "token6": -1.8},
                ]
            )
        ],
    }

    with patch("ray.get", return_value=mock_results):
        output = asyncio.run(mock_ray_deployment.completions(request))

        # Verify top-level structure
        assert isinstance(output, dict)
        assert "id" in output
        assert output["object"] == "text_completion"
        assert "created" in output
        assert output["model"] == "megatron-model"
        assert "choices" in output
        assert "usage" in output

        # Verify choices structure
        assert isinstance(output["choices"], list)
        assert len(output["choices"]) == 1
        choice = output["choices"][0]

        # Test 1: Verify 'text' is a string
        assert "text" in choice
        assert isinstance(choice["text"], str), f"Expected 'text' to be str, got {type(choice['text'])}"

        # Test 2: Verify 'logprobs' is a dictionary
        assert "logprobs" in choice
        assert isinstance(choice["logprobs"], dict), f"Expected 'logprobs' to be dict, got {type(choice['logprobs'])}"

        # Test 3: Verify 'token_logprobs' is a list
        assert "token_logprobs" in choice["logprobs"]
        assert isinstance(choice["logprobs"]["token_logprobs"], list), (
            f"Expected 'token_logprobs' to be list, got {type(choice['logprobs']['token_logprobs'])}"
        )

        # Test 4: Verify 'top_logprobs' is a list of dictionaries
        assert "top_logprobs" in choice["logprobs"]
        assert isinstance(choice["logprobs"]["top_logprobs"], list), (
            f"Expected 'top_logprobs' to be list, got {type(choice['logprobs']['top_logprobs'])}"
        )

        # Verify each element in top_logprobs is a dictionary
        for i, item in enumerate(choice["logprobs"]["top_logprobs"]):
            assert isinstance(item, dict), f"Expected top_logprobs[{i}] to be dict, got {type(item)}"


@pytest.mark.run_only_on("GPU")
def test_completions_text_field_is_string(mock_ray_deployment):
    """Test that choices[0]['text'] is always a string."""
    request = {
        "prompt": "Hello world",
        "max_tokens": 10,
        "temperature": 0.0,
    }

    mock_results = {
        "sentences": ["This", "is", "a", "test"],
    }

    with patch("ray.get", return_value=mock_results):
        output = asyncio.run(mock_ray_deployment.completions(request))

        text = output["choices"][0]["text"]
        assert isinstance(text, str), f"Expected string, got {type(text)}"
        # When multiple sentences are returned, they should be joined with spaces
        assert text == "This is a test"


@pytest.mark.run_only_on("GPU")
def test_completions_logprobs_structure(mock_ray_deployment):
    """Test detailed structure of logprobs field."""
    request = {
        "prompt": "Test",
        "max_tokens": 5,
        "logprobs": 3,
    }

    mock_results = {
        "sentences": ["Generated"],
        "log_probs": np.array([[-0.1, -0.2, -0.3]]),
        "top_logprobs": [json.dumps([{"a": -0.1, "b": -0.5}, {"c": -0.2, "d": -0.6}, {"e": -0.3, "f": -0.7}])],
    }

    with patch("ray.get", return_value=mock_results):
        output = asyncio.run(mock_ray_deployment.completions(request))

        logprobs = output["choices"][0]["logprobs"]

        # Verify logprobs is a dictionary
        assert isinstance(logprobs, dict)

        # Verify token_logprobs is a list
        assert "token_logprobs" in logprobs
        token_logprobs = logprobs["token_logprobs"]
        assert isinstance(token_logprobs, list)
        assert len(token_logprobs) == 3
        for prob in token_logprobs:
            assert isinstance(prob, (int, float)) or prob is None

        # Verify top_logprobs is a list of dictionaries
        assert "top_logprobs" in logprobs
        top_logprobs = logprobs["top_logprobs"]
        assert isinstance(top_logprobs, list)
        assert len(top_logprobs) == 3

        for item in top_logprobs:
            assert isinstance(item, dict)
            # Each dictionary should have string keys and numeric values
            for key, value in item.items():
                assert isinstance(key, str)
                assert isinstance(value, (int, float))


@pytest.mark.run_only_on("GPU")
def test_completions_with_echo_adds_none_to_token_logprobs(mock_ray_deployment):
    """Test that when echo=True, the first element in token_logprobs is None."""
    request = {
        "prompt": "Test",
        "max_tokens": 5,
        "logprobs": 2,
        "echo": True,
    }

    mock_results = {
        "sentences": ["Generated text"],
        "log_probs": np.array([[-0.1, -0.2]]),
        "top_logprobs": [json.dumps([{"a": -0.1}, {"b": -0.2}])],
    }

    with patch("ray.get", return_value=mock_results):
        output = asyncio.run(mock_ray_deployment.completions(request))

        token_logprobs = output["choices"][0]["logprobs"]["token_logprobs"]

        # First element should be None when echo=True
        assert token_logprobs[0] is None
        # Rest should be floats
        assert len(token_logprobs) == 3  # None + 2 original values


@pytest.mark.run_only_on("GPU")
def test_completions_without_logprobs_request(mock_ray_deployment):
    """Test completions when logprobs are not requested."""
    request = {
        "prompt": "Test",
        "max_tokens": 5,
        # No logprobs parameter
    }

    mock_results = {
        "sentences": ["Generated text"],
    }

    with patch("ray.get", return_value=mock_results):
        output = asyncio.run(mock_ray_deployment.completions(request))

        logprobs = output["choices"][0]["logprobs"]

        # logprobs field should still exist but with None values
        assert "token_logprobs" in logprobs
        assert "top_logprobs" in logprobs
        assert logprobs["token_logprobs"] is None
        assert logprobs["top_logprobs"] is None


@pytest.mark.run_only_on("GPU")
def test_completions_complete_output_structure(mock_ray_deployment):
    """Test complete output structure matches OpenAI API format exactly."""
    request = {
        "prompt": "Question: An astronomer observes",
        "max_tokens": 40,
        "temperature": 1.0,
        "logprobs": 2,
    }

    mock_results = {
        "sentences": ["that a planet rotates faster"],
        "log_probs": np.array([[-0.5, -1.0, -0.8, -1.2, -0.9]]),
        "top_logprobs": [
            json.dumps(
                [
                    {"that": -0.5, "which": -1.5},
                    {"a": -1.0, "the": -1.8},
                    {"planet": -0.8, "star": -2.0},
                    {"rotates": -1.2, "spins": -1.9},
                    {"faster": -0.9, "slower": -2.1},
                ]
            )
        ],
    }

    with patch("ray.get", return_value=mock_results):
        output = asyncio.run(mock_ray_deployment.completions(request))

        # Verify complete structure
        assert "id" in output and isinstance(output["id"], str)
        assert output["object"] == "text_completion"
        assert "created" in output and isinstance(output["created"], int)
        assert output["model"] == "megatron-model"

        # Verify choices
        assert len(output["choices"]) == 1
        choice = output["choices"][0]
        assert isinstance(choice["text"], str)
        assert choice["index"] == 0
        assert choice["finish_reason"] in ["stop", "length"]

        # Verify logprobs structure
        logprobs = choice["logprobs"]
        assert isinstance(logprobs["token_logprobs"], list)
        assert isinstance(logprobs["top_logprobs"], list)
        assert len(logprobs["token_logprobs"]) == len(logprobs["top_logprobs"])

        # Verify usage
        usage = output["usage"]
        assert "prompt_tokens" in usage and isinstance(usage["prompt_tokens"], int)
        assert "completion_tokens" in usage and isinstance(usage["completion_tokens"], int)
        assert "total_tokens" in usage and isinstance(usage["total_tokens"], int)
        assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]


@pytest.mark.run_only_on("GPU")
def test_completions_token_logprobs_types(mock_ray_deployment):
    """Test that token_logprobs contains correct types (floats or None)."""
    request = {
        "prompt": "Test",
        "max_tokens": 5,
        "logprobs": 1,
    }

    mock_results = {
        "sentences": ["text"],
        "log_probs": np.array([[-0.1, -0.2, -0.3]]),
        "top_logprobs": [json.dumps([{"a": -0.1}, {"b": -0.2}, {"c": -0.3}])],
    }

    with patch("ray.get", return_value=mock_results):
        output = asyncio.run(mock_ray_deployment.completions(request))

        token_logprobs = output["choices"][0]["logprobs"]["token_logprobs"]

        for prob in token_logprobs:
            # Each element should be a float or None
            assert prob is None or isinstance(prob, (float, np.floating)), f"Expected float or None, got {type(prob)}"


@pytest.mark.run_only_on("GPU")
def test_completions_top_logprobs_dict_values(mock_ray_deployment):
    """Test that each dictionary in top_logprobs has string keys and numeric values."""
    request = {
        "prompt": "Test",
        "max_tokens": 3,
        "logprobs": 2,
    }

    mock_results = {
        "sentences": ["abc"],
        "log_probs": np.array([[-1.0, -2.0]]),
        "top_logprobs": [json.dumps([{"token1": -1.0, "token2": -1.5}, {"token3": -2.0, "token4": -2.5}])],
    }

    with patch("ray.get", return_value=mock_results):
        output = asyncio.run(mock_ray_deployment.completions(request))

        top_logprobs = output["choices"][0]["logprobs"]["top_logprobs"]

        for logprob_dict in top_logprobs:
            assert isinstance(logprob_dict, dict)
            assert len(logprob_dict) > 0  # Should have at least one entry

            for token, prob in logprob_dict.items():
                assert isinstance(token, str), f"Expected string key, got {type(token)}"
                assert isinstance(prob, (int, float, np.number)), f"Expected numeric value, got {type(prob)}"


@pytest.mark.run_only_on("GPU")
def test_completions_finish_reason_values(mock_ray_deployment):
    """Test that finish_reason is either 'stop' or 'length'."""
    # Test with length finish reason
    request_length = {
        "prompt": "Test",
        "max_tokens": 5,
    }

    mock_results_length = {
        "sentences": ["a" * 100],  # Long text that exceeds max_tokens
    }

    with patch("ray.get", return_value=mock_results_length):
        output = asyncio.run(mock_ray_deployment.completions(request_length))
        # The logic checks if generated_texts[0] length >= max_tokens
        finish_reason = output["choices"][0]["finish_reason"]
        assert finish_reason in ["stop", "length"]


@pytest.mark.run_only_on("GPU")
def test_completions_empty_sentences(mock_ray_deployment):
    """Test handling of empty sentences in results."""
    request = {
        "prompt": "Test",
        "max_tokens": 5,
    }

    mock_results = {
        "sentences": [],  # Empty sentences
    }

    with patch("ray.get", return_value=mock_results):
        output = asyncio.run(mock_ray_deployment.completions(request))

        # text should be empty string when no sentences
        assert output["choices"][0]["text"] == ""
        # finish_reason should be 'stop' when no generated text
        assert output["choices"][0]["finish_reason"] == "stop"


@pytest.mark.run_only_on("GPU")
def test_completions_with_real_example_format(mock_ray_deployment):
    """Test with a real-world example matching the user's provided output format."""
    request = {
        "prompt": "Question: An astronomer observes that a planet rotates faster after a meteorite impact. "
        "Which is the most likely effect of this increase in rotation?\nAnswer:",
        "max_tokens": 31,
        "temperature": 1.0,
        "top_p": 0.0,
        "logprobs": 1,
    }

    # Simulate the actual output structure
    mock_results = {
        "sentences": [
            "Question: An astronomer observes that a planet rotates faster after a meteorite impact. "
            "Which is the most likely effect of this increase in rotation?\nAnswer: Planetary years will become longer. This"
        ],
        "log_probs": np.array(
            [
                [
                    None,
                    -8.916147232055664,
                    -1.416137933731079,
                    -5.191495418548584,
                    -8.553435325622559,
                    -0.0010070496937260032,
                ]
            ]
        ),
        "top_logprobs": [
            json.dumps(
                [
                    {"Tags": -5.494272232055664},
                    {":\n": -0.2911378741264343},
                    {" Which": -2.566495418548584},
                    {" ": -1.9284353256225586},
                    {"er": -0.0010070496937260032},
                ]
            )
        ],
    }

    with patch("ray.get", return_value=mock_results):
        output = asyncio.run(mock_ray_deployment.completions(request))

        # Verify it matches the expected format
        assert isinstance(output["choices"][0]["text"], str)
        assert isinstance(output["choices"][0]["logprobs"], dict)
        assert isinstance(output["choices"][0]["logprobs"]["token_logprobs"], list)
        assert isinstance(output["choices"][0]["logprobs"]["top_logprobs"], list)

        # Verify top_logprobs is list of dicts
        for item in output["choices"][0]["logprobs"]["top_logprobs"]:
            assert isinstance(item, dict)

        # Verify the output has all required fields
        assert "id" in output
        assert "object" in output
        assert "created" in output
        assert "model" in output
        assert "choices" in output
        assert "usage" in output


@pytest.mark.run_only_on("GPU")
def test_completions_numpy_array_conversion(mock_ray_deployment):
    """Test that numpy arrays are properly converted to Python lists."""
    request = {
        "prompt": "Test",
        "max_tokens": 3,
        "logprobs": 1,
    }

    # Return numpy arrays (as the actual system might)
    mock_results = {
        "sentences": ["test"],
        "log_probs": np.array([[-1.0, -2.0, -3.0]]),  # numpy array
        "top_logprobs": [json.dumps([{"a": -1.0}, {"b": -2.0}, {"c": -3.0}])],
    }

    with patch("ray.get", return_value=mock_results):
        output = asyncio.run(mock_ray_deployment.completions(request))

        token_logprobs = output["choices"][0]["logprobs"]["token_logprobs"]

        # Should be a list, not numpy array
        assert isinstance(token_logprobs, list)
        # Each element should be JSON serializable (not numpy types)
        json.dumps(token_logprobs)  # Should not raise an error


@pytest.mark.run_only_on("GPU")
def test_completions_id_format(mock_ray_deployment):
    """Test that the ID field follows OpenAI format."""
    request = {
        "prompt": "Test",
        "max_tokens": 5,
    }

    mock_results = {
        "sentences": ["test"],
    }

    with patch("ray.get", return_value=mock_results):
        output = asyncio.run(mock_ray_deployment.completions(request))

        # ID should start with "cmpl-" prefix
        assert output["id"].startswith("cmpl-")
        # The rest should be a timestamp (numeric)
        id_suffix = output["id"].replace("cmpl-", "")
        assert id_suffix.isdigit()


@pytest.mark.run_only_on("GPU")
def test_completions_usage_token_counts(mock_ray_deployment):
    """Test that usage token counts are calculated correctly."""
    request = {
        "prompts": ["Hello world", "Test prompt"],
        "max_tokens": 10,
    }

    mock_results = {
        "sentences": ["Generated response", "Another response"],
    }

    with patch("ray.get", return_value=mock_results):
        output = asyncio.run(mock_ray_deployment.completions(request))

        usage = output["usage"]

        # Verify all fields exist and are integers
        assert isinstance(usage["prompt_tokens"], int)
        assert isinstance(usage["completion_tokens"], int)
        assert isinstance(usage["total_tokens"], int)

        # Verify the math is correct
        assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]

        # Verify they're all positive
        assert usage["prompt_tokens"] > 0
        assert usage["completion_tokens"] > 0
        assert usage["total_tokens"] > 0


# --- Stop words tests for completions endpoint ---
@pytest.mark.run_only_on("GPU")
def test_completions_stop_words_list_passed_to_inference(mock_ray_deployment):
    """Test that a list of stop words from the request is forwarded to inference inputs."""
    request = {
        "prompt": "Hello world",
        "max_tokens": 10,
        "stop": ["foo", "bar"],
    }

    mock_results = {"sentences": ["Generated text"]}

    with patch("ray.get", return_value=mock_results):
        asyncio.run(mock_ray_deployment.completions(request))

        inference_inputs = mock_ray_deployment.primary_worker.infer.remote.call_args[0][0]
        assert inference_inputs["stop_words"] == ["foo", "bar"]


@pytest.mark.run_only_on("GPU")
def test_completions_stop_words_string_converted_to_list(mock_ray_deployment):
    """Test that a single stop word string is converted to a list."""
    request = {
        "prompt": "Hello world",
        "max_tokens": 10,
        "stop": "end_token",
    }

    mock_results = {"sentences": ["Generated text"]}

    with patch("ray.get", return_value=mock_results):
        asyncio.run(mock_ray_deployment.completions(request))

        inference_inputs = mock_ray_deployment.primary_worker.infer.remote.call_args[0][0]
        assert inference_inputs["stop_words"] == ["end_token"]


@pytest.mark.run_only_on("GPU")
def test_completions_stop_words_none_when_absent(mock_ray_deployment):
    """Test that stop_words is None when not provided in the request."""
    request = {
        "prompt": "Hello world",
        "max_tokens": 10,
    }

    mock_results = {"sentences": ["Generated text"]}

    with patch("ray.get", return_value=mock_results):
        asyncio.run(mock_ray_deployment.completions(request))

        inference_inputs = mock_ray_deployment.primary_worker.infer.remote.call_args[0][0]
        assert inference_inputs["stop_words"] is None


# --- Stop words tests for chat completions endpoint ---
@pytest.mark.run_only_on("GPU")
def test_chat_completions_stop_words_list_passed_to_inference(mock_ray_deployment):
    """Test that a list of stop words from the chat request is forwarded to inference inputs."""
    request = {
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 10,
        "stop": ["foo", "bar"],
    }

    mock_results = {"sentences": ["Generated reply"]}

    with patch("ray.get", return_value=mock_results):
        asyncio.run(mock_ray_deployment.chat_completions(request))

        inference_inputs = mock_ray_deployment.primary_worker.infer.remote.call_args[0][0]
        assert inference_inputs["stop_words"] == ["foo", "bar"]


@pytest.mark.run_only_on("GPU")
def test_chat_completions_stop_words_string_converted_to_list(mock_ray_deployment):
    """Test that a single stop word string is converted to a list in chat completions."""
    request = {
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 10,
        "stop": "end_token",
    }

    mock_results = {"sentences": ["Generated reply"]}

    with patch("ray.get", return_value=mock_results):
        asyncio.run(mock_ray_deployment.chat_completions(request))

        inference_inputs = mock_ray_deployment.primary_worker.infer.remote.call_args[0][0]
        assert inference_inputs["stop_words"] == ["end_token"]


@pytest.mark.run_only_on("GPU")
def test_chat_completions_stop_words_none_when_absent(mock_ray_deployment):
    """Test that stop_words is None when not provided in the chat request."""
    request = {
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 10,
    }

    mock_results = {"sentences": ["Generated reply"]}

    with patch("ray.get", return_value=mock_results):
        asyncio.run(mock_ray_deployment.chat_completions(request))

        inference_inputs = mock_ray_deployment.primary_worker.infer.remote.call_args[0][0]
        assert inference_inputs["stop_words"] is None


@pytest.mark.run_only_on("GPU")
def test_chat_completions_output_format_with_stop_words(mock_ray_deployment):
    """Test that chat completions output format is correct when stop words are provided."""
    request = {
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 50,
        "stop": ["<|end|>"],
    }

    mock_results = {"sentences": ["Hi there! How can I help?"]}

    with patch("ray.get", return_value=mock_results):
        output = asyncio.run(mock_ray_deployment.chat_completions(request))

        assert output["object"] == "chat.completion"
        assert output["model"] == "megatron-model"
        assert output["id"].startswith("chatcmpl-")

        choice = output["choices"][0]
        assert choice["message"]["role"] == "assistant"
        assert isinstance(choice["message"]["content"], str)
        assert choice["index"] == 0
        assert choice["finish_reason"] in ["stop", "length"]

        assert "usage" in output
