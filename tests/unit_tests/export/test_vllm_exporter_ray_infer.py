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


from unittest.mock import MagicMock, patch

import numpy as np
import pytest

try:
    import vllm  # noqa: F401

    HAVE_VLLM = True
except ImportError:
    HAVE_VLLM = False


@pytest.fixture
def exporter():
    from nemo_export.vllm_exporter import vLLMExporter

    return vLLMExporter()


@pytest.fixture
def mock_llm():
    with patch("nemo_export.vllm_exporter.LLM") as mock:
        mock_instance = MagicMock()
        mock.return_value = mock_instance
        yield mock


@pytest.mark.skipif(not HAVE_VLLM, reason="Need to enable virtual environment for vLLM")
@pytest.mark.run_only_on("GPU")
def test_ray_infer_fn_basic_usage(exporter, mock_llm):
    """Test ray_infer_fn with basic input."""
    # Mock the forward method
    exporter.model = MagicMock()
    exporter.forward = MagicMock(return_value={"sentences": ["Generated text"]})

    inputs = {
        "prompts": ["Hello, how are you?"],
        "max_tokens": 50,
        "temperature": 0.7,
    }

    result = exporter.ray_infer_fn(inputs)

    assert result["sentences"] == ["Generated text"]
    exporter.forward.assert_called_once_with(
        input_texts=["Hello, how are you?"],
        max_tokens=50,
        temperature=0.7,
    )


@pytest.mark.skipif(not HAVE_VLLM, reason="Need to enable virtual environment for vLLM")
@pytest.mark.run_only_on("GPU")
def test_ray_infer_fn_single_string_prompt(exporter, mock_llm):
    """Test ray_infer_fn with single string prompt (should be converted to list)."""
    exporter.model = MagicMock()
    exporter.forward = MagicMock(return_value={"sentences": ["Generated text"]})

    inputs = {
        "prompts": "Hello, how are you?",
        "max_tokens": 50,
    }

    result = exporter.ray_infer_fn(inputs)

    assert result["sentences"] == ["Generated text"]
    exporter.forward.assert_called_once_with(
        input_texts=["Hello, how are you?"],
        max_tokens=50,
    )


@pytest.mark.skipif(not HAVE_VLLM, reason="Need to enable virtual environment for vLLM")
@pytest.mark.run_only_on("GPU")
def test_ray_infer_fn_with_all_parameters(exporter, mock_llm):
    """Test ray_infer_fn with all possible parameters."""
    exporter.model = MagicMock()
    exporter.forward = MagicMock(return_value={"sentences": ["Generated text"]})

    inputs = {
        "prompts": ["Hello, how are you?"],
        "max_tokens": 100,
        "min_tokens": 10,
        "top_k": 50,
        "top_p": 0.9,
        "temperature": 0.8,
        "seed": 42,
        "n_log_probs": 5,
        "n_prompt_log_probs": 3,
        "lora_model_name": "test_lora",
    }

    result = exporter.ray_infer_fn(inputs)

    assert result["sentences"] == ["Generated text"]
    exporter.forward.assert_called_once_with(
        input_texts=["Hello, how are you?"],
        max_tokens=100,
        min_tokens=10,
        top_k=50,
        top_p=0.9,
        temperature=0.8,
        seed=42,
        n_log_probs=5,
        n_prompt_log_probs=3,
        lora_model_name="test_lora",
    )


@pytest.mark.skipif(not HAVE_VLLM, reason="Need to enable virtual environment for vLLM")
@pytest.mark.run_only_on("GPU")
def test_ray_infer_fn_with_log_probs(exporter, mock_llm):
    """Test ray_infer_fn with log probabilities in output."""
    exporter.model = MagicMock()
    exporter.forward = MagicMock(
        return_value={
            "sentences": ["Generated text"],
            "log_probs": [["token1", "token2"]],
            "prompt_log_probs": [["prompt_token1"]],
        }
    )

    inputs = {
        "prompts": ["Hello, how are you?"],
        "n_log_probs": 2,
        "n_prompt_log_probs": 1,
    }

    result = exporter.ray_infer_fn(inputs)

    assert result["sentences"] == ["Generated text"]
    assert result["log_probs"] == [["token1", "token2"]]
    assert result["prompt_log_probs"] == [["prompt_token1"]]


@pytest.mark.skipif(not HAVE_VLLM, reason="Need to enable virtual environment for vLLM")
@pytest.mark.run_only_on("GPU")
def test_ray_infer_fn_with_empty_prompts(exporter, mock_llm):
    """Test ray_infer_fn with empty prompts list."""
    exporter.model = MagicMock()
    exporter.forward = MagicMock(return_value={"sentences": []})

    inputs = {
        "prompts": [],
        "max_tokens": 50,
    }

    result = exporter.ray_infer_fn(inputs)

    assert result["sentences"] == []
    exporter.forward.assert_called_once_with(
        input_texts=[],
        max_tokens=50,
    )


@pytest.mark.skipif(not HAVE_VLLM, reason="Need to enable virtual environment for vLLM")
@pytest.mark.run_only_on("GPU")
def test_ray_infer_fn_with_multiple_prompts(exporter, mock_llm):
    """Test ray_infer_fn with multiple prompts."""
    exporter.model = MagicMock()
    exporter.forward = MagicMock(return_value={"sentences": ["Text 1", "Text 2", "Text 3"]})

    inputs = {
        "prompts": ["Prompt 1", "Prompt 2", "Prompt 3"],
        "max_tokens": 50,
    }

    result = exporter.ray_infer_fn(inputs)

    assert result["sentences"] == ["Text 1", "Text 2", "Text 3"]
    exporter.forward.assert_called_once_with(
        input_texts=["Prompt 1", "Prompt 2", "Prompt 3"],
        max_tokens=50,
    )


@pytest.mark.skipif(not HAVE_VLLM, reason="Need to enable virtual environment for vLLM")
@pytest.mark.run_only_on("GPU")
def test_ray_infer_fn_with_forward_error(exporter, mock_llm):
    """Test ray_infer_fn when forward method raises an exception."""
    exporter.model = MagicMock()
    exporter.forward = MagicMock(side_effect=Exception("Forward error"))

    inputs = {
        "prompts": ["Hello, how are you?"],
        "max_tokens": 50,
    }

    result = exporter.ray_infer_fn(inputs)

    assert "error" in result
    assert result["error"] == "An error occurred: Forward error"
    assert result["sentences"] == ["An error occurred: Forward error"]


@pytest.mark.skipif(not HAVE_VLLM, reason="Need to enable virtual environment for vLLM")
@pytest.mark.run_only_on("GPU")
def test_ray_infer_fn_with_forward_error_multiple_prompts(exporter, mock_llm):
    """Test ray_infer_fn when forward method raises an exception with multiple prompts."""
    exporter.model = MagicMock()
    exporter.forward = MagicMock(side_effect=Exception("Forward error"))

    inputs = {
        "prompts": ["Prompt 1", "Prompt 2", "Prompt 3"],
        "max_tokens": 50,
    }

    result = exporter.ray_infer_fn(inputs)

    assert "error" in result
    assert result["error"] == "An error occurred: Forward error"
    assert result["sentences"] == [
        "An error occurred: Forward error",
        "An error occurred: Forward error",
        "An error occurred: Forward error",
    ]


@pytest.mark.skipif(not HAVE_VLLM, reason="Need to enable virtual environment for vLLM")
@pytest.mark.run_only_on("GPU")
def test_ray_infer_fn_with_forward_dict_output(exporter, mock_llm):
    """Test ray_infer_fn when forward returns a dict output."""
    exporter.model = MagicMock()
    exporter.forward = MagicMock(return_value={"sentences": ["Generated text"], "custom_field": "value"})

    inputs = {
        "prompts": ["Hello, how are you?"],
        "max_tokens": 50,
    }

    result = exporter.ray_infer_fn(inputs)

    assert result["sentences"] == ["Generated text"]
    assert result["custom_field"] == "value"


@pytest.mark.skipif(not HAVE_VLLM, reason="Need to enable virtual environment for vLLM")
@pytest.mark.run_only_on("GPU")
def test_ray_infer_fn_with_forward_non_dict_output(exporter, mock_llm):
    """Test ray_infer_fn when forward returns a non-dict output."""
    exporter.model = MagicMock()
    exporter.forward = MagicMock(return_value="Not a dict")

    inputs = {
        "prompts": ["Hello, how are you?"],
        "max_tokens": 50,
    }

    result = exporter.ray_infer_fn(inputs)

    assert result["sentences"] == ["An error occurred: the output format is expected to be a dict."]


@pytest.mark.skipif(not HAVE_VLLM, reason="Need to enable virtual environment for vLLM")
@pytest.mark.run_only_on("GPU")
def test_ray_infer_fn_with_missing_prompts(exporter, mock_llm):
    """Test ray_infer_fn with missing prompts key."""
    exporter.model = MagicMock()
    exporter.forward = MagicMock(return_value={"sentences": []})

    inputs = {
        "max_tokens": 50,
    }

    result = exporter.ray_infer_fn(inputs)

    assert result["sentences"] == []
    exporter.forward.assert_called_once_with(
        input_texts=[],
        max_tokens=50,
    )


@pytest.mark.skipif(not HAVE_VLLM, reason="Need to enable virtual environment for vLLM")
@pytest.mark.run_only_on("GPU")
def test_ray_infer_fn_parameter_type_conversion(exporter, mock_llm):
    """Test ray_infer_fn ensures proper type conversion of parameters."""
    exporter.model = MagicMock()
    exporter.forward = MagicMock(return_value={"sentences": ["Generated text"]})

    inputs = {
        "prompts": ["Hello, how are you?"],
        "max_tokens": "50",  # String instead of int
        "temperature": "0.7",  # String instead of float
        "top_k": "10",  # String instead of int
        "top_p": "0.9",  # String instead of float
        "seed": "42",  # String instead of int
        "n_log_probs": "5",  # String instead of int
        "n_prompt_log_probs": "3",  # String instead of int
    }

    result = exporter.ray_infer_fn(inputs)

    assert result["sentences"] == ["Generated text"]
    # Verify that the _infer_fn method handles type conversion
    exporter.forward.assert_called_once_with(
        input_texts=["Hello, how are you?"],
        max_tokens=50,
        temperature=0.7,
        top_k=10,
        top_p=0.9,
        seed=42,
        n_log_probs=5,
        n_prompt_log_probs=3,
    )


@pytest.mark.skipif(not HAVE_VLLM, reason="Need to enable virtual environment for vLLM")
@pytest.mark.run_only_on("GPU")
def test_ray_infer_fn_with_numpy_arrays(exporter, mock_llm):
    """Test ray_infer_fn with numpy array inputs."""
    exporter.model = MagicMock()
    exporter.forward = MagicMock(return_value={"sentences": ["Generated text"]})

    inputs = {
        "prompts": ["Hello, how are you?"],
        "max_tokens": np.array([50]),
        "temperature": np.array([0.7]),
        "top_k": np.array([10]),
        "top_p": np.array([0.9]),
        "seed": np.array([42]),
        "n_log_probs": np.array([5]),
        "n_prompt_log_probs": np.array([3]),
    }

    result = exporter.ray_infer_fn(inputs)

    assert result["sentences"] == ["Generated text"]
    # Verify that numpy arrays are properly converted
    exporter.forward.assert_called_once_with(
        input_texts=["Hello, how are you?"],
        max_tokens=50,
        temperature=0.7,
        top_k=10,
        top_p=0.9,
        seed=42,
        n_log_probs=5,
        n_prompt_log_probs=3,
    )


# Tests for post_process_logprobs_to_OAI method and ray_infer_fn with logprobs processing
@pytest.mark.skipif(not HAVE_VLLM, reason="Need to enable virtual environment for vLLM")
@pytest.mark.run_only_on("GPU")
def test_post_process_logprobs_to_OAI_basic(exporter, mock_llm):
    """Test post_process_logprobs_to_OAI with basic log probabilities."""
    import json

    exporter.model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_tokenizer.decode.side_effect = lambda x: f"token_{x[0]}"
    exporter.model.get_tokenizer.return_value = mock_tokenizer

    output_dict = {
        "sentences": ["Generated text"],
        "token_ids": [[1, 2, 3]],
        "log_probs": np.array(
            [
                [
                    json.dumps({"token_1": -0.1, " alt1": -2.5}),
                    json.dumps({"token_2": -0.2, " alt2": -3.0}),
                    json.dumps({"token_3": -0.3, " alt3": -2.8}),
                ]
            ]
        ),
    }

    result = exporter.post_process_logprobs_to_OAI(output_dict, echo=False, n_top_logprobs=0)

    assert "log_probs" in result
    assert result["log_probs"] == [[-0.1, -0.2, -0.3]]


@pytest.mark.skipif(not HAVE_VLLM, reason="Need to enable virtual environment for vLLM")
@pytest.mark.run_only_on("GPU")
def test_post_process_logprobs_to_OAI_with_top_logprobs(exporter, mock_llm):
    """Test post_process_logprobs_to_OAI with top_logprobs processing."""
    import json

    exporter.model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_tokenizer.decode.side_effect = lambda x: f"token_{x[0]}"
    exporter.model.get_tokenizer.return_value = mock_tokenizer

    output_dict = {
        "sentences": ["Generated text"],
        "token_ids": [[1, 2]],
        "log_probs": np.array(
            [
                [
                    json.dumps({"token_1": -0.1, " alt1": -2.5, " alt2": -3.0}),
                    json.dumps({"token_2": -0.2, " alt3": -2.8, " alt4": -3.2}),
                ]
            ]
        ),
    }

    result = exporter.post_process_logprobs_to_OAI(output_dict, echo=False, n_top_logprobs=2)

    assert "log_probs" in result
    assert result["log_probs"] == [[-0.1, -0.2]]
    assert "top_logprobs" in result
    assert len(result["top_logprobs"]) == 1
    assert len(result["top_logprobs"][0]) == 2
    # Check that top 2 are selected and sorted by value
    assert result["top_logprobs"][0][0] == {"token_1": -0.1, " alt1": -2.5}
    assert result["top_logprobs"][0][1] == {"token_2": -0.2, " alt3": -2.8}


@pytest.mark.skipif(not HAVE_VLLM, reason="Need to enable virtual environment for vLLM")
@pytest.mark.run_only_on("GPU")
def test_post_process_logprobs_to_OAI_with_echo(exporter, mock_llm):
    """Test post_process_logprobs_to_OAI with echo mode enabled."""
    import json

    exporter.model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_tokenizer.decode.side_effect = lambda x: f"token_{x[0]}"
    exporter.model.get_tokenizer.return_value = mock_tokenizer

    output_dict = {
        "sentences": ["Prompt text generated text"],
        "token_ids": [[3, 4]],  # Generated tokens
        "prompt_token_ids": [[1, 2]],  # Prompt tokens
        "log_probs": np.array(
            [
                [
                    json.dumps({"token_3": -0.3, " alt1": -2.5}),
                    json.dumps({"token_4": -0.4, " alt2": -3.0}),
                ]
            ]
        ),
        "prompt_log_probs": np.array(
            [
                [
                    json.dumps({"token_1": -0.1, " alt3": -2.8}),
                    json.dumps({"token_2": -0.2, " alt4": -3.2}),
                ]
            ]
        ),
    }

    result = exporter.post_process_logprobs_to_OAI(output_dict, echo=True, n_top_logprobs=0)

    assert "log_probs" in result
    # Should have prompt logprobs first, then generated logprobs
    assert result["log_probs"] == [[-0.1, -0.2, -0.3, -0.4]]


@pytest.mark.skipif(not HAVE_VLLM, reason="Need to enable virtual environment for vLLM")
@pytest.mark.run_only_on("GPU")
def test_post_process_logprobs_to_OAI_with_echo_and_top_logprobs(exporter, mock_llm):
    """Test post_process_logprobs_to_OAI with both echo and top_logprobs."""
    import json

    exporter.model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_tokenizer.decode.side_effect = lambda x: f"token_{x[0]}"
    exporter.model.get_tokenizer.return_value = mock_tokenizer

    output_dict = {
        "sentences": ["Prompt text generated text"],
        "token_ids": [[3]],  # Generated tokens
        "prompt_token_ids": [[1, 2]],  # Prompt tokens
        "log_probs": np.array([[json.dumps({"token_3": -0.3, " alt1": -2.5, " alt2": -3.0})]]),
        "prompt_log_probs": np.array(
            [
                [
                    json.dumps({"token_1": -0.1, " alt3": -2.8, " alt4": -3.2}),
                    json.dumps({"token_2": -0.2, " alt5": -2.9, " alt6": -3.1}),
                ]
            ]
        ),
    }

    result = exporter.post_process_logprobs_to_OAI(output_dict, echo=True, n_top_logprobs=2)

    assert "log_probs" in result
    assert result["log_probs"] == [[-0.1, -0.2, -0.3]]
    assert "top_logprobs" in result
    assert len(result["top_logprobs"]) == 1
    # Should have 3 dicts: 2 prompt + 1 generated
    assert len(result["top_logprobs"][0]) == 3
    assert result["top_logprobs"][0][0] == {"token_1": -0.1, " alt3": -2.8}
    assert result["top_logprobs"][0][1] == {"token_2": -0.2, " alt5": -2.9}
    assert result["top_logprobs"][0][2] == {"token_3": -0.3, " alt1": -2.5}


@pytest.mark.skipif(not HAVE_VLLM, reason="Need to enable virtual environment for vLLM")
@pytest.mark.run_only_on("GPU")
def test_post_process_logprobs_to_OAI_with_empty_decoded_token(exporter, mock_llm):
    """Test post_process_logprobs_to_OAI when tokenizer.decode returns empty string (edge case for special tokens)."""
    import json

    exporter.model = MagicMock()
    mock_tokenizer = MagicMock()
    # Simulate <|end_of_text|> token (128001) decoding to empty string
    mock_tokenizer.decode.side_effect = lambda x: "" if x[0] == 128001 else f"token_{x[0]}"
    exporter.model.get_tokenizer.return_value = mock_tokenizer

    output_dict = {
        "sentences": ["Generated text"],
        "token_ids": [[1, 128001]],  # Second token is end-of-text
        "log_probs": np.array(
            [
                [
                    json.dumps({"token_1": -0.1, " alt1": -2.5}),
                    json.dumps({"": -0.5, " alt2": -3.0}),  # Empty string key for special token
                ]
            ]
        ),
    }

    result = exporter.post_process_logprobs_to_OAI(output_dict, echo=False, n_top_logprobs=0)

    # Should handle the empty decoded token by taking first entry
    assert "log_probs" in result
    assert result["log_probs"] == [[-0.1, -0.5]]  # Should use "" key's value


@pytest.mark.skipif(not HAVE_VLLM, reason="Need to enable virtual environment for vLLM")
@pytest.mark.run_only_on("GPU")
def test_post_process_logprobs_to_OAI_no_token_match_fallback(exporter, mock_llm):
    """Test post_process_logprobs_to_OAI when decoded token doesn't match any key (fallback to first entry)."""
    import json

    exporter.model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_tokenizer.decode.side_effect = lambda x: "decoded_token"  # Returns a token that won't match
    exporter.model.get_tokenizer.return_value = mock_tokenizer

    output_dict = {
        "sentences": ["Generated text"],
        "token_ids": [[123]],
        "log_probs": np.array(
            [
                [json.dumps({"different_token": -0.5, " alt": -2.0})]  # No exact match
            ]
        ),
    }

    result = exporter.post_process_logprobs_to_OAI(output_dict, echo=False, n_top_logprobs=0)

    # Should fall back to first entry
    assert "log_probs" in result
    assert result["log_probs"] == [[-0.5]]  # Should use first entry's value


@pytest.mark.skipif(not HAVE_VLLM, reason="Need to enable virtual environment for vLLM")
@pytest.mark.run_only_on("GPU")
def test_post_process_logprobs_to_OAI_missing_token_ids(exporter, mock_llm):
    """Test post_process_logprobs_to_OAI when token_ids are missing."""
    exporter.model = MagicMock()

    output_dict = {
        "sentences": ["Generated text"],
        "log_probs": np.array([[{"token": -0.1}]]),
        # No token_ids
    }

    result = exporter.post_process_logprobs_to_OAI(output_dict, echo=False, n_top_logprobs=0)

    # Should return unchanged
    assert result == output_dict


@pytest.mark.skipif(not HAVE_VLLM, reason="Need to enable virtual environment for vLLM")
@pytest.mark.run_only_on("GPU")
def test_ray_infer_fn_with_compute_logprob_and_n_top_logprobs(exporter, mock_llm):
    """Test ray_infer_fn with compute_logprob and n_top_logprobs parameters."""
    import json

    exporter.model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_tokenizer.decode.side_effect = lambda x: f"token_{x[0]}"
    exporter.model.get_tokenizer.return_value = mock_tokenizer

    # Mock forward to return raw output
    exporter.forward = MagicMock(
        return_value={
            "sentences": ["Generated text"],
            "token_ids": [[1, 2]],
            "log_probs": np.array(
                [
                    [
                        json.dumps({"token_1": -0.1, " alt1": -2.5}),
                        json.dumps({"token_2": -0.2, " alt2": -3.0}),
                    ]
                ]
            ),
        }
    )

    inputs = {
        "prompts": ["Test prompt"],
        "max_tokens": 10,
        "compute_logprob": True,
        "n_top_logprobs": 2,
        "echo": False,
    }

    result = exporter.ray_infer_fn(inputs)

    assert result["sentences"] == ["Generated text"]
    assert "log_probs" in result
    assert result["log_probs"] == [[-0.1, -0.2]]
    assert "top_logprobs" in result
    assert len(result["top_logprobs"]) == 1
    assert len(result["top_logprobs"][0]) == 2


@pytest.mark.skipif(not HAVE_VLLM, reason="Need to enable virtual environment for vLLM")
@pytest.mark.run_only_on("GPU")
def test_ray_infer_fn_with_echo_mode(exporter, mock_llm):
    """Test ray_infer_fn with echo mode enabled."""
    import json

    exporter.model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_tokenizer.decode.side_effect = lambda x: f"token_{x[0]}"
    exporter.model.get_tokenizer.return_value = mock_tokenizer

    # Mock forward to return output with prompt logprobs
    exporter.forward = MagicMock(
        return_value={
            "sentences": ["Prompt text generated"],
            "token_ids": [[3]],
            "prompt_token_ids": [[1, 2]],
            "log_probs": np.array([[json.dumps({"token_3": -0.3, " alt": -2.5})]]),
            "prompt_log_probs": np.array(
                [
                    [
                        json.dumps({"token_1": -0.1, " alt": -2.8}),
                        json.dumps({"token_2": -0.2, " alt": -3.0}),
                    ]
                ]
            ),
        }
    )

    inputs = {
        "prompts": ["Test prompt"],
        "max_tokens": 10,
        "compute_logprob": True,
        "n_top_logprobs": 1,
        "echo": True,
    }

    result = exporter.ray_infer_fn(inputs)

    assert result["sentences"] == ["Prompt text generated"]
    assert "log_probs" in result
    # Should include prompt logprobs + generated logprobs
    assert result["log_probs"] == [[-0.1, -0.2, -0.3]]


@pytest.mark.skipif(not HAVE_VLLM, reason="Need to enable virtual environment for vLLM")
@pytest.mark.run_only_on("GPU")
def test_ray_infer_fn_handles_post_processing_error(exporter, mock_llm):
    """Test ray_infer_fn handles errors from post_process_logprobs_to_OAI gracefully."""
    exporter.model = MagicMock()

    # Mock forward to raise an exception
    exporter.forward = MagicMock(side_effect=Exception("Post-processing error"))

    inputs = {
        "prompts": ["Test prompt"],
        "max_tokens": 10,
        "compute_logprob": True,
        "n_top_logprobs": 1,
    }

    result = exporter.ray_infer_fn(inputs)

    assert "error" in result
    assert "Post-processing error" in result["error"]
    assert result["sentences"] == ["An error occurred: Post-processing error"]


@pytest.mark.skipif(not HAVE_VLLM, reason="Need to enable virtual environment for vLLM")
@pytest.mark.run_only_on("GPU")
def test_ray_infer_fn_maps_hf_style_parameters_to_vllm(exporter, mock_llm):
    """Test ray_infer_fn correctly maps HF-style parameters (compute_logprob, n_top_logprobs, echo) to vLLM parameters."""
    import json

    exporter.model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_tokenizer.decode.side_effect = lambda x: f"token_{x[0]}"
    exporter.model.get_tokenizer.return_value = mock_tokenizer

    # Track what parameters are passed to forward
    captured_forward_params = {}

    def mock_forward(**kwargs):
        captured_forward_params.update(kwargs)
        return {
            "sentences": ["Generated"],
            "token_ids": [[1]],
            "prompt_token_ids": [[2]],
            "log_probs": np.array([[json.dumps({"token_1": -0.1})]]),
            "prompt_log_probs": np.array([[json.dumps({"token_2": -0.2})]]),
        }

    exporter.forward = mock_forward

    inputs = {
        "prompts": ["Test"],
        "compute_logprob": True,
        "n_top_logprobs": 3,
        "echo": True,
    }

    result = exporter.ray_infer_fn(inputs)

    # Verify vLLM parameters were set correctly
    assert captured_forward_params["n_log_probs"] == 3
    assert captured_forward_params["n_prompt_log_probs"] == 3

    # Verify output is processed
    assert "log_probs" in result
    assert result["log_probs"] == [[-0.2, -0.1]]  # Prompt first, then generated


@pytest.mark.skipif(not HAVE_VLLM, reason="Need to enable virtual environment for vLLM")
@pytest.mark.run_only_on("GPU")
def test_post_process_logprobs_to_OAI_multiple_samples(exporter, mock_llm):
    """Test post_process_logprobs_to_OAI with multiple samples (batch processing)."""
    import json

    exporter.model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_tokenizer.decode.side_effect = lambda x: f"token_{x[0]}"
    exporter.model.get_tokenizer.return_value = mock_tokenizer

    output_dict = {
        "sentences": ["Generated text 1", "Generated text 2"],
        "token_ids": [[1, 2], [3, 4]],
        "log_probs": np.array(
            [
                [
                    json.dumps({"token_1": -0.1, " alt": -2.5}),
                    json.dumps({"token_2": -0.2, " alt": -3.0}),
                ],
                [
                    json.dumps({"token_3": -0.3, " alt": -2.8}),
                    json.dumps({"token_4": -0.4, " alt": -3.2}),
                ],
            ]
        ),
    }

    result = exporter.post_process_logprobs_to_OAI(output_dict, echo=False, n_top_logprobs=0)

    assert "log_probs" in result
    assert len(result["log_probs"]) == 2
    assert result["log_probs"][0] == [-0.1, -0.2]
    assert result["log_probs"][1] == [-0.3, -0.4]


class TestVLLMExporterApplyChatTemplate:
    """Tests for apply_chat_template functionality added to vLLMExporter."""

    @pytest.fixture
    def mock_tokenizer_with_chat_template(self):
        """Create a mock tokenizer with chat template support."""
        tokenizer = MagicMock()
        tokenizer.chat_template = "{% for message in messages %}{{ message.role }}: {{ message.content }}\n{% endfor %}"
        tokenizer.apply_chat_template = MagicMock(return_value="<|begin_of_text|>user: Hello\nassistant:")
        return tokenizer

    @pytest.fixture
    def mock_tokenizer_without_chat_template(self):
        """Create a mock tokenizer without chat template support."""
        tokenizer = MagicMock()
        tokenizer.chat_template = None
        return tokenizer

    @pytest.fixture
    def exporter_with_chat_template(self, mock_tokenizer_with_chat_template):
        """Create vLLMExporter instance with chat template support."""
        from nemo_export.vllm_exporter import vLLMExporter

        exporter = vLLMExporter()
        exporter.model = MagicMock()
        exporter.model.get_tokenizer.return_value = mock_tokenizer_with_chat_template
        return exporter

    @pytest.fixture
    def exporter_without_chat_template(self, mock_tokenizer_without_chat_template):
        """Create vLLMExporter instance without chat template support."""
        from nemo_export.vllm_exporter import vLLMExporter

        exporter = vLLMExporter()
        exporter.model = MagicMock()
        exporter.model.get_tokenizer.return_value = mock_tokenizer_without_chat_template
        return exporter

    @pytest.mark.skipif(not HAVE_VLLM, reason="Need to enable virtual environment for vLLM")
    @pytest.mark.run_only_on("GPU")
    def test_apply_chat_template_with_messages_list(self, exporter_with_chat_template, mock_tokenizer_with_chat_template):
        """Test apply_chat_template with a list of message dictionaries."""
        messages = [{"role": "user", "content": "Hello, how are you?"}]

        result = exporter_with_chat_template.apply_chat_template(messages)

        assert result == "<|begin_of_text|>user: Hello\nassistant:"
        mock_tokenizer_with_chat_template.apply_chat_template.assert_called_once_with(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    @pytest.mark.skipif(not HAVE_VLLM, reason="Need to enable virtual environment for vLLM")
    @pytest.mark.run_only_on("GPU")
    def test_apply_chat_template_with_json_string(self, exporter_with_chat_template, mock_tokenizer_with_chat_template):
        """Test apply_chat_template with JSON string input."""
        import json

        messages = [{"role": "user", "content": "Hello"}]
        messages_json = json.dumps(messages)

        result = exporter_with_chat_template.apply_chat_template(messages_json)

        assert result == "<|begin_of_text|>user: Hello\nassistant:"
        # Verify it was called with the parsed list (not the JSON string)
        mock_tokenizer_with_chat_template.apply_chat_template.assert_called_once()
        call_args = mock_tokenizer_with_chat_template.apply_chat_template.call_args
        assert call_args[0][0] == messages  # First positional arg should be the parsed messages list

    @pytest.mark.skipif(not HAVE_VLLM, reason="Need to enable virtual environment for vLLM")
    @pytest.mark.run_only_on("GPU")
    def test_apply_chat_template_without_generation_prompt(self, exporter_with_chat_template, mock_tokenizer_with_chat_template):
        """Test apply_chat_template with add_generation_prompt=False."""
        messages = [{"role": "user", "content": "Hello"}]

        exporter_with_chat_template.apply_chat_template(messages, add_generation_prompt=False)

        mock_tokenizer_with_chat_template.apply_chat_template.assert_called_once_with(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )

    @pytest.mark.skipif(not HAVE_VLLM, reason="Need to enable virtual environment for vLLM")
    @pytest.mark.run_only_on("GPU")
    def test_apply_chat_template_raises_error_when_no_template(self, exporter_without_chat_template):
        """Test apply_chat_template raises ValueError when tokenizer has no chat template."""
        messages = [{"role": "user", "content": "Hello"}]

        with pytest.raises(ValueError, match="The tokenizer does not have a chat template defined"):
            exporter_without_chat_template.apply_chat_template(messages)

    @pytest.mark.skipif(not HAVE_VLLM, reason="Need to enable virtual environment for vLLM")
    @pytest.mark.run_only_on("GPU")
    def test_apply_chat_template_with_multi_turn_conversation(self, exporter_with_chat_template, mock_tokenizer_with_chat_template):
        """Test apply_chat_template with multi-turn conversation."""
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
        ]

        exporter_with_chat_template.apply_chat_template(messages)

        mock_tokenizer_with_chat_template.apply_chat_template.assert_called_once_with(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

    @pytest.mark.skipif(not HAVE_VLLM, reason="Need to enable virtual environment for vLLM")
    @pytest.mark.run_only_on("GPU")
    def test_ray_infer_fn_with_apply_chat_template(self, exporter_with_chat_template, mock_tokenizer_with_chat_template):
        """Test ray_infer_fn correctly applies chat template when requested."""
        exporter_with_chat_template.forward = MagicMock(return_value={"sentences": ["I'm doing well, thank you!"]})

        messages = [{"role": "user", "content": "Hello"}]
        inputs = {
            "prompts": [messages],
            "max_tokens": 100,
            "apply_chat_template": True,
        }

        result = exporter_with_chat_template.ray_infer_fn(inputs)

        assert "sentences" in result
        assert result["sentences"] == ["I'm doing well, thank you!"]
        # Verify apply_chat_template was called
        mock_tokenizer_with_chat_template.apply_chat_template.assert_called_once()
        # Verify forward was called with the formatted prompt string
        exporter_with_chat_template.forward.assert_called_once()
        call_kwargs = exporter_with_chat_template.forward.call_args[1]
        assert call_kwargs["input_texts"] == ["<|begin_of_text|>user: Hello\nassistant:"]

    @pytest.mark.skipif(not HAVE_VLLM, reason="Need to enable virtual environment for vLLM")
    @pytest.mark.run_only_on("GPU")
    def test_ray_infer_fn_without_apply_chat_template(self, exporter_with_chat_template, mock_tokenizer_with_chat_template):
        """Test ray_infer_fn does not apply chat template when not requested."""
        exporter_with_chat_template.forward = MagicMock(return_value={"sentences": ["Generated text"]})

        inputs = {
            "prompts": ["plain text prompt"],
            "max_tokens": 100,
            "apply_chat_template": False,
        }

        result = exporter_with_chat_template.ray_infer_fn(inputs)

        assert "sentences" in result
        # Verify apply_chat_template was NOT called
        mock_tokenizer_with_chat_template.apply_chat_template.assert_not_called()
        # Verify forward was called with the original prompt
        exporter_with_chat_template.forward.assert_called_once()
        call_kwargs = exporter_with_chat_template.forward.call_args[1]
        assert call_kwargs["input_texts"] == ["plain text prompt"]

    @pytest.mark.skipif(not HAVE_VLLM, reason="Need to enable virtual environment for vLLM")
    @pytest.mark.run_only_on("GPU")
    def test_ray_infer_fn_with_apply_chat_template_default_false(self, exporter_with_chat_template, mock_tokenizer_with_chat_template):
        """Test ray_infer_fn defaults to not applying chat template."""
        exporter_with_chat_template.forward = MagicMock(return_value={"sentences": ["Generated text"]})

        inputs = {
            "prompts": ["plain text prompt"],
            "max_tokens": 100,
            # apply_chat_template not specified - should default to False
        }

        result = exporter_with_chat_template.ray_infer_fn(inputs)

        assert "sentences" in result
        # Verify apply_chat_template was NOT called (default is False)
        mock_tokenizer_with_chat_template.apply_chat_template.assert_not_called()

    @pytest.mark.skipif(not HAVE_VLLM, reason="Need to enable virtual environment for vLLM")
    @pytest.mark.run_only_on("GPU")
    def test_ray_infer_fn_with_multiple_chat_prompts(self, exporter_with_chat_template, mock_tokenizer_with_chat_template):
        """Test ray_infer_fn with multiple chat prompts applies template to each."""
        mock_tokenizer_with_chat_template.apply_chat_template.side_effect = [
            "<|begin_of_text|>user: Hello\nassistant:",
            "<|begin_of_text|>user: Goodbye\nassistant:",
        ]
        exporter_with_chat_template.forward = MagicMock(
            return_value={"sentences": ["Hi there!", "See you later!"]}
        )

        messages1 = [{"role": "user", "content": "Hello"}]
        messages2 = [{"role": "user", "content": "Goodbye"}]
        inputs = {
            "prompts": [messages1, messages2],
            "max_tokens": 100,
            "apply_chat_template": True,
        }

        result = exporter_with_chat_template.ray_infer_fn(inputs)

        assert "sentences" in result
        assert len(result["sentences"]) == 2
        # Verify apply_chat_template was called twice (once for each prompt)
        assert mock_tokenizer_with_chat_template.apply_chat_template.call_count == 2
        # Verify forward was called with both formatted prompts
        call_kwargs = exporter_with_chat_template.forward.call_args[1]
        assert call_kwargs["input_texts"] == [
            "<|begin_of_text|>user: Hello\nassistant:",
            "<|begin_of_text|>user: Goodbye\nassistant:",
        ]
