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
