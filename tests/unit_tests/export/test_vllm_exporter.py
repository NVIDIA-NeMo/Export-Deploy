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
def test_init(exporter):
    """Test initialization of vLLMExporter"""
    assert exporter.model is None
    assert exporter.lora_models is None


@pytest.mark.skipif(not HAVE_VLLM, reason="Need to enable virtual environment for vLLM")
@pytest.mark.run_only_on("GPU")
def test_export(exporter, mock_llm):
    """Test export method"""
    model_path = "/path/to/model"
    exporter.export(model_path_id=model_path)

    assert exporter.model is not None
    mock_llm.assert_called_once_with(
        model=model_path,
        tokenizer=None,
        trust_remote_code=False,
        enable_lora=False,
        tensor_parallel_size=1,
        dtype="auto",
        quantization=None,
        seed=0,
        gpu_memory_utilization=0.9,
        swap_space=4,
        cpu_offload_gb=0,
        enforce_eager=False,
        max_seq_len_to_capture=8192,
        task="auto",
    )


@pytest.mark.skipif(not HAVE_VLLM, reason="Need to enable virtual environment for vLLM")
@pytest.mark.run_only_on("GPU")
def test_export_with_lora(exporter, mock_llm):
    """Test export method with LoRA enabled"""
    model_path = "/path/to/model"
    exporter.export(model_path_id=model_path, enable_lora=True)

    assert exporter.model is not None
    mock_llm.assert_called_once_with(
        model=model_path,
        tokenizer=None,
        trust_remote_code=False,
        enable_lora=True,
        tensor_parallel_size=1,
        dtype="auto",
        quantization=None,
        seed=0,
        gpu_memory_utilization=0.9,
        swap_space=4,
        cpu_offload_gb=0,
        enforce_eager=False,
        max_seq_len_to_capture=8192,
        task="auto",
    )


@pytest.mark.skipif(not HAVE_VLLM, reason="Need to enable virtual environment for vLLM")
@pytest.mark.run_only_on("GPU")
def test_export_with_custom_params(exporter, mock_llm):
    """Test export method with custom parameters"""
    model_path = "/path/to/model"
    exporter.export(model_path_id=model_path, trust_remote_code=True, tensor_parallel_size=2, dtype="float16")

    assert exporter.model is not None
    mock_llm.assert_called_once_with(
        model=model_path,
        tokenizer=None,
        trust_remote_code=True,
        enable_lora=False,
        tensor_parallel_size=2,
        dtype="float16",
        quantization=None,
        seed=0,
        gpu_memory_utilization=0.9,
        swap_space=4,
        cpu_offload_gb=0,
        enforce_eager=False,
        max_seq_len_to_capture=8192,
        task="auto",
    )


@pytest.mark.skipif(not HAVE_VLLM, reason="Need to enable virtual environment for vLLM")
@pytest.mark.run_only_on("GPU")
def test_add_lora_models(exporter):
    """Test adding LoRA models"""
    lora_name = "test_lora"
    lora_model = "path/to/lora"

    exporter.add_lora_models(lora_name, lora_model)

    assert exporter.lora_models is not None
    assert lora_name in exporter.lora_models
    assert exporter.lora_models[lora_name] == lora_model


@pytest.mark.skipif(not HAVE_VLLM, reason="Need to enable virtual environment for vLLM")
@pytest.mark.run_only_on("GPU")
def test_get_triton_input(exporter):
    """Test triton input configuration"""
    inputs = exporter.get_triton_input

    # Check that we have all expected inputs
    input_names = [tensor.name for tensor in inputs]
    expected_inputs = [
        "prompts",
        "max_tokens",
        "min_tokens",
        "top_k",
        "top_p",
        "temperature",
        "seed",
        "n_log_probs",
        "n_prompt_log_probs",
    ]

    for expected_input in expected_inputs:
        assert expected_input in input_names, f"Missing expected input: {expected_input}"

    # Check data types and optionality
    for tensor in inputs:
        if tensor.name == "prompts":
            assert tensor.dtype == bytes
            assert not tensor.optional
        elif tensor.name in ["max_tokens", "min_tokens", "top_k", "seed", "n_log_probs", "n_prompt_log_probs"]:
            assert tensor.dtype == np.int_
            assert tensor.optional if tensor.name != "prompts" else not tensor.optional
        elif tensor.name in ["top_p", "temperature"]:
            assert tensor.dtype == np.single
            assert tensor.optional


@pytest.mark.skipif(not HAVE_VLLM, reason="Need to enable virtual environment for vLLM")
@pytest.mark.run_only_on("GPU")
def test_get_triton_output(exporter):
    """Test triton output configuration"""
    outputs = exporter.get_triton_output

    assert len(outputs) == 3
    output_names = [tensor.name for tensor in outputs]
    expected_outputs = ["sentences", "log_probs", "prompt_log_probs"]

    for expected_output in expected_outputs:
        assert expected_output in output_names, f"Missing expected output: {expected_output}"

    # All outputs should be bytes
    for tensor in outputs:
        assert tensor.dtype == bytes


@pytest.mark.skipif(not HAVE_VLLM, reason="Need to enable virtual environment for vLLM")
@pytest.mark.run_only_on("GPU")
def test_forward_without_model(exporter):
    """Test forward method without initialized model"""
    with pytest.raises(AssertionError, match="Model is not initialized"):
        exporter.forward(["test prompt"])


@pytest.mark.skipif(not HAVE_VLLM, reason="Need to enable virtual environment for vLLM")
@pytest.mark.run_only_on("GPU")
def test_forward_with_lora_not_added(exporter, mock_llm):
    """Test forward method with non-existent LoRA model"""
    exporter.export(model_path_id="/path/to/model")

    with pytest.raises(Exception, match="No lora models are available"):
        exporter.forward(["test prompt"], lora_model_name="non_existent_lora")


@pytest.mark.skipif(not HAVE_VLLM, reason="Need to enable virtual environment for vLLM")
@pytest.mark.run_only_on("GPU")
def test_forward_with_invalid_lora(exporter, mock_llm):
    """Test forward method with invalid LoRA model name"""
    exporter.export(model_path_id="/path/to/model")
    exporter.add_lora_models("valid_lora", "path/to/lora")

    with pytest.raises(AssertionError, match="Lora model was not added before"):
        exporter.forward(["test prompt"], lora_model_name="invalid_lora")


@pytest.mark.skipif(not HAVE_VLLM, reason="Need to enable virtual environment for vLLM")
@pytest.mark.run_only_on("GPU")
def test_forward_basic_usage(exporter, mock_llm):
    """Test basic forward method usage"""
    # Setup mock
    mock_output = MagicMock()
    mock_output.outputs = [MagicMock(text="test output", logprobs=None)]
    mock_output.prompt_logprobs = None
    mock_llm.return_value.generate.return_value = [mock_output]

    exporter.export(model_path_id="/path/to/model")

    result = exporter.forward(["test prompt"])

    assert "sentences" in result
    assert result["sentences"] == ["test output"]


@pytest.mark.skipif(not HAVE_VLLM, reason="Need to enable virtual environment for vLLM")
@pytest.mark.run_only_on("GPU")
def test_forward_with_all_params(exporter, mock_llm):
    """Test forward method with all parameters specified"""
    with (
        patch("nemo_export.vllm_exporter.SamplingParams") as mock_sampling_params,
        patch("nemo_export.vllm_exporter.LoRARequest") as mock_lora_request,
    ):
        # Setup mock outputs with log probabilities
        mock_logprob_value = MagicMock()
        mock_logprob_value.rank = 1
        mock_logprob_value.decoded_token = "test_token"
        mock_logprob_value.logprob = -0.5

        mock_logprob_dict = MagicMock()
        mock_logprob_dict.values.return_value = [mock_logprob_value]

        mock_prompt_logprob_value = MagicMock()
        mock_prompt_logprob_value.rank = 1
        mock_prompt_logprob_value.decoded_token = "prompt_token"
        mock_prompt_logprob_value.logprob = -0.3

        mock_prompt_logprob_dict = MagicMock()
        mock_prompt_logprob_dict.values.return_value = [mock_prompt_logprob_value]

        mock_output = MagicMock()
        mock_output.outputs = [MagicMock(text="Generated text response", logprobs=[mock_logprob_dict])]
        mock_output.prompt_logprobs = [mock_prompt_logprob_dict]

        mock_llm.return_value.generate.return_value = [mock_output]

        # Setup LoRA model
        exporter.export(model_path_id="/path/to/model")
        exporter.add_lora_models("test_lora", "/path/to/lora")

        # Call forward with all parameters
        input_texts = ["Test prompt 1", "Test prompt 2"]
        result = exporter.forward(
            input_texts=input_texts,
            max_tokens=50,
            min_tokens=5,
            top_k=10,
            top_p=0.95,
            temperature=0.8,
            n_log_probs=5,
            n_prompt_log_probs=3,
            seed=42,
            lora_model_name="test_lora",
        )

        # Verify SamplingParams was called with correct parameters
        mock_sampling_params.assert_called_once_with(
            max_tokens=50,
            min_tokens=5,
            logprobs=5,
            prompt_logprobs=3,
            seed=42,
            temperature=0.8,
            top_k=10,
            top_p=0.95,
        )

        # Verify LoRARequest was called correctly
        mock_lora_request.assert_called_once_with("test_lora", 1, "/path/to/lora")

        # Verify model.generate was called with correct arguments
        mock_llm.return_value.generate.assert_called_once()
        call_args = mock_llm.return_value.generate.call_args
        assert call_args[0][0] == input_texts  # input_texts
        assert call_args[1]["lora_request"] == mock_lora_request.return_value

        # Verify output structure
        assert "sentences" in result
        assert "log_probs" in result
        assert "prompt_log_probs" in result
        assert result["sentences"] == ["Generated text response"]

        # Verify log probabilities are properly formatted
        assert isinstance(result["log_probs"], np.ndarray)
        assert isinstance(result["prompt_log_probs"], np.ndarray)


@pytest.mark.skipif(not HAVE_VLLM, reason="Need to enable virtual environment for vLLM")
@pytest.mark.run_only_on("GPU")
def test_forward_with_multiple_outputs_and_logprobs(exporter, mock_llm):
    """Test forward method with multiple outputs and varying logprob lengths"""
    with patch("nemo_export.vllm_exporter.SamplingParams"):
        # Setup mock outputs with different logprob lengths to test padding
        mock_logprob_value1 = MagicMock()
        mock_logprob_value1.rank = 1
        mock_logprob_value1.decoded_token = "token1"
        mock_logprob_value1.logprob = -0.1

        mock_logprob_value2 = MagicMock()
        mock_logprob_value2.rank = 2
        mock_logprob_value2.decoded_token = "token2"
        mock_logprob_value2.logprob = -0.2

        mock_logprob_dict1 = MagicMock()
        mock_logprob_dict1.values.return_value = [mock_logprob_value1]

        mock_logprob_dict2 = MagicMock()
        mock_logprob_dict2.values.return_value = [mock_logprob_value2]

        # First output with 2 logprobs
        mock_output1 = MagicMock()
        mock_output1.outputs = [MagicMock(text="Output 1", logprobs=[mock_logprob_dict1, mock_logprob_dict2])]
        mock_output1.prompt_logprobs = [mock_logprob_dict1]

        # Second output with 1 logprob (will be padded)
        mock_output2 = MagicMock()
        mock_output2.outputs = [MagicMock(text="Output 2", logprobs=[mock_logprob_dict1])]
        mock_output2.prompt_logprobs = [mock_logprob_dict1, mock_logprob_dict2]

        mock_llm.return_value.generate.return_value = [mock_output1, mock_output2]

        exporter.export(model_path_id="/path/to/model")

        result = exporter.forward(
            input_texts=["Prompt 1", "Prompt 2"], max_tokens=30, n_log_probs=2, n_prompt_log_probs=2
        )

        # Verify output structure
        assert "sentences" in result
        assert "log_probs" in result
        assert "prompt_log_probs" in result
        assert len(result["sentences"]) == 2
        assert result["sentences"] == ["Output 1", "Output 2"]

        # Verify log probabilities arrays are properly padded
        assert result["log_probs"].shape == (2, 2)  # 2 outputs, max 2 logprobs


@pytest.mark.skipif(not HAVE_VLLM, reason="Need to enable virtual environment for vLLM")
@pytest.mark.run_only_on("GPU")
def test_forward_no_logprobs(exporter, mock_llm):
    """Test forward method when log probabilities are not requested"""
    with patch("nemo_export.vllm_exporter.SamplingParams") as mock_sampling_params:
        mock_output = MagicMock()
        mock_output.outputs = [MagicMock(text="No logprobs output", logprobs=None)]
        mock_output.prompt_logprobs = None

        mock_llm.return_value.generate.return_value = [mock_output]

        exporter.export(model_path_id="/path/to/model")

        result = exporter.forward(
            input_texts=["Test prompt"],
            max_tokens=20,
            # n_log_probs and n_prompt_log_probs are None (default)
        )

        # Verify SamplingParams called with None for logprobs
        mock_sampling_params.assert_called_once_with(
            max_tokens=20,
            min_tokens=0,
            logprobs=None,
            prompt_logprobs=None,
            seed=None,
            temperature=1.0,
            top_k=1,
            top_p=0.1,
        )

        # Verify output structure (no log probabilities)
        assert "sentences" in result
        assert "log_probs" not in result
        assert "prompt_log_probs" not in result
        assert result["sentences"] == ["No logprobs output"]


@pytest.mark.skipif(not HAVE_VLLM, reason="Need to enable virtual environment for vLLM")
@pytest.mark.run_only_on("GPU")
def test_dict_to_str_method(exporter):
    """Test the _dict_to_str utility method"""
    test_dict = {"key1": "value1", "key2": 42, "key3": [1, 2, 3]}
    result = exporter._dict_to_str(test_dict)

    import json

    expected = json.dumps(test_dict)
    assert result == expected
    assert isinstance(result, str)
