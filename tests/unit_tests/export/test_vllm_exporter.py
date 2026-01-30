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
    exporter.export(model_path_id=model_path, model_format="hf")

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
        enforce_eager=True,
        task="auto",
    )


@pytest.mark.skipif(not HAVE_VLLM, reason="Need to enable virtual environment for vLLM")
@pytest.mark.run_only_on("GPU")
def test_export_with_lora(exporter, mock_llm):
    """Test export method with LoRA enabled"""
    model_path = "/path/to/model"
    exporter.export(model_path_id=model_path, enable_lora=True, model_format="hf")

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
        enforce_eager=True,
        task="auto",
    )


@pytest.mark.skipif(not HAVE_VLLM, reason="Need to enable virtual environment for vLLM")
@pytest.mark.run_only_on("GPU")
def test_export_with_custom_params(exporter, mock_llm):
    """Test export method with custom parameters"""
    model_path = "/path/to/model"
    exporter.export(
        model_path_id=model_path,
        trust_remote_code=True,
        tensor_parallel_size=2,
        dtype="float16",
        model_format="hf",
    )

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
        enforce_eager=True,
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
    exporter.export(model_path_id="/path/to/model", model_format="hf")

    with pytest.raises(Exception, match="No lora models are available"):
        exporter.forward(["test prompt"], lora_model_name="non_existent_lora")


@pytest.mark.skipif(not HAVE_VLLM, reason="Need to enable virtual environment for vLLM")
@pytest.mark.run_only_on("GPU")
def test_forward_with_invalid_lora(exporter, mock_llm):
    """Test forward method with invalid LoRA model name"""
    exporter.export(model_path_id="/path/to/model", model_format="hf")
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

    exporter.export(model_path_id="/path/to/model", model_format="hf")

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
        exporter.export(model_path_id="/path/to/model", model_format="hf")
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

        exporter.export(model_path_id="/path/to/model", model_format="hf")

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

        exporter.export(model_path_id="/path/to/model", model_format="hf")

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
def test_ray_infer_fn_with_error_handling(exporter, mock_llm):
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


# ============================================================================
# Megatron Checkpoint Tests
# ============================================================================


@pytest.mark.skipif(not HAVE_VLLM, reason="Need to enable virtual environment for vLLM")
@pytest.mark.run_only_on("GPU")
def test_export_megatron_bridge_not_available(exporter):
    """Test export with megatron_bridge format when megatron.bridge is not installed"""
    with patch("nemo_export.vllm_exporter.HAVE_MEGATRON_BRIDGE", False):
        with pytest.raises(Exception, match="Megatron-Bridge is not available"):
            exporter.export(model_path_id="/path/to/megatron/checkpoint", model_format="megatron_bridge")


@pytest.mark.skipif(not HAVE_VLLM, reason="Need to enable virtual environment for vLLM")
@pytest.mark.run_only_on("GPU")
def test_export_megatron_bridge_no_hf_model_id(exporter, mock_llm):
    """Test export with megatron_bridge when hf_model_id cannot be extracted and not provided"""
    # Create mock AutoBridge class
    mock_auto_bridge = MagicMock()
    mock_auto_bridge.get_hf_model_id_from_checkpoint.return_value = None

    with (
        patch("nemo_export.vllm_exporter.HAVE_MEGATRON_BRIDGE", True),
        patch("nemo_export.vllm_exporter.AutoBridge", mock_auto_bridge, create=True),
    ):
        with pytest.raises(
            Exception,
            match="Could not find HuggingFace model ID in Megatron-Bridge checkpoint metadata",
        ):
            exporter.export(model_path_id="/path/to/megatron/checkpoint", model_format="megatron_bridge")


@pytest.mark.skipif(not HAVE_VLLM, reason="Need to enable virtual environment for vLLM")
@pytest.mark.run_only_on("GPU")
def test_export_megatron_bridge_with_explicit_hf_model_id(exporter, mock_llm):
    """Test export with megatron_bridge when hf_model_id is provided explicitly"""
    # Create mock classes
    mock_auto_bridge = MagicMock()
    mock_auto_config = MagicMock()

    mock_auto_bridge.get_hf_model_id_from_checkpoint.return_value = None
    mock_config = MagicMock()
    mock_auto_config.from_pretrained.return_value = mock_config
    mock_auto_bridge.supports.return_value = True

    mock_bridge_instance = MagicMock()
    mock_auto_bridge.from_hf_pretrained.return_value = mock_bridge_instance

    with (
        patch("nemo_export.vllm_exporter.HAVE_MEGATRON_BRIDGE", True),
        patch("nemo_export.vllm_exporter.AutoBridge", mock_auto_bridge, create=True),
        patch("nemo_export.vllm_exporter.AutoConfig", mock_auto_config, create=True),
        patch("nemo_export.vllm_exporter.tempfile.TemporaryDirectory") as mock_temp_dir,
        patch("nemo_export.vllm_exporter.Path") as mock_path,
    ):
        # Mock temp directory
        mock_temp_dir.return_value.__enter__.return_value = "/tmp/test_hf_export"
        mock_path_instance = MagicMock()
        mock_path_instance.iterdir.return_value = ["model.safetensors", "config.json"]  # Non-empty
        mock_path.return_value = mock_path_instance

        # Test export with explicit hf_model_id
        exporter.export(
            model_path_id="/path/to/megatron/checkpoint",
            model_format="megatron_bridge",
            hf_model_id="meta-llama/Llama-3-8B",
        )

        # Verify AutoConfig was called with provided hf_model_id
        mock_auto_config.from_pretrained.assert_called_once_with("meta-llama/Llama-3-8B", trust_remote_code=False)

        # Verify AutoBridge.from_hf_pretrained was called
        mock_auto_bridge.from_hf_pretrained.assert_called_once_with("meta-llama/Llama-3-8B", trust_remote_code=False)

        # Verify export_ckpt was called
        mock_bridge_instance.export_ckpt.assert_called_once()
        call_kwargs = mock_bridge_instance.export_ckpt.call_args[1]
        assert call_kwargs["megatron_path"] == "/path/to/megatron/checkpoint"
        assert call_kwargs["hf_path"] == "/tmp/test_hf_export"
        assert call_kwargs["show_progress"] is True
        assert call_kwargs["source_path"] == "meta-llama/Llama-3-8B"

        # Verify LLM was initialized with temp directory
        assert exporter.model is not None


@pytest.mark.skipif(not HAVE_VLLM, reason="Need to enable virtual environment for vLLM")
@pytest.mark.run_only_on("GPU")
def test_export_megatron_bridge_with_extracted_hf_model_id(exporter, mock_llm):
    """Test export with megatron_bridge when hf_model_id is extracted from checkpoint"""
    # Create mock classes
    mock_auto_bridge = MagicMock()
    mock_auto_config = MagicMock()

    # Setup mocks - AutoBridge extracts model ID from checkpoint
    mock_auto_bridge.get_hf_model_id_from_checkpoint.return_value = "meta-llama/Llama-3-8B"
    mock_config = MagicMock()
    mock_auto_config.from_pretrained.return_value = mock_config
    mock_auto_bridge.supports.return_value = True

    mock_bridge_instance = MagicMock()
    mock_auto_bridge.from_hf_pretrained.return_value = mock_bridge_instance

    with (
        patch("nemo_export.vllm_exporter.HAVE_MEGATRON_BRIDGE", True),
        patch("nemo_export.vllm_exporter.AutoBridge", mock_auto_bridge, create=True),
        patch("nemo_export.vllm_exporter.AutoConfig", mock_auto_config, create=True),
        patch("nemo_export.vllm_exporter.tempfile.TemporaryDirectory") as mock_temp_dir,
        patch("nemo_export.vllm_exporter.Path") as mock_path,
    ):
        # Mock temp directory
        mock_temp_dir.return_value.__enter__.return_value = "/tmp/test_hf_export"
        mock_path_instance = MagicMock()
        mock_path_instance.iterdir.return_value = ["model.safetensors", "config.json"]  # Non-empty
        mock_path.return_value = mock_path_instance

        # Test export without explicit hf_model_id (will be extracted)
        exporter.export(model_path_id="/path/to/megatron/checkpoint", model_format="megatron_bridge")

        # Verify hf_model_id was extracted from checkpoint
        mock_auto_bridge.get_hf_model_id_from_checkpoint.assert_called_once_with("/path/to/megatron/checkpoint")

        # Verify AutoConfig was called with extracted hf_model_id
        mock_auto_config.from_pretrained.assert_called_once_with("meta-llama/Llama-3-8B", trust_remote_code=False)

        # Verify AutoBridge.from_hf_pretrained was called with extracted ID
        mock_auto_bridge.from_hf_pretrained.assert_called_once_with("meta-llama/Llama-3-8B", trust_remote_code=False)


@pytest.mark.skipif(not HAVE_VLLM, reason="Need to enable virtual environment for vLLM")
@pytest.mark.run_only_on("GPU")
def test_export_megatron_bridge_model_not_supported(exporter, mock_llm):
    """Test export with megatron_bridge when model is not supported by AutoBridge"""
    # Create mock classes
    mock_auto_bridge = MagicMock()
    mock_auto_config = MagicMock()

    # Setup mocks
    mock_auto_bridge.get_hf_model_id_from_checkpoint.return_value = "unsupported/model"
    mock_config = MagicMock()
    mock_auto_config.from_pretrained.return_value = mock_config
    mock_auto_bridge.supports.return_value = False
    mock_auto_bridge.list_supported_models.return_value = [
        "LlamaForCausalLM",
        "MistralForCausalLM",
        "GPT2ForCausalLM",
    ]

    with (
        patch("nemo_export.vllm_exporter.HAVE_MEGATRON_BRIDGE", True),
        patch("nemo_export.vllm_exporter.AutoBridge", mock_auto_bridge, create=True),
        patch("nemo_export.vllm_exporter.AutoConfig", mock_auto_config, create=True),
    ):
        with pytest.raises(Exception, match="Model 'unsupported/model' is not supported by AutoBridge"):
            exporter.export(model_path_id="/path/to/megatron/checkpoint", model_format="megatron_bridge")


@pytest.mark.skipif(not HAVE_VLLM, reason="Need to enable virtual environment for vLLM")
@pytest.mark.run_only_on("GPU")
def test_export_megatron_bridge_conversion_failed_empty_dir(exporter, mock_llm):
    """Test export with megatron_bridge when checkpoint conversion results in empty directory"""
    # Create mock classes
    mock_auto_bridge = MagicMock()
    mock_auto_config = MagicMock()

    # Setup mocks
    mock_auto_bridge.get_hf_model_id_from_checkpoint.return_value = "meta-llama/Llama-3-8B"
    mock_config = MagicMock()
    mock_auto_config.from_pretrained.return_value = mock_config
    mock_auto_bridge.supports.return_value = True

    mock_bridge_instance = MagicMock()
    mock_auto_bridge.from_hf_pretrained.return_value = mock_bridge_instance

    with (
        patch("nemo_export.vllm_exporter.HAVE_MEGATRON_BRIDGE", True),
        patch("nemo_export.vllm_exporter.AutoBridge", mock_auto_bridge, create=True),
        patch("nemo_export.vllm_exporter.AutoConfig", mock_auto_config, create=True),
        patch("nemo_export.vllm_exporter.tempfile.TemporaryDirectory") as mock_temp_dir,
        patch("nemo_export.vllm_exporter.Path") as mock_path,
    ):
        # Mock temp directory - but empty after conversion
        mock_temp_dir.return_value.__enter__.return_value = "/tmp/test_hf_export"
        mock_path_instance = MagicMock()
        mock_path_instance.iterdir.return_value = []  # Empty directory
        mock_path.return_value = mock_path_instance

        with pytest.raises(
            Exception,
            match="Megatron-Bridge checkpoint conversion failed.*Error occurred during Hugging Face conversion",
        ):
            exporter.export(model_path_id="/path/to/megatron/checkpoint", model_format="megatron_bridge")


@pytest.mark.skipif(not HAVE_VLLM, reason="Need to enable virtual environment for vLLM")
@pytest.mark.run_only_on("GPU")
def test_export_megatron_bridge_with_trust_remote_code(exporter, mock_llm):
    """Test export with megatron_bridge using trust_remote_code option"""
    # Create mock classes
    mock_auto_bridge = MagicMock()
    mock_auto_config = MagicMock()

    # Setup mocks
    mock_auto_bridge.get_hf_model_id_from_checkpoint.return_value = "meta-llama/Llama-3-8B"
    mock_config = MagicMock()
    mock_auto_config.from_pretrained.return_value = mock_config
    mock_auto_bridge.supports.return_value = True

    mock_bridge_instance = MagicMock()
    mock_auto_bridge.from_hf_pretrained.return_value = mock_bridge_instance

    with (
        patch("nemo_export.vllm_exporter.HAVE_MEGATRON_BRIDGE", True),
        patch("nemo_export.vllm_exporter.AutoBridge", mock_auto_bridge, create=True),
        patch("nemo_export.vllm_exporter.AutoConfig", mock_auto_config, create=True),
        patch("nemo_export.vllm_exporter.tempfile.TemporaryDirectory") as mock_temp_dir,
        patch("nemo_export.vllm_exporter.Path") as mock_path,
    ):
        # Mock temp directory
        mock_temp_dir.return_value.__enter__.return_value = "/tmp/test_hf_export"
        mock_path_instance = MagicMock()
        mock_path_instance.iterdir.return_value = ["model.safetensors"]
        mock_path.return_value = mock_path_instance

        # Test export with trust_remote_code=True
        exporter.export(
            model_path_id="/path/to/megatron/checkpoint",
            model_format="megatron_bridge",
            trust_remote_code=True,
        )

        # Verify AutoConfig was called with trust_remote_code=True
        mock_auto_config.from_pretrained.assert_called_once_with("meta-llama/Llama-3-8B", trust_remote_code=True)

        # Verify AutoBridge.from_hf_pretrained was called with trust_remote_code=True
        mock_auto_bridge.from_hf_pretrained.assert_called_once_with("meta-llama/Llama-3-8B", trust_remote_code=True)


@pytest.mark.skipif(not HAVE_VLLM, reason="Need to enable virtual environment for vLLM")
@pytest.mark.run_only_on("GPU")
def test_export_megatron_bridge_full_success_path(exporter, mock_llm):
    """Test complete successful export path for megatron_bridge checkpoint"""
    # Create mock classes
    mock_auto_bridge = MagicMock()
    mock_auto_config = MagicMock()

    # Setup mocks
    mock_auto_bridge.get_hf_model_id_from_checkpoint.return_value = "meta-llama/Llama-3-8B"
    mock_config = MagicMock()
    mock_auto_config.from_pretrained.return_value = mock_config
    mock_auto_bridge.supports.return_value = True

    mock_bridge_instance = MagicMock()
    mock_auto_bridge.from_hf_pretrained.return_value = mock_bridge_instance

    with (
        patch("nemo_export.vllm_exporter.HAVE_MEGATRON_BRIDGE", True),
        patch("nemo_export.vllm_exporter.AutoBridge", mock_auto_bridge, create=True),
        patch("nemo_export.vllm_exporter.AutoConfig", mock_auto_config, create=True),
        patch("nemo_export.vllm_exporter.tempfile.TemporaryDirectory") as mock_temp_dir,
        patch("nemo_export.vllm_exporter.Path") as mock_path,
    ):
        # Mock temp directory with successful conversion
        mock_temp_dir.return_value.__enter__.return_value = "/tmp/test_hf_export"
        mock_path_instance = MagicMock()
        mock_path_instance.iterdir.return_value = [
            "model.safetensors",
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
        ]
        mock_path.return_value = mock_path_instance

        # Test complete export with all parameters
        exporter.export(
            model_path_id="/path/to/megatron/checkpoint",
            model_format="megatron_bridge",
            tensor_parallel_size=2,
            dtype="bfloat16",
            gpu_memory_utilization=0.8,
        )

        # Verify complete flow
        mock_auto_bridge.get_hf_model_id_from_checkpoint.assert_called_once_with("/path/to/megatron/checkpoint")
        mock_auto_config.from_pretrained.assert_called_once()
        mock_auto_bridge.supports.assert_called_once_with(mock_config)
        mock_auto_bridge.from_hf_pretrained.assert_called_once()
        mock_bridge_instance.export_ckpt.assert_called_once()

        # Verify LLM was initialized with correct parameters
        mock_llm.assert_called_once()
        call_kwargs = mock_llm.call_args[1]
        assert call_kwargs["model"] == "/tmp/test_hf_export"
        assert call_kwargs["tensor_parallel_size"] == 2
        assert call_kwargs["dtype"] == "bfloat16"
        assert call_kwargs["gpu_memory_utilization"] == 0.8

        # Verify model was set
        assert exporter.model is not None


@pytest.mark.skipif(not HAVE_VLLM, reason="Need to enable virtual environment for vLLM")
@pytest.mark.run_only_on("GPU")
def test_export_megatron_bridge_with_all_vllm_params(exporter, mock_llm):
    """Test export with megatron_bridge passing all vLLM parameters"""
    # Create mock classes
    mock_auto_bridge = MagicMock()
    mock_auto_config = MagicMock()

    # Setup mocks
    mock_auto_bridge.get_hf_model_id_from_checkpoint.return_value = "meta-llama/Llama-3-8B"
    mock_config = MagicMock()
    mock_auto_config.from_pretrained.return_value = mock_config
    mock_auto_bridge.supports.return_value = True

    mock_bridge_instance = MagicMock()
    mock_auto_bridge.from_hf_pretrained.return_value = mock_bridge_instance

    with (
        patch("nemo_export.vllm_exporter.HAVE_MEGATRON_BRIDGE", True),
        patch("nemo_export.vllm_exporter.AutoBridge", mock_auto_bridge, create=True),
        patch("nemo_export.vllm_exporter.AutoConfig", mock_auto_config, create=True),
        patch("nemo_export.vllm_exporter.tempfile.TemporaryDirectory") as mock_temp_dir,
        patch("nemo_export.vllm_exporter.Path") as mock_path,
    ):
        # Mock temp directory
        mock_temp_dir.return_value.__enter__.return_value = "/tmp/test_hf_export"
        mock_path_instance = MagicMock()
        mock_path_instance.iterdir.return_value = ["model.safetensors"]
        mock_path.return_value = mock_path_instance

        # Test with all vLLM parameters
        exporter.export(
            model_path_id="/path/to/megatron/checkpoint",
            model_format="megatron_bridge",
            tokenizer="/path/to/tokenizer",
            trust_remote_code=True,
            enable_lora=True,
            tensor_parallel_size=4,
            dtype="float16",
            quantization="awq",
            seed=42,
            gpu_memory_utilization=0.85,
            swap_space=8,
            cpu_offload_gb=2,
            enforce_eager=True,
            task="generate",
        )

        # Verify LLM was called with all parameters
        mock_llm.assert_called_once()
        call_kwargs = mock_llm.call_args[1]
        assert call_kwargs["model"] == "/tmp/test_hf_export"
        assert call_kwargs["tokenizer"] == "/path/to/tokenizer"
        assert call_kwargs["trust_remote_code"] is True
        assert call_kwargs["enable_lora"] is True
        assert call_kwargs["tensor_parallel_size"] == 4
        assert call_kwargs["dtype"] == "float16"
        assert call_kwargs["quantization"] == "awq"
        assert call_kwargs["seed"] == 42
        assert call_kwargs["gpu_memory_utilization"] == 0.85
        assert call_kwargs["swap_space"] == 8
        assert call_kwargs["cpu_offload_gb"] == 2
        assert call_kwargs["enforce_eager"] is True
        assert call_kwargs["task"] == "generate"
