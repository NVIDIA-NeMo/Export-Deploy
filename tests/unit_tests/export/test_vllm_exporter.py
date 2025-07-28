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
    exporter.export(
        model_path_id=model_path, trust_remote_code=True, tensor_parallel_size=2, dtype="float16"
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

