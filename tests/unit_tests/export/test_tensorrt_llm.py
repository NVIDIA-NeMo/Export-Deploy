# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import os
import re
from unittest.mock import patch

import pytest
import torch
import numpy as np


@pytest.mark.run_only_on("GPU")
@pytest.mark.unit
def test_get_nemo_to_trtllm_conversion_dict_on_nemo_model():
    try:
        from nemo_export.tensorrt_llm import TensorRTLLM
    except ImportError:
        pytest.skip(
            "Could not import TRTLLM helpers. tensorrt_llm is likely not installed"
        )
        return

    dummy_state = object()
    model_state_dict = {
        "model.embedding.word_embeddings.weight": dummy_state,
        "model.decoder.layers.0.self_attention.linear_proj.weight": dummy_state,
    }
    nemo_model_conversion_dict = TensorRTLLM.get_nemo_to_trtllm_conversion_dict(
        model_state_dict
    )

    # Check that every key starts with 'model.' and not 'model..' by using a regex
    # This pattern ensures:
    #   - The key starts with 'model.'
    #   - Immediately after 'model.', there must be at least one character that is NOT a '.'
    #     (preventing the 'model..' scenario)
    pattern = re.compile(r"^model\.[^.].*")
    for key in nemo_model_conversion_dict.keys():
        assert pattern.match(key), f"Key '{key}' does not properly start with 'model.'"


@pytest.mark.run_only_on("GPU")
@pytest.mark.unit
def test_get_nemo_to_trtllm_conversion_dict_on_mcore_model():
    try:
        from megatron.core.export.trtllm.model_to_trllm_mapping.default_conversion_dict import (
            DEFAULT_CONVERSION_DICT,
        )

        from nemo_export.tensorrt_llm import TensorRTLLM
    except ImportError:
        pytest.skip(
            "Could not import TRTLLM helpers. tensorrt_llm is likely not installed"
        )
        return

    dummy_state = object()
    model_state_dict = {
        "embedding.word_embeddings.weight": dummy_state,
        "decoder.layers.0.self_attention.linear_proj.weight": dummy_state,
    }
    nemo_model_conversion_dict = TensorRTLLM.get_nemo_to_trtllm_conversion_dict(
        model_state_dict
    )

    # This is essentially a no-op
    assert nemo_model_conversion_dict == DEFAULT_CONVERSION_DICT


@pytest.mark.run_only_on("GPU")
@pytest.mark.unit
def test_tensorrt_llm_initialization():
    try:
        from nemo_export.tensorrt_llm import TensorRTLLM
    except ImportError:
        pytest.skip(
            "Could not import TRTLLM helpers. tensorrt_llm is likely not installed"
        )
        return

    # Test basic initialization
    model_dir = "/tmp/test_model_dir"
    trt_llm = TensorRTLLM(model_dir=model_dir, load_model=False)
    assert trt_llm.model_dir == model_dir
    assert trt_llm.engine_dir == os.path.join(model_dir, "trtllm_engine")
    assert trt_llm.model is None
    assert trt_llm.tokenizer is None
    assert trt_llm.config is None

    # Test initialization with lora checkpoints
    lora_ckpt_list = ["/path/to/lora1", "/path/to/lora2"]
    trt_llm = TensorRTLLM(
        model_dir=model_dir, lora_ckpt_list=lora_ckpt_list, load_model=False
    )
    assert trt_llm.lora_ckpt_list == lora_ckpt_list

    # Test initialization with python runtime options
    trt_llm = TensorRTLLM(
        model_dir=model_dir,
        use_python_runtime=False,
        enable_chunked_context=False,
        max_tokens_in_paged_kv_cache=None,
        load_model=False,
    )
    assert trt_llm.use_python_runtime is False
    assert trt_llm.enable_chunked_context is False
    assert trt_llm.max_tokens_in_paged_kv_cache is None


@pytest.mark.run_only_on("GPU")
@pytest.mark.unit
def test_tensorrt_llm_supported_models():
    try:
        from nemo_export.tensorrt_llm import TensorRTLLM
    except ImportError:
        pytest.skip(
            "Could not import TRTLLM helpers. tensorrt_llm is likely not installed"
        )
        return

    model_dir = "/tmp/test_model_dir"
    trt_llm = TensorRTLLM(model_dir=model_dir, load_model=False)

    # Test supported models list
    supported_models = trt_llm.get_supported_models_list
    assert isinstance(supported_models, list)
    assert len(supported_models) > 0
    assert all(isinstance(model, str) for model in supported_models)

    # Test HF model mapping
    hf_mapping = trt_llm.get_supported_hf_model_mapping
    assert isinstance(hf_mapping, dict)
    assert len(hf_mapping) > 0


@pytest.mark.run_only_on("GPU")
@pytest.mark.unit
def test_tensorrt_llm_input_dtype():
    try:
        from nemo_export.tensorrt_llm import TensorRTLLM
    except ImportError:
        pytest.skip(
            "Could not import TRTLLM helpers. tensorrt_llm is likely not installed"
        )
        return

    model_dir = "/tmp/test_model_dir"
    trt_llm = TensorRTLLM(model_dir=model_dir, load_model=False)

    from megatron.core.export.data_type import DataType

    # Test different storage dtypes
    test_cases = [
        (torch.float32, DataType.float32),
        (torch.float16, DataType.float16),
        (torch.bfloat16, DataType.bfloat16),
    ]

    for storage_dtype, expected_dtype in test_cases:
        input_dtype = trt_llm.get_input_dtype(storage_dtype)
        assert input_dtype == expected_dtype, (
            f"Expected {expected_dtype} for {storage_dtype}, got {input_dtype}"
        )


@pytest.mark.run_only_on("GPU")
@pytest.mark.unit
def test_tensorrt_llm_hidden_size():
    try:
        from nemo_export.tensorrt_llm import TensorRTLLM
    except ImportError:
        pytest.skip(
            "Could not import TRTLLM helpers. tensorrt_llm is likely not installed"
        )
        return

    model_dir = "/tmp/test_model_dir"
    trt_llm = TensorRTLLM(model_dir=model_dir, load_model=False)

    # Test hidden size property
    hidden_size = trt_llm.get_hidden_size
    if hidden_size is not None:
        assert isinstance(hidden_size, int)
        assert hidden_size > 0
    else:
        assert hidden_size is None


@pytest.mark.run_only_on("GPU")
@pytest.mark.unit
def test_tensorrt_llm_triton_io():
    try:
        from nemo_export.tensorrt_llm import TensorRTLLM
    except ImportError:
        pytest.skip(
            "Could not import TRTLLM helpers. tensorrt_llm is likely not installed"
        )
        return

    model_dir = "/tmp/test_model_dir"
    trt_llm = TensorRTLLM(model_dir=model_dir, load_model=False)

    # Test Triton input configuration
    triton_input = trt_llm.get_triton_input
    assert isinstance(triton_input, tuple)
    assert triton_input[0].name == "prompts"
    assert triton_input[1].name == "max_output_len"
    assert triton_input[2].name == "top_k"
    assert triton_input[3].name == "top_p"
    assert triton_input[4].name == "temperature"
    assert triton_input[5].name == "random_seed"
    assert triton_input[6].name == "stop_words_list"
    assert triton_input[7].name == "bad_words_list"
    assert triton_input[8].name == "no_repeat_ngram_size"

    # Test Triton output configuration
    triton_output = trt_llm.get_triton_output
    assert isinstance(triton_output, tuple)
    assert triton_output[0].name == "outputs"
    assert triton_output[1].name == "generation_logits"
    assert triton_output[2].name == "context_logits"


@pytest.mark.run_only_on("GPU")
@pytest.mark.unit
def test_tensorrt_llm_pad_logits():
    try:
        from nemo_export.tensorrt_llm import TensorRTLLM
    except ImportError:
        pytest.skip(
            "Could not import TRTLLM helpers. tensorrt_llm is likely not installed"
        )
        return

    model_dir = "/tmp/test_model_dir"
    trt_llm = TensorRTLLM(model_dir=model_dir, load_model=False)

    # Create a sample logits tensor
    batch_size = 2
    seq_len = 3
    vocab_size = 1000
    logits = torch.randn(batch_size, seq_len, vocab_size)

    # Test padding logits
    padded_logits = trt_llm._pad_logits(logits)
    assert isinstance(padded_logits, torch.Tensor)
    assert padded_logits.shape[0] == batch_size
    assert padded_logits.shape[1] == seq_len
    assert padded_logits.shape[2] >= vocab_size  # Should be padded to a multiple of 8


@pytest.mark.run_only_on("GPU")
@pytest.mark.unit
def test_ray_infer_fn_basic():
    """Test basic functionality of ray_infer_fn method."""
    try:
        from nemo_export.tensorrt_llm import TensorRTLLM
    except ImportError:
        pytest.skip(
            "Could not import TRTLLM helpers. tensorrt_llm is likely not installed"
        )
        return

    model_dir = "/tmp/test_model_dir"
    trt_llm = TensorRTLLM(model_dir=model_dir, load_model=False)

    # Mock the forward method
    with patch.object(trt_llm, "forward") as mock_forward:
        mock_forward.return_value = ["Generated text 1", "Generated text 2"]
        
        inputs = {
            "prompts": ["Hello", "World"],
            "max_output_len": 256,
            "temperature": 0.8,
            "top_k": 50,
            "top_p": 0.9,
        }
        
        result = trt_llm.ray_infer_fn(inputs)
        
        # Verify the result structure
        assert "sentences" in result
        assert result["sentences"] == ["Generated text 1", "Generated text 2"]
        
        # Verify forward was called with correct parameters
        mock_forward.assert_called_once()
        call_kwargs = mock_forward.call_args[1]
        assert call_kwargs["input_texts"] == ["Hello", "World"]
        assert call_kwargs["max_output_len"] == 256
        assert call_kwargs["temperature"] == 0.8
        assert call_kwargs["top_k"] == 50
        assert call_kwargs["top_p"] == 0.9


@pytest.mark.run_only_on("GPU")
@pytest.mark.unit
def test_ray_infer_fn_with_single_string_prompt():
    """Test ray_infer_fn method with a single string prompt (not in a list)."""
    try:
        from nemo_export.tensorrt_llm import TensorRTLLM
    except ImportError:
        pytest.skip(
            "Could not import TRTLLM helpers. tensorrt_llm is likely not installed"
        )
        return

    model_dir = "/tmp/test_model_dir"
    trt_llm = TensorRTLLM(model_dir=model_dir, load_model=False)

    # Mock the forward method
    with patch.object(trt_llm, "forward") as mock_forward:
        mock_forward.return_value = ["Generated response"]
        
        inputs = {
            "prompts": "Hello world",  # Single string instead of list
            "temperature": 1.0,
        }
        
        result = trt_llm.ray_infer_fn(inputs)
        
        # Verify the result
        assert result["sentences"] == ["Generated response"]
        
        # Verify forward was called with prompts converted to list
        mock_forward.assert_called_once()
        call_kwargs = mock_forward.call_args[1]
        assert call_kwargs["input_texts"] == ["Hello world"]


@pytest.mark.run_only_on("GPU")
@pytest.mark.unit
def test_ray_infer_fn_with_stop_words():
    """Test ray_infer_fn method with stop words list."""
    try:
        from nemo_export.tensorrt_llm import TensorRTLLM
    except ImportError:
        pytest.skip(
            "Could not import TRTLLM helpers. tensorrt_llm is likely not installed"
        )
        return

    model_dir = "/tmp/test_model_dir"
    trt_llm = TensorRTLLM(model_dir=model_dir, load_model=False)

    # Mock the forward method
    with patch.object(trt_llm, "forward") as mock_forward:
        mock_forward.return_value = ["Generated text"]
        
        inputs = {
            "prompts": ["Test prompt"],
            "stop_words_list": ["stop", "end"],
            "bad_words_list": ["bad", "word"],
        }
        
        result = trt_llm.ray_infer_fn(inputs)
        
        # Verify the result
        assert result["sentences"] == ["Generated text"]
        
        # Verify forward was called with properly formatted word lists
        mock_forward.assert_called_once()
        call_kwargs = mock_forward.call_args[1]
        assert call_kwargs["stop_words_list"] == [["stop"], ["end"]]
        assert call_kwargs["bad_words_list"] == [["bad"], ["word"]]


@pytest.mark.run_only_on("GPU")
@pytest.mark.unit
def test_ray_infer_fn_with_generation_logits():
    """Test ray_infer_fn method with generation logits enabled."""
    try:
        from nemo_export.tensorrt_llm import TensorRTLLM
    except ImportError:
        pytest.skip(
            "Could not import TRTLLM helpers. tensorrt_llm is likely not installed"
        )
        return

    model_dir = "/tmp/test_model_dir"
    trt_llm = TensorRTLLM(model_dir=model_dir, load_model=False)

    # Create mock logits tensors
    mock_logits = [torch.randn(1, 1, 10, 1000), torch.randn(1, 1, 15, 1000)]
    
    # Mock the forward method
    with patch.object(trt_llm, "forward") as mock_forward:
        mock_forward.return_value = (["Generated text 1", "Generated text 2"], mock_logits)
        
        inputs = {
            "prompts": ["Test prompt 1", "Test prompt 2"],
            "output_generation_logits": True,
        }
        
        result = trt_llm.ray_infer_fn(inputs)
        
        # Verify the result structure
        assert "sentences" in result
        assert "generation_logits" in result
        assert result["sentences"] == ["Generated text 1", "Generated text 2"]
        assert len(result["generation_logits"]) == 2
        assert isinstance(result["generation_logits"][0], np.ndarray)
        assert isinstance(result["generation_logits"][1], np.ndarray)
        
        # Verify forward was called with generation logits enabled
        mock_forward.assert_called_once()
        call_kwargs = mock_forward.call_args[1]
        assert call_kwargs["output_generation_logits"] is True


@pytest.mark.run_only_on("GPU")
@pytest.mark.unit
def test_ray_infer_fn_with_context_logits():
    """Test ray_infer_fn method with context logits enabled."""
    try:
        from nemo_export.tensorrt_llm import TensorRTLLM
    except ImportError:
        pytest.skip(
            "Could not import TRTLLM helpers. tensorrt_llm is likely not installed"
        )
        return

    model_dir = "/tmp/test_model_dir"
    trt_llm = TensorRTLLM(model_dir=model_dir, load_model=False)

    # Create mock context logits tensors
    mock_context_logits = [torch.randn(10, 1000), torch.randn(15, 1000)]
    
    # Mock the forward method and _pad_logits method
    with patch.object(trt_llm, "forward") as mock_forward, \
         patch.object(trt_llm, "_pad_logits") as mock_pad_logits:
        
        mock_forward.return_value = (["Generated text 1", "Generated text 2"], mock_context_logits)
        mock_pad_logits.return_value = mock_context_logits
        
        inputs = {
            "prompts": ["Test prompt 1", "Test prompt 2"],
            "output_context_logits": True,
        }
        
        result = trt_llm.ray_infer_fn(inputs)
        
        # Verify the result structure
        assert "sentences" in result
        assert "context_logits" in result
        assert result["sentences"] == ["Generated text 1", "Generated text 2"]
        assert len(result["context_logits"]) == 2
        assert isinstance(result["context_logits"][0], np.ndarray)
        assert isinstance(result["context_logits"][1], np.ndarray)
        
        # Verify forward was called with context logits enabled
        mock_forward.assert_called_once()
        call_kwargs = mock_forward.call_args[1]
        assert call_kwargs["output_context_logits"] is True
        
        # Verify _pad_logits was called
        mock_pad_logits.assert_called_once_with(mock_context_logits)


@pytest.mark.run_only_on("GPU")
@pytest.mark.unit
def test_ray_infer_fn_with_task_ids_and_lora():
    """Test ray_infer_fn method with task IDs and LoRA UIDs."""
    try:
        from nemo_export.tensorrt_llm import TensorRTLLM
    except ImportError:
        pytest.skip(
            "Could not import TRTLLM helpers. tensorrt_llm is likely not installed"
        )
        return

    model_dir = "/tmp/test_model_dir"
    trt_llm = TensorRTLLM(model_dir=model_dir, load_model=False)

    # Mock the forward method
    with patch.object(trt_llm, "forward") as mock_forward:
        mock_forward.return_value = ["Generated text with LoRA"]
        
        inputs = {
            "prompts": ["Test prompt"],
            "task_ids": ["task1"],
            "lora_uids": ["lora_uid_1"],
            "random_seed": 42,
            "no_repeat_ngram_size": 3,
        }
        
        result = trt_llm.ray_infer_fn(inputs)
        
        # Verify the result
        assert result["sentences"] == ["Generated text with LoRA"]
        
        # Verify forward was called with all parameters
        mock_forward.assert_called_once()
        call_kwargs = mock_forward.call_args[1]
        assert call_kwargs["task_ids"] == ["task1"]
        assert call_kwargs["lora_uids"] == ["lora_uid_1"]
        assert call_kwargs["random_seed"] == 42
        assert call_kwargs["no_repeat_ngram_size"] == 3


@pytest.mark.run_only_on("GPU")
@pytest.mark.unit
def test_ray_infer_fn_empty_prompts():
    """Test ray_infer_fn method with empty prompts."""
    try:
        from nemo_export.tensorrt_llm import TensorRTLLM
    except ImportError:
        pytest.skip(
            "Could not import TRTLLM helpers. tensorrt_llm is likely not installed"
        )
        return

    model_dir = "/tmp/test_model_dir"
    trt_llm = TensorRTLLM(model_dir=model_dir, load_model=False)

    # Mock the forward method
    with patch.object(trt_llm, "forward") as mock_forward:
        mock_forward.return_value = []
        
        inputs = {}  # No prompts provided
        
        result = trt_llm.ray_infer_fn(inputs)
        
        # Verify the result
        assert result["sentences"] == []
        
        # Verify forward was called with empty input_texts
        mock_forward.assert_called_once()
        call_kwargs = mock_forward.call_args[1]
        assert call_kwargs["input_texts"] == []


@pytest.mark.run_only_on("GPU")
@pytest.mark.unit
def test_ray_infer_fn_error_handling():
    """Test ray_infer_fn method error handling."""
    try:
        from nemo_export.tensorrt_llm import TensorRTLLM
    except ImportError:
        pytest.skip(
            "Could not import TRTLLM helpers. tensorrt_llm is likely not installed"
        )
        return

    model_dir = "/tmp/test_model_dir"
    trt_llm = TensorRTLLM(model_dir=model_dir, load_model=False)

    # Mock the forward method to raise an exception
    with patch.object(trt_llm, "forward") as mock_forward:
        mock_forward.side_effect = Exception("Model inference failed")
        
        inputs = {
            "prompts": ["Test prompt 1", "Test prompt 2"],
        }
        
        result = trt_llm.ray_infer_fn(inputs)
        
        # Verify error handling
        assert "sentences" in result
        assert "error" in result
        assert len(result["sentences"]) == 2  # Should match number of prompts
        assert all("An error occurred" in sentence for sentence in result["sentences"])
        assert "Model inference failed" in result["error"]


@pytest.mark.run_only_on("GPU")
@pytest.mark.unit
def test_ray_infer_fn_all_parameters():
    """Test ray_infer_fn method with all possible parameters."""
    try:
        from nemo_export.tensorrt_llm import TensorRTLLM
    except ImportError:
        pytest.skip(
            "Could not import TRTLLM helpers. tensorrt_llm is likely not installed"
        )
        return

    model_dir = "/tmp/test_model_dir"
    trt_llm = TensorRTLLM(model_dir=model_dir, load_model=False)

    # Mock the forward method
    with patch.object(trt_llm, "forward") as mock_forward:
        mock_forward.return_value = ["Comprehensive test response"]
        
        inputs = {
            "prompts": ["Comprehensive test prompt"],
            "max_output_len": 512,
            "top_k": 50,
            "top_p": 0.9,
            "temperature": 0.7,
            "random_seed": 123,
            "stop_words_list": [["stop"], ["end"]],  # Already in correct format
            "bad_words_list": [["bad"], ["inappropriate"]],  # Already in correct format
            "no_repeat_ngram_size": 4,
            "task_ids": ["comprehensive_task"],
            "lora_uids": ["comprehensive_lora"],
            "output_log_probs": True,
        }
        
        result = trt_llm.ray_infer_fn(inputs)
        
        # Verify the result
        assert result["sentences"] == ["Comprehensive test response"]
        
        # Verify forward was called with all parameters
        mock_forward.assert_called_once()
        call_kwargs = mock_forward.call_args[1]
        expected_params = [
            "input_texts", "max_output_len", "top_k", "top_p", "temperature",
            "random_seed", "stop_words_list", "bad_words_list", "no_repeat_ngram_size",
            "task_ids", "lora_uids", "output_log_probs"
        ]
        
        for param in expected_params:
            assert param in call_kwargs, f"Parameter {param} not found in forward call"
        
        # Verify specific values
        assert call_kwargs["input_texts"] == ["Comprehensive test prompt"]
        assert call_kwargs["max_output_len"] == 512
        assert call_kwargs["top_k"] == 50
        assert call_kwargs["top_p"] == 0.9
        assert call_kwargs["temperature"] == 0.7
        assert call_kwargs["random_seed"] == 123
        assert call_kwargs["stop_words_list"] == [["stop"], ["end"]]
        assert call_kwargs["bad_words_list"] == [["bad"], ["inappropriate"]]
        assert call_kwargs["no_repeat_ngram_size"] == 4
        assert call_kwargs["task_ids"] == ["comprehensive_task"]
        assert call_kwargs["lora_uids"] == ["comprehensive_lora"]
        assert call_kwargs["output_log_probs"] is True
