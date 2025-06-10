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

import pytest
import numpy as np
from megatron.core.inference.common_inference_params import CommonInferenceParams

from nemo_deploy.nlp.megatronllm_deployable import MegatronLLMDeployableNemo2


@pytest.fixture
def mock_engine_and_tokenizer():
    """Fixture to mock the engine and tokenizer needed for testing."""
    mock_engine = MagicMock()
    mock_model = MagicMock()
    mock_tokenizer = MagicMock()
    mock_tokenizer.tokenizer.tokenizer = MagicMock()
    mock_tokenizer.tokenizer.tokenizer.chat_template = "{{messages}}"
    mock_tokenizer.tokenizer.tokenizer.bos_token = "<bos>"
    mock_tokenizer.tokenizer.tokenizer.eos_token = "<eos>"

    return mock_engine, mock_model, mock_tokenizer


@pytest.fixture
def deployable(mock_engine_and_tokenizer):
    """Fixture to create a deployable instance with mocked dependencies."""
    mock_engine, mock_model, mock_tokenizer = mock_engine_and_tokenizer

    # Patch the __init__ method to avoid file loading
    with patch.object(MegatronLLMDeployableNemo2, "__init__", return_value=None):
        deployable = MegatronLLMDeployableNemo2()

        # Set required attributes manually
        deployable.mcore_engine = mock_engine
        deployable.inference_wrapped_model = mock_model
        deployable.mcore_tokenizer = mock_tokenizer
        deployable.nemo_checkpoint_filepath = "dummy.nemo"
        deployable.max_batch_size = 32
        deployable.enable_cuda_graphs = True

        yield deployable


@pytest.mark.run_only_on("GPU")
def test_initialization(deployable):
    """Test initialization of the deployable class."""
    assert deployable.nemo_checkpoint_filepath == "dummy.nemo"
    assert deployable.max_batch_size == 32
    assert deployable.enable_cuda_graphs is True


@pytest.mark.run_only_on("GPU")
def test_generate_without_cuda_graphs(deployable):
    """Test text generation without CUDA graphs."""
    # Temporarily disable CUDA graphs
    deployable.enable_cuda_graphs = False

    prompts = ["Hello", "World"]
    inference_params = CommonInferenceParams(
        temperature=1.0,
        top_k=1,
        top_p=0.0,
        num_tokens_to_generate=256,
        return_log_probs=False,
    )

    # Mock the generate method
    with patch.object(deployable.mcore_engine, "generate") as mock_generate:
        mock_result = MagicMock()
        mock_result.generated_text = "Generated text"
        mock_generate.return_value = [mock_result, mock_result]

        results = deployable.generate(prompts, inference_params)
        assert len(results) == 2
        mock_generate.assert_called_once_with(
            prompts=prompts, add_BOS=False, common_inference_params=inference_params
        )


@pytest.mark.run_only_on("GPU")
def test_generate_with_cuda_graphs(deployable):
    """Test text generation with CUDA graphs enabled."""
    # Ensure CUDA graphs is enabled
    deployable.enable_cuda_graphs = True
    deployable.max_batch_size = 4

    prompts = ["Hello", "World"]
    inference_params = CommonInferenceParams(
        temperature=1.0,
        top_k=1,
        top_p=0.0,
        num_tokens_to_generate=256,
        return_log_probs=False,
    )

    # Mock the generate method
    with patch.object(deployable.mcore_engine, "generate") as mock_generate:
        mock_result1 = MagicMock()
        mock_result1.generated_text = "Generated text 1"
        mock_result2 = MagicMock()
        mock_result2.generated_text = "Generated text 2"
        mock_result_pad = MagicMock()
        mock_result_pad.generated_text = "Padding text"
        mock_generate.return_value = [
            mock_result1,
            mock_result2,
            mock_result_pad,
            mock_result_pad,
        ]

        results = deployable.generate(prompts, inference_params)

        # Should only return the actual results, not the padding
        assert len(results) == 2

        # Check that the padding was applied in the call
        called_args = mock_generate.call_args[1]
        assert len(called_args["prompts"]) == 4  # Should pad to max_batch_size
        assert called_args["prompts"][:2] == prompts  # Original prompts should be first
        assert called_args["add_BOS"] is False
        assert called_args["common_inference_params"] == inference_params


@pytest.mark.run_only_on("GPU")
def test_apply_chat_template(deployable):
    """Test chat template application."""
    messages = [{"role": "user", "content": "Hello"}]

    # Set up jinja2 mock

    template_mock = MagicMock()
    template_mock.render.return_value = "Rendered template with Hello"

    with patch(
        "nemo_deploy.nlp.megatronllm_deployable.Template", return_value=template_mock
    ):
        template = deployable.apply_chat_template(messages)
        assert template == "Rendered template with Hello"
        template_mock.render.assert_called_once()


@pytest.mark.run_only_on("GPU")
def test_remove_eos_token(deployable):
    """Test EOS token removal."""
    texts = ["Hello<eos>", "World", "Test<eos>"]
    cleaned_texts = deployable.remove_eos_token(texts)
    assert cleaned_texts == ["Hello", "World", "Test"]


@pytest.mark.run_only_on("GPU")
def test_str_to_dict(deployable):
    """Test string to dictionary conversion."""
    json_str = '{"key": "value"}'
    result = deployable.str_to_dict(json_str)
    assert isinstance(result, dict)
    assert result["key"] == "value"


@pytest.mark.run_only_on("GPU")
def test_triton_input_output(deployable):
    """Test Triton input and output tensor definitions."""
    # Mock the Tensor class from pytriton.model_config
    with patch("nemo_deploy.nlp.megatronllm_deployable.Tensor") as mock_tensor:
        # Set up mock to return itself for testing
        mock_tensor.side_effect = lambda name, shape, dtype, optional=False: MagicMock(
            name=name, shape=shape, dtype=dtype, optional=optional
        )

        inputs = deployable.get_triton_input
        outputs = deployable.get_triton_output

        # Extract mock calls to see what was created
        input_calls = mock_tensor.call_args_list[:9]  # First 9 calls are for inputs
        output_calls = mock_tensor.call_args_list[9:]  # Rest are for outputs

        # Check inputs (simplified to just check count and first param names)
        assert len(input_calls) == 9
        input_names = [call[1]["name"] for call in input_calls]
        assert "prompts" in input_names
        assert "max_length" in input_names
        assert "max_batch_size" in input_names
        assert "top_k" in input_names
        assert "top_p" in input_names
        assert "temperature" in input_names
        assert "random_seed" in input_names
        assert "compute_logprob" in input_names
        assert "apply_chat_template" in input_names

        # Check outputs
        assert len(output_calls) == 2
        output_names = [call[1]["name"] for call in output_calls]
        assert "sentences" in output_names
        assert "log_probs" in output_names


@pytest.mark.run_only_on("GPU")
def test_infer_fn_basic(deployable):
    """Test basic functionality of _infer_fn method."""
    prompts = ["Hello", "World"]

    # Mock the generate method and remove_eos_token
    with (
        patch.object(deployable, "generate") as mock_generate,
        patch.object(deployable, "remove_eos_token") as mock_remove_eos,
    ):
        # Set up mock results
        mock_result1 = MagicMock()
        mock_result1.generated_text = "Generated text 1"
        mock_result1.generated_log_probs = [0.1, 0.2, 0.3]

        mock_result2 = MagicMock()
        mock_result2.generated_text = "Generated text 2"
        mock_result2.generated_log_probs = [0.4, 0.5]

        mock_generate.return_value = [mock_result1, mock_result2]
        mock_remove_eos.return_value = ["Generated text 1", "Generated text 2"]

        # Test without log probabilities
        output_texts, output_log_probs = deployable._infer_fn(
            prompts=prompts,
            temperature=1.0,
            top_k=1,
            top_p=0.0,
            num_tokens_to_generate=256,
            log_probs=False,
            apply_chat_template=False,
        )

        assert output_texts == ["Generated text 1", "Generated text 2"]
        assert output_log_probs is None

        # Verify generate was called with correct parameters
        mock_generate.assert_called_once()
        call_args = mock_generate.call_args[0]
        assert call_args[0] == prompts

        # Verify CommonInferenceParams
        inference_params = mock_generate.call_args[0][1]
        assert inference_params.temperature == 1.0
        assert inference_params.top_k == 1
        assert inference_params.top_p == 0.0
        assert inference_params.num_tokens_to_generate == 256
        assert inference_params.return_log_probs is False


@pytest.mark.run_only_on("GPU")
def test_infer_fn_with_log_probs(deployable):
    """Test _infer_fn method with log probabilities enabled."""
    prompts = ["Hello"]

    with (
        patch.object(deployable, "generate") as mock_generate,
        patch.object(deployable, "remove_eos_token") as mock_remove_eos,
        patch("torch.tensor") as mock_tensor,
    ):
        # Set up mock results
        mock_result = MagicMock()
        mock_result.generated_text = "Generated text"
        mock_result.generated_log_probs = [0.1, 0.2, 0.3]

        mock_generate.return_value = [mock_result]
        mock_remove_eos.return_value = ["Generated text"]

        # Mock torch.tensor to return a mock tensor with cpu().detach().numpy()
        mock_tensor_instance = MagicMock()
        mock_tensor_instance.cpu.return_value.detach.return_value.numpy.return_value = (
            np.array([0.1, 0.2, 0.3])
        )
        mock_tensor.return_value = mock_tensor_instance

        # Test with log probabilities
        output_texts, output_log_probs = deployable._infer_fn(
            prompts=prompts,
            temperature=1.0,
            top_k=1,
            top_p=0.0,
            num_tokens_to_generate=256,
            log_probs=True,
            apply_chat_template=False,
        )

        assert output_texts == ["Generated text"]
        assert output_log_probs is not None
        assert len(output_log_probs) == 1

        # Verify torch.tensor was called with log probs
        mock_tensor.assert_called_once_with([0.1, 0.2, 0.3])


@pytest.mark.run_only_on("GPU")
def test_infer_fn_with_chat_template(deployable):
    """Test _infer_fn method with chat template application."""
    prompts = [{"role": "user", "content": "Hello"}]

    with (
        patch.object(deployable, "generate") as mock_generate,
        patch.object(deployable, "remove_eos_token") as mock_remove_eos,
        patch.object(deployable, "apply_chat_template") as mock_apply_template,
    ):
        # Set up mocks
        mock_apply_template.return_value = "Templated: Hello"
        mock_result = MagicMock()
        mock_result.generated_text = "Generated response"
        mock_generate.return_value = [mock_result]
        mock_remove_eos.return_value = ["Generated response"]

        # Test with chat template
        output_texts, output_log_probs = deployable._infer_fn(
            prompts=prompts,
            temperature=1.0,
            top_k=1,
            top_p=0.0,
            num_tokens_to_generate=256,
            log_probs=False,
            apply_chat_template=True,
        )

        assert output_texts == ["Generated response"]
        assert output_log_probs is None

        # Verify chat template was applied
        mock_apply_template.assert_called_once_with(
            {"role": "user", "content": "Hello"}
        )

        # Verify generate was called with templated prompt
        call_args = mock_generate.call_args[0]
        assert call_args[0] == ["Templated: Hello"]


@pytest.mark.run_only_on("GPU")
def test_infer_fn_with_distributed(deployable):
    """Test _infer_fn method with distributed training setup."""
    prompts = ["Hello", "World"]

    with (
        patch.object(deployable, "generate") as mock_generate,
        patch.object(deployable, "remove_eos_token") as mock_remove_eos,
        patch("torch.distributed.is_initialized", return_value=True),
        patch("torch.distributed.get_world_size", return_value=2),
        patch("torch.distributed.broadcast") as mock_broadcast,
        patch(
            "nemo_deploy.nlp.megatronllm_deployable.broadcast_list"
        ) as mock_broadcast_list,
    ):
        # Set up mock results
        mock_result = MagicMock()
        mock_result.generated_text = "Generated text"
        mock_generate.return_value = [mock_result, mock_result]
        mock_remove_eos.return_value = ["Generated text", "Generated text"]

        # Test with distributed setup
        output_texts, output_log_probs = deployable._infer_fn(
            prompts=prompts,
            temperature=1.0,
            top_k=1,
            top_p=0.0,
            num_tokens_to_generate=256,
            log_probs=False,
            apply_chat_template=False,
        )

        assert output_texts == ["Generated text", "Generated text"]
        assert output_log_probs is None

        # Verify distributed operations were called
        mock_broadcast.assert_called_once()
        assert (
            mock_broadcast_list.call_count == 2
        )  # One for prompts, one for parameters


@pytest.mark.run_only_on("GPU")
def test_infer_fn_empty_log_probs(deployable):
    """Test _infer_fn method when log probabilities are empty."""
    prompts = ["Hello"]

    with (
        patch.object(deployable, "generate") as mock_generate,
        patch.object(deployable, "remove_eos_token") as mock_remove_eos,
        patch("torch.tensor") as mock_tensor,
    ):
        # Set up mock results with empty log probs
        mock_result = MagicMock()
        mock_result.generated_text = "Generated text"
        mock_result.generated_log_probs = []

        mock_generate.return_value = [mock_result]
        mock_remove_eos.return_value = ["Generated text"]

        # Mock torch.tensor to return empty array
        mock_tensor_instance = MagicMock()
        mock_tensor_instance.cpu.return_value.detach.return_value.numpy.return_value = (
            np.array([])
        )
        mock_tensor.return_value = mock_tensor_instance

        # Test with log probabilities but empty results
        output_texts, output_log_probs = deployable._infer_fn(
            prompts=prompts,
            temperature=1.0,
            top_k=1,
            top_p=0.0,
            num_tokens_to_generate=256,
            log_probs=True,
            apply_chat_template=False,
        )

        assert output_texts == ["Generated text"]
        assert output_log_probs is not None
        # When log probs are empty, should default to [0]
        assert len(output_log_probs) == 1


@pytest.mark.run_only_on("GPU")
def test_ray_infer_fn_basic(deployable):
    """Test basic functionality of ray_infer_fn method."""
    inputs = {
        "prompts": ["Hello", "World"],
        "temperature": 1.0,
        "top_k": 1,
        "top_p": 0.0,
        "max_length": 256,
        "compute_logprob": False,
        "apply_chat_template": False,
    }

    # Mock the _infer_fn method
    with patch.object(deployable, "_infer_fn") as mock_infer_fn:
        mock_infer_fn.return_value = (["Generated text 1", "Generated text 2"], None)

        result = deployable.ray_infer_fn(inputs)

        assert result == {"sentences": ["Generated text 1", "Generated text 2"]}

        # Verify _infer_fn was called with correct parameters
        mock_infer_fn.assert_called_once_with(
            prompts=["Hello", "World"],
            temperature=1.0,
            top_k=1,
            top_p=0.0,
            num_tokens_to_generate=256,
            log_probs=False,
            apply_chat_template=False,
        )


@pytest.mark.run_only_on("GPU")
def test_ray_infer_fn_with_defaults(deployable):
    """Test ray_infer_fn method with default parameters."""
    inputs = {"prompts": ["Hello"]}

    # Mock the _infer_fn method
    with patch.object(deployable, "_infer_fn") as mock_infer_fn:
        mock_infer_fn.return_value = (["Generated text"], None)

        result = deployable.ray_infer_fn(inputs)

        assert result == {"sentences": ["Generated text"]}

        # Verify _infer_fn was called with default parameters
        mock_infer_fn.assert_called_once_with(
            prompts=["Hello"],
            temperature=1.0,  # default
            top_k=0.0,  # default
            top_p=0.0,  # default
            num_tokens_to_generate=256,  # default
            log_probs=False,  # default
            apply_chat_template=False,  # default
        )


@pytest.mark.run_only_on("GPU")
def test_ray_infer_fn_with_log_probs(deployable):
    """Test ray_infer_fn method with log probabilities enabled."""
    inputs = {"prompts": ["Hello"], "compute_logprob": True}

    # Mock the _infer_fn method
    with patch.object(deployable, "_infer_fn") as mock_infer_fn:
        mock_log_probs = np.array([[0.1, 0.2, 0.3]])
        mock_infer_fn.return_value = (["Generated text"], mock_log_probs)

        result = deployable.ray_infer_fn(inputs)

        assert result == {"sentences": ["Generated text"], "log_probs": mock_log_probs}

        # Verify _infer_fn was called with log_probs=True
        mock_infer_fn.assert_called_once()
        call_kwargs = mock_infer_fn.call_args[1]
        assert call_kwargs["log_probs"] is True


@pytest.mark.run_only_on("GPU")
def test_ray_infer_fn_with_chat_template(deployable):
    """Test ray_infer_fn method with chat template enabled."""
    inputs = {
        "prompts": [{"role": "user", "content": "Hello"}],
        "apply_chat_template": True,
    }

    # Mock the _infer_fn method
    with patch.object(deployable, "_infer_fn") as mock_infer_fn:
        mock_infer_fn.return_value = (["Generated response"], None)

        result = deployable.ray_infer_fn(inputs)

        assert result == {"sentences": ["Generated response"]}

        # Verify _infer_fn was called with apply_chat_template=True
        mock_infer_fn.assert_called_once()
        call_kwargs = mock_infer_fn.call_args[1]
        assert call_kwargs["apply_chat_template"] is True
        assert call_kwargs["prompts"] == [{"role": "user", "content": "Hello"}]


@pytest.mark.run_only_on("GPU")
def test_ray_infer_fn_empty_prompts(deployable):
    """Test ray_infer_fn method with empty prompts list."""
    inputs = {}

    # Mock the _infer_fn method
    with patch.object(deployable, "_infer_fn") as mock_infer_fn:
        mock_infer_fn.return_value = ([], None)

        result = deployable.ray_infer_fn(inputs)

        assert result == {"sentences": []}

        # Verify _infer_fn was called with empty prompts
        mock_infer_fn.assert_called_once()
        call_kwargs = mock_infer_fn.call_args[1]
        assert call_kwargs["prompts"] == []


@pytest.mark.run_only_on("GPU")
def test_ray_infer_fn_all_parameters(deployable):
    """Test ray_infer_fn method with all parameters specified."""
    inputs = {
        "prompts": ["Test prompt"],
        "temperature": 0.7,
        "top_k": 50,
        "top_p": 0.9,
        "max_length": 512,
        "compute_logprob": True,
        "apply_chat_template": True,
    }

    # Mock the _infer_fn method
    with patch.object(deployable, "_infer_fn") as mock_infer_fn:
        mock_log_probs = np.array([[0.1, 0.2]])
        mock_infer_fn.return_value = (["Generated response"], mock_log_probs)

        result = deployable.ray_infer_fn(inputs)

        assert result == {
            "sentences": ["Generated response"],
            "log_probs": mock_log_probs,
        }

        # Verify _infer_fn was called with all specified parameters
        mock_infer_fn.assert_called_once_with(
            prompts=["Test prompt"],
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            num_tokens_to_generate=512,
            log_probs=True,
            apply_chat_template=True,
        )
