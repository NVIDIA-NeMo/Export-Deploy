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
from megatron.core.inference.common_inference_params import CommonInferenceParams

from nemo_deploy.nlp.megatronllm_deployable import MegatronLLMDeploy, MegatronLLMDeployableNemo2, dict_to_str


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


# Additional tests for improved coverage
@pytest.mark.run_only_on("GPU")
def test_megatron_llm_deploy():
    """Test the MegatronLLMDeploy class also returns MegatronLLMDeployableNemo2 instance."""
    with patch("nemo_deploy.nlp.megatronllm_deployable.nemo_checkpoint_version") as mock_version:
        with patch("nemo_deploy.nlp.megatronllm_deployable.NEMO2", "nemo2"):
            mock_version.return_value = "nemo2"
            with patch.object(MegatronLLMDeployableNemo2, "__init__", return_value=None) as mock_init:
                deployable = MegatronLLMDeploy.get_deployable(
                    nemo_checkpoint_filepath="test.nemo",
                    num_devices=2,
                    num_nodes=1,
                    tensor_model_parallel_size=2,
                    pipeline_model_parallel_size=1,
                    expert_model_parallel_size=1,
                    context_parallel_size=1,
                    max_batch_size=16,
                    random_seed=42,
                    enable_flash_decode=True,
                    enable_cuda_graphs=True,
                    legacy_ckpt=True,
                )

                # Verify the correct instance is returned
                assert isinstance(deployable, MegatronLLMDeployableNemo2)
                mock_init.assert_called_once()


@pytest.mark.run_only_on("GPU")
def test_megatron_llm_deploy_unsupported_version():
    """Test the MegatronLLMDeploy class with nemo1 checkpoint version."""
    with patch("nemo_deploy.nlp.megatronllm_deployable.nemo_checkpoint_version") as mock_version:
        with patch("nemo_deploy.nlp.megatronllm_deployable.NEMO2", "nemo2"):
            mock_version.return_value = "nemo1"  # Different from NEMO2
            with pytest.raises(Exception, match="Only NeMo 2.0 checkpoint is supported"):
                MegatronLLMDeploy.get_deployable(nemo_checkpoint_filepath="test.nemo")


@pytest.mark.run_only_on("GPU")
def test_dict_to_str():
    """Test the dict_to_str utility function."""
    test_dict = {"role": "user", "content": "Hello world"}
    result = dict_to_str(test_dict)
    assert isinstance(result, str)
    assert '"role": "user"' in result
    assert '"content": "Hello world"' in result


@pytest.mark.run_only_on("GPU")
def test_apply_chat_template_none_template(deployable):
    """Test chat template application when tokenizer has no chat template."""
    deployable.mcore_tokenizer.tokenizer.tokenizer.chat_template = None
    messages = [{"role": "user", "content": "Hello"}]

    with pytest.raises(ValueError, match="The tokenizer does not have a chat template defined"):
        deployable.apply_chat_template(messages)


@pytest.mark.run_only_on("GPU")
def test_apply_chat_template_attribute_error(deployable):
    """Test chat template application when tokenizer raises AttributeError."""
    # Remove the chat_template attribute to trigger AttributeError
    del deployable.mcore_tokenizer.tokenizer.tokenizer.chat_template
    messages = [{"role": "user", "content": "Hello"}]

    with pytest.raises(ValueError, match="The tokenizer does not have chat template"):
        deployable.apply_chat_template(messages)


@pytest.mark.run_only_on("GPU")
def test_apply_chat_template_with_generation_prompt_false(deployable):
    """Test chat template application with add_generation_prompt=False."""
    messages = [{"role": "user", "content": "Hello"}]

    template_mock = MagicMock()
    template_mock.render.return_value = "Rendered template without generation prompt"

    with patch("nemo_deploy.nlp.megatronllm_deployable.Template", return_value=template_mock):
        template = deployable.apply_chat_template(messages, add_generation_prompt=False)
        assert template == "Rendered template without generation prompt"

        # Verify render was called with add_generation_prompt=False
        call_args = template_mock.render.call_args[1]
        assert call_args["add_generation_prompt"] is False


@pytest.mark.run_only_on("GPU")
def test_str_to_dict_invalid_json(deployable):
    """Test string to dictionary conversion with invalid JSON."""
    invalid_json = '{"key": invalid}'

    with pytest.raises(Exception):  # json.JSONDecodeError or similar
        deployable.str_to_dict(invalid_json)


@pytest.mark.run_only_on("GPU")
def test_generate_with_cuda_graphs_empty_prompts(deployable):
    """Test text generation with CUDA graphs and empty prompts."""
    deployable.enable_cuda_graphs = True
    deployable.max_batch_size = 4
    prompts = []
    inference_params = CommonInferenceParams()

    with patch.object(deployable.mcore_engine, "generate") as mock_generate:
        mock_generate.return_value = ["", "", "", ""]

        results = deployable.generate(prompts, inference_params)
        assert len(results) == 0

        # Should still pad to max_batch_size but return empty results
        called_args = mock_generate.call_args[1]
        assert len(called_args["prompts"]) == 4


@pytest.mark.run_only_on("GPU")
def test_generate_other_ranks_exit_signal(deployable):
    """Test generate_other_ranks method when receiving exit signal."""
    with patch("torch.distributed.broadcast") as mock_broadcast, patch("torch.empty") as mock_empty:
        # Mock the message tensor to return 1 (exit signal)
        mock_message = MagicMock()
        mock_message.__eq__ = MagicMock(return_value=False)  # Not equal to 0
        mock_empty.return_value = mock_message

        # Should return immediately without further processing
        deployable.generate_other_ranks()

        mock_broadcast.assert_called_once()


@pytest.mark.run_only_on("GPU")
def test_generate_other_ranks_continue_processing(deployable):
    """Test generate_other_ranks method when continuing processing."""
    with (
        patch("torch.distributed.broadcast") as mock_broadcast,
        patch("torch.empty") as mock_empty,
        patch("nemo_deploy.nlp.megatronllm_deployable.broadcast_list") as mock_broadcast_list,
        patch.object(deployable, "generate") as mock_generate,
    ):
        # Mock the message tensor to return 0 first (continue), then 1 (exit)
        mock_message = MagicMock()
        mock_message.__eq__ = MagicMock(side_effect=[True, False])  # First 0, then not 0
        mock_empty.return_value = mock_message

        # Mock broadcast_list returns
        mock_broadcast_list.side_effect = [
            ["test prompt"],  # prompts
            [1.0, 1, 0.0, 256, False],  # inference parameters
        ]

        deployable.generate_other_ranks()

        # Should call broadcast twice (for message check)
        assert mock_broadcast.call_count == 2
        # Should call broadcast_list twice (prompts + params)
        assert mock_broadcast_list.call_count == 2
        # Should call generate once
        mock_generate.assert_called_once()


@pytest.mark.run_only_on("GPU")
def test_triton_infer_fn_with_top_logprobs(deployable):
    """Test triton inference with top logprobs by testing the underlying _infer_fn method."""
    prompts = ["Hello"]

    with (
        patch.object(deployable, "generate") as mock_generate,
        patch.object(deployable, "remove_eos_token") as mock_remove_eos,
        patch("nemo_deploy.nlp.megatronllm_deployable.dict_to_str") as mock_dict_to_str,
    ):
        mock_result = MagicMock()
        mock_result.generated_text = "Generated text"
        mock_result.generated_top_n_logprobs = {"token1": 0.5, "token2": 0.3}

        mock_generate.return_value = [mock_result]
        mock_remove_eos.return_value = ["Generated text"]
        mock_dict_to_str.return_value = '{"token1": 0.5, "token2": 0.3}'

        # Test the underlying inference logic with top_logprobs
        output_infer = deployable._infer_fn(
            prompts=prompts, top_logprobs=5, temperature=1.0, top_k=1, top_p=0.0, num_tokens_to_generate=256
        )

        assert output_infer["sentences"] == ["Generated text"]
        assert "top_logprobs" in output_infer.keys()
        assert output_infer["top_logprobs"] == ['{"token1": 0.5, "token2": 0.3}']

        mock_dict_to_str.assert_called_once_with({"token1": 0.5, "token2": 0.3})


@pytest.mark.run_only_on("GPU")
def test_infer_fn_with_echo_and_log_probs(deployable):
    """Test _infer_fn method with echo=True and log probabilities."""
    prompts = ["Hello"]

    with (
        patch.object(deployable, "generate") as mock_generate,
        patch.object(deployable, "remove_eos_token") as mock_remove_eos,
        patch("torch.tensor") as mock_tensor,
    ):
        # Set up mock results with both prompt and generated log probs
        mock_result = MagicMock()
        mock_result.prompt = "Hello"
        mock_result.generated_text = " World"
        mock_result.prompt_log_probs = [0.1, 0.2]
        mock_result.generated_log_probs = [0.3, 0.4]

        mock_generate.return_value = [mock_result]
        mock_remove_eos.return_value = ["Hello World"]

        # Mock torch.tensor to return appropriate tensor
        mock_tensor_instance = MagicMock()
        mock_tensor_instance.cpu.return_value.detach.return_value.numpy.return_value = np.array([0.1, 0.2, 0.3, 0.4])
        mock_tensor.return_value = mock_tensor_instance

        # Test with echo=True and log probabilities
        output_infer = deployable._infer_fn(prompts=prompts, echo=True, log_probs=True, text_only=True)

        assert output_infer["sentences"] == ["Hello World"]
        assert "log_probs" in output_infer.keys()

        # Verify torch.tensor was called with combined log probs
        mock_tensor.assert_called_once_with([0.1, 0.2, 0.3, 0.4])


@pytest.mark.run_only_on("GPU")
def test_infer_fn_with_echo_and_prompt_top_logprobs(deployable):
    """Test _infer_fn method with echo=True and top_logprobs, ensuring prompt_top_n_logprobs is included."""
    prompts = ["Hello"]

    with (
        patch.object(deployable, "generate") as mock_generate,
        patch.object(deployable, "remove_eos_token") as mock_remove_eos,
        patch("nemo_deploy.nlp.megatronllm_deployable.dict_to_str") as mock_dict_to_str,
    ):
        mock_result = MagicMock()
        mock_result.prompt = "Hello"
        mock_result.generated_text = " World"
        mock_result.prompt_top_n_logprobs = [{"tokenA": 0.9}]
        mock_result.generated_top_n_logprobs = [{"tokenB": 0.8}]

        mock_generate.return_value = [mock_result]
        mock_remove_eos.return_value = ["Hello World"]
        mock_dict_to_str.return_value = '[{"tokenA": 0.9}, {"tokenB": 0.8}]'

        output_infer = deployable._infer_fn(prompts=prompts, echo=True, top_logprobs=1, text_only=True)
        assert output_infer["sentences"] == ["Hello World"]
        assert "top_logprobs" in output_infer
        assert output_infer["top_logprobs"] == ['[{"tokenA": 0.9}, {"tokenB": 0.8}]']
        mock_dict_to_str.assert_called_once_with([{"tokenA": 0.9}] + [{"tokenB": 0.8}])


@pytest.mark.run_only_on("GPU")
def test_infer_fn_with_echo_text_only_false(deployable):
    """Test _infer_fn method with echo=True and text_only=False."""
    prompts = ["Hello"]

    with (
        patch.object(deployable, "generate") as mock_generate,
        patch.object(deployable, "remove_eos_token") as mock_remove_eos,
    ):
        mock_result = MagicMock()
        mock_result.prompt = "Hello"
        mock_result.generated_text = " World"

        mock_generate.return_value = [mock_result]
        mock_remove_eos.return_value = [mock_result]  # When text_only=False, returns full result object

        output_infer = deployable._infer_fn(prompts=prompts, echo=True, text_only=False)

        assert output_infer["sentences"] == [mock_result]


@pytest.mark.run_only_on("GPU")
def test_infer_fn_echo_with_log_probs_different_lengths(deployable):
    """Test _infer_fn method with echo=True and different log prob lengths (padding scenario)."""
    prompts = ["Hello", "Hi"]

    with (
        patch.object(deployable, "generate") as mock_generate,
        patch.object(deployable, "remove_eos_token") as mock_remove_eos,
        patch("torch.tensor") as mock_tensor,
        patch("numpy.pad") as mock_pad,
    ):
        # Set up mock results with different prompt log prob lengths
        mock_result1 = MagicMock()
        mock_result1.prompt = "Hello"
        mock_result1.generated_text = " World"
        mock_result1.prompt_log_probs = [0.1, 0.2, 0.3]  # 3 tokens
        mock_result1.generated_log_probs = [0.4]

        mock_result2 = MagicMock()
        mock_result2.prompt = "Hi"
        mock_result2.generated_text = " There"
        mock_result2.prompt_log_probs = [0.5]  # 1 token
        mock_result2.generated_log_probs = [0.6]

        mock_generate.return_value = [mock_result1, mock_result2]
        mock_remove_eos.return_value = ["Hello World", "Hi There"]

        # Mock torch.tensor to return different length arrays
        mock_tensor_instance1 = MagicMock()
        mock_tensor_instance1.cpu.return_value.detach.return_value.numpy.return_value = (
            np.array([0.1, 0.2, 0.3, 0.4])  # Length 4
        )
        mock_tensor_instance2 = MagicMock()
        mock_tensor_instance2.cpu.return_value.detach.return_value.numpy.return_value = (
            np.array([0.5, 0.6])  # Length 2
        )
        mock_tensor.side_effect = [mock_tensor_instance1, mock_tensor_instance2]

        # Mock numpy.pad to simulate padding behavior
        mock_pad.side_effect = [
            np.array([0.1, 0.2, 0.3, 0.4]),  # First array doesn't need padding
            np.array([0.5, 0.6, 0, 0]),  # Second array padded to length 4
        ]

        output_infer = deployable._infer_fn(prompts=prompts, echo=True, log_probs=True)

        assert output_infer["sentences"] == ["Hello World", "Hi There"]
        assert "log_probs" in output_infer.keys()

        # Verify padding was called
        assert mock_pad.call_count == 2


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
        mock_generate.assert_called_once_with(prompts=prompts, add_BOS=False, common_inference_params=inference_params)


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

    with patch("nemo_deploy.nlp.megatronllm_deployable.Template", return_value=template_mock):
        template = deployable.apply_chat_template(messages)
        assert template == "Rendered template with Hello"
        template_mock.render.assert_called_once()


@pytest.mark.run_only_on("GPU")
def test_remove_eos_token(deployable):
    """Test EOS token removal covering all code paths."""
    # Test normal case - eos_token accessible via self.mcore_tokenizer.tokenizer.tokenizer.eos_token
    texts = ["Hello<eos>", "World", "Test<eos>"]
    cleaned_texts = deployable.remove_eos_token(texts)
    assert cleaned_texts == ["Hello", "World", "Test"]


@pytest.mark.run_only_on("GPU")
def test_remove_eos_token_comprehensive(deployable):
    """Test remove_eos_token method covering all code paths and edge cases."""

    # Test case 1: Normal case - eos_token accessible
    deployable.mcore_tokenizer.tokenizer.tokenizer.eos_token = "<eos>"
    texts = ["Hello<eos>", "World", "Test<eos>"]
    cleaned_texts = deployable.remove_eos_token(texts)
    assert cleaned_texts == ["Hello", "World", "Test"]

    # Test case 2: Multiple EOS tokens in same text (should remove only the last one)
    texts = ["Hello<eos>World<eos>"]
    cleaned_texts = deployable.remove_eos_token(texts)
    assert cleaned_texts == ["Hello<eos>World"]

    # Test case 3: Text that is just the EOS token
    texts = ["<eos>"]
    cleaned_texts = deployable.remove_eos_token(texts)
    assert cleaned_texts == [""]

    # Test case 4: Empty text list
    texts = []
    cleaned_texts = deployable.remove_eos_token(texts)
    assert cleaned_texts == []

    # Test case 5: Empty strings in the list
    texts = ["", "Hello<eos>", ""]
    cleaned_texts = deployable.remove_eos_token(texts)
    assert cleaned_texts == ["", "Hello", ""]

    # Test case 6: Fallback case 1 - AttributeError on eos_token, but eos_id and special_tokens work
    del deployable.mcore_tokenizer.tokenizer.tokenizer.eos_token
    deployable.mcore_tokenizer.tokenizer.tokenizer.eos_id = 2
    deployable.mcore_tokenizer.tokenizer.tokenizer.special_tokens = {2: "</s>"}

    texts = ["Hello</s>", "World", "Test</s>"]
    cleaned_texts = deployable.remove_eos_token(texts)
    assert cleaned_texts == ["Hello", "World", "Test"]

    # Test case 7: Fallback case 1 with multiple tokens
    texts = ["Hello</s>World</s>"]
    cleaned_texts = deployable.remove_eos_token(texts)
    assert cleaned_texts == ["Hello</s>World"]

    # Test case 8: Fallback case 2 - Both AttributeErrors, return unchanged
    del deployable.mcore_tokenizer.tokenizer.tokenizer.eos_id

    texts = ["Hello<eos>", "World</s>", "Test"]
    cleaned_texts = deployable.remove_eos_token(texts)
    assert cleaned_texts == ["Hello<eos>", "World</s>", "Test"]  # Should return unchanged

    # Test case 9: Fallback case 2 - AttributeError on eos_id but not on eos_token access
    # First delete the special_tokens to trigger AttributeError in second try block
    deployable.mcore_tokenizer.tokenizer.tokenizer.eos_id = 2
    del deployable.mcore_tokenizer.tokenizer.tokenizer.special_tokens

    texts = ["Hello<eos>", "World"]
    cleaned_texts = deployable.remove_eos_token(texts)
    assert cleaned_texts == ["Hello<eos>", "World"]  # Should return unchanged


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

        _ = deployable.get_triton_input
        _ = deployable.get_triton_output

        # Extract mock calls to see what was created
        input_calls = mock_tensor.call_args_list[:11]  # First 9 calls are for inputs
        output_calls = mock_tensor.call_args_list[11:]  # Rest are for outputs

        # Check inputs (simplified to just check count and first param names)
        assert len(input_calls) == 11
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
        assert len(output_calls) == 3
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
        output_infer = deployable._infer_fn(
            prompts=prompts,
            temperature=1.0,
            top_k=1,
            top_p=0.0,
            num_tokens_to_generate=256,
            log_probs=False,
            apply_chat_template=False,
        )

        assert output_infer["sentences"] == ["Generated text 1", "Generated text 2"]
        assert not "log_probs" in output_infer.keys()

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
        mock_tensor_instance.cpu.return_value.detach.return_value.numpy.return_value = np.array([0.1, 0.2, 0.3])
        mock_tensor.return_value = mock_tensor_instance

        # Test with log probabilities
        output_infer = deployable._infer_fn(
            prompts=prompts,
            temperature=1.0,
            top_k=1,
            top_p=0.0,
            num_tokens_to_generate=256,
            log_probs=True,
            apply_chat_template=False,
        )

        assert output_infer["sentences"] == ["Generated text"]
        assert "log_probs" in output_infer.keys()
        assert len(output_infer["log_probs"]) == 1

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
        output_infer = deployable._infer_fn(
            prompts=prompts,
            temperature=1.0,
            top_k=1,
            top_p=0.0,
            num_tokens_to_generate=256,
            log_probs=False,
            apply_chat_template=True,
        )

        assert output_infer["sentences"] == ["Generated response"]
        assert not "log_probs" in output_infer.keys()

        # Verify chat template was applied
        mock_apply_template.assert_called_once_with({"role": "user", "content": "Hello"})

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
        patch("nemo_deploy.nlp.megatronllm_deployable.broadcast_list") as mock_broadcast_list,
    ):
        # Set up mock results
        mock_result = MagicMock()
        mock_result.generated_text = "Generated text"
        mock_generate.return_value = [mock_result, mock_result]
        mock_remove_eos.return_value = ["Generated text", "Generated text"]

        # Test with distributed setup
        output_infer = deployable._infer_fn(
            prompts=prompts,
            temperature=1.0,
            top_k=1,
            top_p=0.0,
            num_tokens_to_generate=256,
            log_probs=False,
            apply_chat_template=False,
        )

        assert output_infer["sentences"] == ["Generated text", "Generated text"]
        assert not "log_probs" in output_infer.keys()

        # Verify distributed operations were called
        mock_broadcast.assert_called_once()
        assert mock_broadcast_list.call_count == 2  # One for prompts, one for parameters


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
        mock_tensor_instance.cpu.return_value.detach.return_value.numpy.return_value = np.array([])
        mock_tensor.return_value = mock_tensor_instance

        # Test with log probabilities but empty results
        output_infer = deployable._infer_fn(
            prompts=prompts,
            temperature=1.0,
            top_k=1,
            top_p=0.0,
            num_tokens_to_generate=256,
            log_probs=True,
            apply_chat_template=False,
        )

        assert output_infer["sentences"] == ["Generated text"]
        assert "log_probs" in output_infer.keys()
        # When log probs are empty, should default to [0]
        assert len(output_infer["log_probs"]) == 1


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
        mock_infer_fn.return_value = {"sentences": ["Generated text 1", "Generated text 2"]}

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
            text_only=True,
            top_logprobs=0,
            echo=False,
        )


@pytest.mark.run_only_on("GPU")
def test_ray_infer_fn_with_defaults(deployable):
    """Test ray_infer_fn method with default parameters."""
    inputs = {"prompts": ["Hello"]}

    # Mock the _infer_fn method
    with patch.object(deployable, "_infer_fn") as mock_infer_fn:
        mock_infer_fn.return_value = {"sentences": ["Generated text"]}

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
            text_only=True,
            top_logprobs=0,
            echo=False,
        )


@pytest.mark.run_only_on("GPU")
def test_ray_infer_fn_with_log_probs(deployable):
    """Test ray_infer_fn method with log probabilities enabled."""
    inputs = {"prompts": ["Hello"], "compute_logprob": True}

    # Mock the _infer_fn method
    with patch.object(deployable, "_infer_fn") as mock_infer_fn:
        mock_log_probs = np.array([[0.1, 0.2, 0.3]])
        mock_infer_fn.return_value = {
            "sentences": ["Generated text"],
            "log_probs": mock_log_probs,
        }

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
        mock_infer_fn.return_value = {"sentences": ["Generated response"]}

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
        mock_infer_fn.return_value = {"sentences": []}

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
        mock_infer_fn.return_value = {
            "sentences": ["Generated response"],
            "log_probs": mock_log_probs,
        }

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
            text_only=True,
            top_logprobs=0,
            echo=False,
        )
