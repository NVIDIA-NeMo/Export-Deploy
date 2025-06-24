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

from nemo_deploy.nlp.trtllm_api_deployable import TensorRTLLMAPIDeployable


@pytest.fixture
def mock_llm():
    llm = MagicMock()

    # Mock output structure for generate method
    mock_output = MagicMock()
    mock_output.outputs = [MagicMock()]
    mock_output.outputs[0].text = "Generated text"

    llm.generate.return_value = [mock_output]
    return llm


@pytest.fixture
def mock_sampling_params():
    with patch("nemo_deploy.nlp.trtllm_api_deployable.SamplingParams") as mock:
        yield mock


@pytest.fixture
def mock_pytorch_config():
    with patch("nemo_deploy.nlp.trtllm_api_deployable.PyTorchConfig") as mock:
        mock.__annotations__ = {}
        yield mock


@pytest.mark.run_only_on("GPU")
class TestTensorRTLLMAPIDeployable:
    def test_initialization_with_defaults(self, mock_pytorch_config):
        with patch("nemo_deploy.nlp.trtllm_api_deployable.LLM") as mock_llm_class:
            mock_llm_instance = MagicMock()
            mock_llm_class.return_value = mock_llm_instance

            deployer = TensorRTLLMAPIDeployable(hf_model_id_path="test/model")

            assert deployer.model == mock_llm_instance
            mock_llm_class.assert_called_once()

    def test_initialization_with_custom_params(self, mock_pytorch_config):
        with patch("nemo_deploy.nlp.trtllm_api_deployable.LLM") as mock_llm_class:
            mock_llm_instance = MagicMock()
            mock_llm_class.return_value = mock_llm_instance

            deployer = TensorRTLLMAPIDeployable(
                hf_model_id_path="test/model",
                tokenizer="test/tokenizer",
                tensor_parallel_size=2,
                pipeline_parallel_size=2,
                max_batch_size=16,
                max_num_tokens=4096,
                backend="torch",
                dtype="float16",
            )

            assert deployer.model == mock_llm_instance
            mock_llm_class.assert_called_once()

            # Verify the call arguments
            call_args = mock_llm_class.call_args
            assert call_args.kwargs["model"] == "test/model"
            assert call_args.kwargs["tokenizer"] == "test/tokenizer"
            assert call_args.kwargs["tensor_parallel_size"] == 2
            assert call_args.kwargs["pipeline_parallel_size"] == 2
            assert call_args.kwargs["max_batch_size"] == 16
            assert call_args.kwargs["max_num_tokens"] == 4096
            assert call_args.kwargs["backend"] == "torch"
            assert call_args.kwargs["dtype"] == "float16"

    def test_generate_without_model(self):
        with patch("nemo_deploy.nlp.trtllm_api_deployable.LLM"):
            deployer = TensorRTLLMAPIDeployable(hf_model_id_path="test/model")
            deployer.model = None

            with pytest.raises(RuntimeError, match="Model is not initialized"):
                deployer.generate(prompts=["test prompt"])

    def test_generate_with_model(self, mock_llm, mock_sampling_params, mock_pytorch_config):
        with patch("nemo_deploy.nlp.trtllm_api_deployable.LLM") as mock_llm_class:
            mock_llm_class.return_value = mock_llm

            deployer = TensorRTLLMAPIDeployable(hf_model_id_path="test/model")
            output = deployer.generate(prompts=["test prompt"])

            assert output == ["Generated text"]
            mock_llm.generate.assert_called_once()
            mock_sampling_params.assert_called_once()

    def test_generate_with_parameters(self, mock_llm, mock_sampling_params, mock_pytorch_config):
        with patch("nemo_deploy.nlp.trtllm_api_deployable.LLM") as mock_llm_class:
            mock_llm_class.return_value = mock_llm

            deployer = TensorRTLLMAPIDeployable(hf_model_id_path="test/model")
            output = deployer.generate(prompts=["test prompt"], max_length=100, temperature=0.8, top_k=50, top_p=0.95)

            assert output == ["Generated text"]
            mock_llm.generate.assert_called_once()
            mock_sampling_params.assert_called_once_with(max_tokens=100, temperature=0.8, top_k=50, top_p=0.95)

    def test_triton_input_output_config(self, mock_pytorch_config):
        with patch("nemo_deploy.nlp.trtllm_api_deployable.LLM"):
            deployer = TensorRTLLMAPIDeployable(hf_model_id_path="test/model")

            inputs = deployer.get_triton_input
            outputs = deployer.get_triton_output

            assert len(inputs) == 6  # Verify number of input tensors
            assert len(outputs) == 1  # Verify number of output tensors

            # Verify required input tensor names
            input_names = [tensor.name for tensor in inputs]
            assert "prompts" in input_names
            assert "max_length" in input_names
            assert "max_batch_size" in input_names
            assert "top_k" in input_names
            assert "top_p" in input_names
            assert "temperature" in input_names

            # Verify output tensor names
            assert outputs[0].name == "sentences"
