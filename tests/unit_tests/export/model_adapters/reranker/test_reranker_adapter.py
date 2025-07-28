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

from unittest.mock import Mock, patch

import pytest
import torch

from nemo_export.model_adapters.reranker.reranker_adapter import (
    SequenceClassificationModelAdapterWithoutTypeIds,
    SequenceClassificationModelAdapterWithTypeIds,
    get_llama_reranker_hf_model,
)


class TestSequenceClassificationModelAdapterWithoutTypeIds:
    """Test cases for the SequenceClassificationModelAdapterWithoutTypeIds class."""

    def test_adapter_initialization(self):
        """Test adapter initialization with a mock model."""
        mock_model = Mock()
        mock_config = Mock()
        mock_model.config = mock_config

        adapter = SequenceClassificationModelAdapterWithoutTypeIds(mock_model)

        assert adapter._model == mock_model
        assert adapter.config == mock_config

    def test_adapter_forward_pass(self):
        """Test forward pass through the adapter."""
        # Create mock model and logits
        mock_model = Mock()
        mock_config = Mock()
        mock_model.config = mock_config

        # Mock the model output with logits
        expected_logits = torch.tensor([[0.1, 0.9], [0.8, 0.2]])
        mock_output = Mock()
        mock_output.logits = expected_logits
        mock_model.return_value = mock_output

        adapter = SequenceClassificationModelAdapterWithoutTypeIds(mock_model)

        # Create test inputs
        batch_size, seq_len = 2, 10
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)

        # Call forward
        result = adapter.forward(input_ids, attention_mask)

        # Verify the model was called with correct arguments
        mock_model.assert_called_once_with(input_ids=input_ids, attention_mask=attention_mask)

        # Verify the result is the logits
        torch.testing.assert_close(result, expected_logits)

    def test_adapter_forward_with_different_input_shapes(self):
        """Test forward pass with different input tensor shapes."""
        mock_model = Mock()
        mock_config = Mock()
        mock_model.config = mock_config

        # Mock the model output
        mock_output = Mock()
        mock_output.logits = torch.tensor([[0.5, 0.5]])
        mock_model.return_value = mock_output

        adapter = SequenceClassificationModelAdapterWithoutTypeIds(mock_model)

        # Test with single sample
        input_ids = torch.randint(0, 1000, (1, 5))
        attention_mask = torch.ones(1, 5)

        result = adapter.forward(input_ids, attention_mask)

        mock_model.assert_called_once_with(input_ids=input_ids, attention_mask=attention_mask)
        assert result.shape == (1, 2)

    def test_adapter_inherits_from_nn_module(self):
        """Test that the adapter properly inherits from torch.nn.Module."""
        mock_model = Mock()
        mock_model.config = Mock()

        adapter = SequenceClassificationModelAdapterWithoutTypeIds(mock_model)

        assert isinstance(adapter, torch.nn.Module)


class TestSequenceClassificationModelAdapterWithTypeIds:
    """Test cases for the SequenceClassificationModelAdapterWithTypeIds class."""

    def test_adapter_initialization(self):
        """Test adapter initialization with a mock model."""
        mock_model = Mock()
        mock_config = Mock()
        mock_model.config = mock_config

        adapter = SequenceClassificationModelAdapterWithTypeIds(mock_model)

        assert adapter._model == mock_model
        assert adapter.config == mock_config

    def test_adapter_forward_pass_with_type_ids(self):
        """Test forward pass through the adapter with token type IDs."""
        # Create mock model and logits
        mock_model = Mock()
        mock_config = Mock()
        mock_model.config = mock_config

        # Mock the model output with logits
        expected_logits = torch.tensor([[0.3, 0.7], [0.6, 0.4]])
        mock_output = Mock()
        mock_output.logits = expected_logits
        mock_model.return_value = mock_output

        adapter = SequenceClassificationModelAdapterWithTypeIds(mock_model)

        # Create test inputs
        batch_size, seq_len = 2, 8
        input_ids = torch.randint(0, 1000, (batch_size, seq_len))
        token_type_ids = torch.zeros(batch_size, seq_len, dtype=torch.long)
        attention_mask = torch.ones(batch_size, seq_len)

        # Call forward
        result = adapter.forward(input_ids, token_type_ids, attention_mask)

        # Verify the model was called with correct arguments
        mock_model.assert_called_once_with(
            input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask
        )

        # Verify the result is the logits
        torch.testing.assert_close(result, expected_logits)

    def test_adapter_forward_with_mixed_token_types(self):
        """Test forward pass with mixed token type IDs."""
        mock_model = Mock()
        mock_config = Mock()
        mock_model.config = mock_config

        # Mock the model output
        mock_output = Mock()
        mock_output.logits = torch.tensor([[0.1, 0.9]])
        mock_model.return_value = mock_output

        adapter = SequenceClassificationModelAdapterWithTypeIds(mock_model)

        # Create test inputs with mixed token types
        input_ids = torch.randint(0, 1000, (1, 6))
        token_type_ids = torch.tensor([[0, 0, 0, 1, 1, 1]])  # Two different segments
        attention_mask = torch.ones(1, 6)

        result = adapter.forward(input_ids, token_type_ids, attention_mask)

        mock_model.assert_called_once_with(
            input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask
        )
        assert result.shape == (1, 2)

    def test_adapter_inherits_from_nn_module(self):
        """Test that the adapter properly inherits from torch.nn.Module."""
        mock_model = Mock()
        mock_model.config = Mock()

        adapter = SequenceClassificationModelAdapterWithTypeIds(mock_model)

        assert isinstance(adapter, torch.nn.Module)


class TestGetLlamaRerankerHfModel:
    """Test cases for the get_llama_reranker_hf_model function."""

    @patch("nemo_export.model_adapters.reranker.reranker_adapter.AutoTokenizer")
    @patch("nemo_export.model_adapters.reranker.reranker_adapter.AutoModelForSequenceClassification")
    def test_get_model_without_token_type_ids(self, mock_auto_model, mock_auto_tokenizer):
        """Test loading a model that doesn't use token type IDs."""
        # Setup mocks
        mock_model = Mock()
        mock_config = Mock()
        mock_model.config = mock_config
        mock_model.eval.return_value = mock_model
        mock_auto_model.from_pretrained.return_value = mock_model

        mock_tokenizer = Mock()
        mock_tokenizer.model_input_names = ["input_ids", "attention_mask"]  # No token_type_ids
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        # Call the function
        model, tokenizer = get_llama_reranker_hf_model("test-model")

        # Verify model loading
        mock_auto_model.from_pretrained.assert_called_once_with(
            "test-model", trust_remote_code=False, attn_implementation=None
        )
        mock_model.eval.assert_called_once()

        # Verify tokenizer loading
        mock_auto_tokenizer.from_pretrained.assert_called_once_with("test-model", trust_remote_code=False)

        # Verify adapter type
        assert isinstance(model, SequenceClassificationModelAdapterWithoutTypeIds)
        assert model._model == mock_model
        assert tokenizer == mock_tokenizer

    @patch("nemo_export.model_adapters.reranker.reranker_adapter.AutoTokenizer")
    @patch("nemo_export.model_adapters.reranker.reranker_adapter.AutoModelForSequenceClassification")
    def test_get_model_with_token_type_ids(self, mock_auto_model, mock_auto_tokenizer):
        """Test loading a model that uses token type IDs."""
        # Setup mocks
        mock_model = Mock()
        mock_config = Mock()
        mock_model.config = mock_config
        mock_model.eval.return_value = mock_model
        mock_auto_model.from_pretrained.return_value = mock_model

        mock_tokenizer = Mock()
        mock_tokenizer.model_input_names = ["input_ids", "attention_mask", "token_type_ids"]
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        # Call the function
        model, tokenizer = get_llama_reranker_hf_model("test-bert-model")

        # Verify model loading
        mock_auto_model.from_pretrained.assert_called_once_with(
            "test-bert-model", trust_remote_code=False, attn_implementation=None
        )

        # Verify adapter type
        assert isinstance(model, SequenceClassificationModelAdapterWithTypeIds)
        assert model._model == mock_model
        assert tokenizer == mock_tokenizer

    @patch("nemo_export.model_adapters.reranker.reranker_adapter.AutoTokenizer")
    @patch("nemo_export.model_adapters.reranker.reranker_adapter.AutoModelForSequenceClassification")
    def test_get_model_with_trust_remote_code(self, mock_auto_model, mock_auto_tokenizer):
        """Test loading a model with trust_remote_code=True."""
        # Setup mocks
        mock_model = Mock()
        mock_config = Mock()
        mock_model.config = mock_config
        mock_model.eval.return_value = mock_model
        mock_auto_model.from_pretrained.return_value = mock_model

        mock_tokenizer = Mock()
        mock_tokenizer.model_input_names = ["input_ids", "attention_mask"]
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        # Call the function with trust_remote_code=True
        model, tokenizer = get_llama_reranker_hf_model("test-model", trust_remote_code=True)

        # Verify model loading with trust_remote_code
        mock_auto_model.from_pretrained.assert_called_once_with(
            "test-model", trust_remote_code=True, attn_implementation=None
        )

        # Verify tokenizer loading with trust_remote_code
        mock_auto_tokenizer.from_pretrained.assert_called_once_with("test-model", trust_remote_code=True)

    @patch("nemo_export.model_adapters.reranker.reranker_adapter.AutoTokenizer")
    @patch("nemo_export.model_adapters.reranker.reranker_adapter.AutoModelForSequenceClassification")
    def test_get_model_with_attn_implementation(self, mock_auto_model, mock_auto_tokenizer):
        """Test loading a model with specific attention implementation."""
        # Setup mocks
        mock_model = Mock()
        mock_config = Mock()
        mock_model.config = mock_config
        mock_model.eval.return_value = mock_model
        mock_auto_model.from_pretrained.return_value = mock_model

        mock_tokenizer = Mock()
        mock_tokenizer.model_input_names = ["input_ids", "attention_mask"]
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        # Call the function with attention implementation
        attn_impl = "flash_attention_2"
        model, tokenizer = get_llama_reranker_hf_model("test-model", attn_implementation=attn_impl)

        # Verify model loading with attention implementation
        mock_auto_model.from_pretrained.assert_called_once_with(
            "test-model", trust_remote_code=False, attn_implementation=attn_impl
        )

        # Verify config is reset after init
        assert mock_config._attn_implementation == attn_impl

    @patch("nemo_export.model_adapters.reranker.reranker_adapter.AutoTokenizer")
    @patch("nemo_export.model_adapters.reranker.reranker_adapter.AutoModelForSequenceClassification")
    def test_get_model_with_pathlike_input(self, mock_auto_model, mock_auto_tokenizer):
        """Test loading a model with a PathLike input."""
        from pathlib import Path

        # Setup mocks
        mock_model = Mock()
        mock_config = Mock()
        mock_model.config = mock_config
        mock_model.eval.return_value = mock_model
        mock_auto_model.from_pretrained.return_value = mock_model

        mock_tokenizer = Mock()
        mock_tokenizer.model_input_names = ["input_ids", "attention_mask"]
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        # Call the function with Path object
        model_path = Path("/path/to/model")
        model, tokenizer = get_llama_reranker_hf_model(model_path)

        # Verify model loading
        mock_auto_model.from_pretrained.assert_called_once_with(
            model_path, trust_remote_code=False, attn_implementation=None
        )

        # Verify tokenizer loading
        mock_auto_tokenizer.from_pretrained.assert_called_once_with(model_path, trust_remote_code=False)

    @patch("nemo_export.model_adapters.reranker.reranker_adapter.AutoTokenizer")
    @patch("nemo_export.model_adapters.reranker.reranker_adapter.AutoModelForSequenceClassification")
    def test_get_model_all_parameters(self, mock_auto_model, mock_auto_tokenizer):
        """Test loading a model with all parameters specified."""
        # Setup mocks
        mock_model = Mock()
        mock_config = Mock()
        mock_model.config = mock_config
        mock_model.eval.return_value = mock_model
        mock_auto_model.from_pretrained.return_value = mock_model

        mock_tokenizer = Mock()
        mock_tokenizer.model_input_names = ["input_ids", "attention_mask", "token_type_ids"]
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        # Call the function with all parameters
        model, tokenizer = get_llama_reranker_hf_model(
            "test-model", trust_remote_code=True, attn_implementation="flash_attention_2"
        )

        # Verify model loading
        mock_auto_model.from_pretrained.assert_called_once_with(
            "test-model", trust_remote_code=True, attn_implementation="flash_attention_2"
        )

        # Verify tokenizer loading
        mock_auto_tokenizer.from_pretrained.assert_called_once_with("test-model", trust_remote_code=True)

        # Verify adapter type (should be with token type IDs)
        assert isinstance(model, SequenceClassificationModelAdapterWithTypeIds)

        # Verify config is reset
        assert mock_config._attn_implementation == "flash_attention_2"

    @patch("nemo_export.model_adapters.reranker.reranker_adapter.AutoTokenizer")
    @patch("nemo_export.model_adapters.reranker.reranker_adapter.AutoModelForSequenceClassification")
    def test_get_model_exception_handling(self, mock_auto_model, mock_auto_tokenizer):
        """Test that exceptions from HuggingFace are properly propagated."""
        # Setup mock to raise exception
        mock_auto_model.from_pretrained.side_effect = ValueError("Model not found")

        # Verify exception is propagated
        with pytest.raises(ValueError, match="Model not found"):
            get_llama_reranker_hf_model("non-existent-model")

    @patch("nemo_export.model_adapters.reranker.reranker_adapter.AutoTokenizer")
    @patch("nemo_export.model_adapters.reranker.reranker_adapter.AutoModelForSequenceClassification")
    def test_model_eval_mode(self, mock_auto_model, mock_auto_tokenizer):
        """Test that the model is put in evaluation mode."""
        # Setup mocks
        mock_model = Mock()
        mock_config = Mock()
        mock_model.config = mock_config
        mock_model.eval.return_value = mock_model
        mock_auto_model.from_pretrained.return_value = mock_model

        mock_tokenizer = Mock()
        mock_tokenizer.model_input_names = ["input_ids", "attention_mask"]
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        # Call the function
        model, tokenizer = get_llama_reranker_hf_model("test-model")

        # Verify eval() was called
        mock_model.eval.assert_called_once()
