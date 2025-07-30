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
import torch.nn.functional as F

from nemo_export.model_adapters.embedding.embedding_adapter import (
    LlamaBidirectionalHFAdapter,
    Pooling,
    get_llama_bidirectional_hf_model,
)


class TestPooling:
    """Test cases for the Pooling class."""

    def test_pooling_init(self):
        """Test Pooling initialization."""
        pooling = Pooling(pooling_mode="avg")
        assert pooling.pooling_mode == "avg"

    @pytest.mark.parametrize("pool_type", ["avg", "cls", "cls__left", "last", "last__right"])
    def test_pooling_forward_valid_modes(self, pool_type):
        """Test pooling forward with valid pooling modes."""
        pooling = Pooling(pooling_mode=pool_type)

        # Create test tensors
        batch_size, seq_len, hidden_size = 2, 10, 768
        last_hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        attention_mask = torch.ones(batch_size, seq_len)

        # Mask some tokens for realistic scenario
        attention_mask[0, 7:] = 0  # First sequence has 7 valid tokens
        attention_mask[1, 5:] = 0  # Second sequence has 5 valid tokens

        result = pooling.forward(last_hidden_states, attention_mask)

        assert result.shape == (batch_size, hidden_size)
        assert not torch.isnan(result).any()

    def test_pooling_avg_mode(self):
        """Test average pooling specifically."""
        pooling = Pooling(pooling_mode="avg")

        # Simple case for manual verification
        last_hidden_states = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
        attention_mask = torch.tensor([[1, 1, 0]])  # Only first two tokens are valid

        result = pooling.forward(last_hidden_states, attention_mask)

        # Expected: average of first two tokens: [(1+3)/2, (2+4)/2] = [2.0, 3.0]
        expected = torch.tensor([[2.0, 3.0]])
        torch.testing.assert_close(result, expected, atol=1e-6, rtol=1e-6)

    def test_pooling_cls_mode(self):
        """Test CLS pooling."""
        pooling = Pooling(pooling_mode="cls")

        last_hidden_states = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
        attention_mask = torch.tensor([[1, 1, 1]])

        result = pooling.forward(last_hidden_states, attention_mask)

        # Expected: first token [1.0, 2.0]
        expected = torch.tensor([[1.0, 2.0]])
        torch.testing.assert_close(result, expected)

    def test_pooling_last_mode(self):
        """Test last token pooling."""
        pooling = Pooling(pooling_mode="last")

        last_hidden_states = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
        attention_mask = torch.tensor([[1, 1, 1]])

        result = pooling.forward(last_hidden_states, attention_mask)

        # Expected: last token [5.0, 6.0]
        expected = torch.tensor([[5.0, 6.0]])
        torch.testing.assert_close(result, expected)

    def test_pooling_last_right_mode(self):
        """Test last token pooling with right padding."""
        pooling = Pooling(pooling_mode="last__right")

        last_hidden_states = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
        attention_mask = torch.tensor([[1, 1, 0]])  # Last token is padding

        result = pooling.forward(last_hidden_states, attention_mask)

        # Expected: second token (last valid) [3.0, 4.0]
        expected = torch.tensor([[3.0, 4.0]])
        torch.testing.assert_close(result, expected)

    def test_pooling_cls_left_mode(self):
        """Test CLS pooling with left padding."""
        pooling = Pooling(pooling_mode="cls__left")

        last_hidden_states = torch.tensor([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]])
        attention_mask = torch.tensor([[0, 1, 1]])  # First token is padding

        result = pooling.forward(last_hidden_states, attention_mask)

        # Expected: second token (first valid) [3.0, 4.0]
        expected = torch.tensor([[3.0, 4.0]])
        torch.testing.assert_close(result, expected)

    def test_pooling_invalid_mode(self):
        """Test error handling for invalid pooling mode."""
        pooling = Pooling(pooling_mode="invalid")

        last_hidden_states = torch.randn(1, 3, 2)
        attention_mask = torch.ones(1, 3)

        with pytest.raises(ValueError, match="pool_type invalid not supported"):
            pooling.forward(last_hidden_states, attention_mask)

    def test_pooling_with_zero_division_protection(self):
        """Test that avg pooling handles zero attention masks gracefully."""
        pooling = Pooling(pooling_mode="avg")

        batch_size, seq_len, hidden_size = 1, 3, 2
        last_hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        attention_mask = torch.zeros(batch_size, seq_len)  # All tokens masked

        result = pooling.forward(last_hidden_states, attention_mask)

        # Should not produce NaN or inf
        assert not torch.isnan(result).any()
        assert not torch.isinf(result).any()


class TestLlamaBidirectionalHFAdapter:
    """Test cases for the LlamaBidirectionalHFAdapter class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_model = Mock()
        self.mock_model.device = torch.device("cpu")
        self.mock_pooling = Mock()
        self.adapter = LlamaBidirectionalHFAdapter(
            model=self.mock_model, normalize=True, pooling_module=self.mock_pooling
        )

    def test_adapter_init(self):
        """Test adapter initialization."""
        assert self.adapter.model == self.mock_model
        assert self.adapter.normalize is True
        assert self.adapter.pooling_module == self.mock_pooling

    def test_device_property(self):
        """Test device property."""
        assert self.adapter.device == torch.device("cpu")

    def test_forward_basic(self):
        """Test basic forward pass."""
        # Setup mock returns
        hidden_states = torch.randn(2, 10, 768)
        self.mock_model.return_value = {"last_hidden_state": hidden_states}
        embeddings = torch.randn(2, 768)
        self.mock_pooling.return_value = embeddings

        # Test inputs
        input_ids = torch.randint(0, 1000, (2, 10))
        attention_mask = torch.ones(2, 10)

        result = self.adapter.forward(input_ids, attention_mask)

        # Verify model was called correctly
        self.mock_model.assert_called_once_with(input_ids=input_ids, attention_mask=attention_mask)

        # Verify pooling was called correctly
        self.mock_pooling.assert_called_once_with(hidden_states.to(torch.float32), attention_mask)

        # Verify normalization was applied
        expected_norm = F.normalize(embeddings, p=2, dim=1)
        torch.testing.assert_close(result, expected_norm)

    def test_forward_with_token_type_ids(self):
        """Test forward pass with token_type_ids."""
        hidden_states = torch.randn(2, 10, 768)
        self.mock_model.return_value = {"last_hidden_state": hidden_states}
        embeddings = torch.randn(2, 768)
        self.mock_pooling.return_value = embeddings

        input_ids = torch.randint(0, 1000, (2, 10))
        attention_mask = torch.ones(2, 10)
        token_type_ids = torch.zeros(2, 10)

        self.adapter.forward(input_ids, attention_mask, token_type_ids)

        # Verify model was called with token_type_ids
        self.mock_model.assert_called_once_with(
            input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids
        )

    def test_forward_without_normalization(self):
        """Test forward pass without normalization."""
        adapter = LlamaBidirectionalHFAdapter(model=self.mock_model, normalize=False, pooling_module=self.mock_pooling)

        hidden_states = torch.randn(2, 10, 768)
        self.mock_model.return_value = {"last_hidden_state": hidden_states}
        embeddings = torch.randn(2, 768)
        self.mock_pooling.return_value = embeddings

        input_ids = torch.randint(0, 1000, (2, 10))
        attention_mask = torch.ones(2, 10)

        result = adapter.forward(input_ids, attention_mask)

        # Result should be the same as pooling output (no normalization)
        torch.testing.assert_close(result, embeddings)

    def test_forward_with_dimensions(self):
        """Test forward pass with dimension reduction."""
        hidden_states = torch.randn(2, 10, 768)
        self.mock_model.return_value = {"last_hidden_state": hidden_states}
        embeddings = torch.randn(2, 768)
        self.mock_pooling.return_value = embeddings

        input_ids = torch.randint(0, 1000, (2, 10))
        attention_mask = torch.ones(2, 10)
        dimensions = torch.tensor([256, 512])  # Different dimensions for each sample

        result = self.adapter.forward(input_ids, attention_mask, dimensions=dimensions)

        # Check that dimensions were applied correctly
        assert result.shape[1] == dimensions.max().item()

    def test_forward_with_zero_dimensions(self):
        """Test forward pass with invalid dimensions."""
        hidden_states = torch.randn(2, 10, 768)
        self.mock_model.return_value = {"last_hidden_state": hidden_states}
        embeddings = torch.randn(2, 768)
        self.mock_pooling.return_value = embeddings

        input_ids = torch.randint(0, 1000, (2, 10))
        attention_mask = torch.ones(2, 10)
        dimensions = torch.tensor([0, 256])  # Invalid zero dimension

        with pytest.raises(ValueError, match="Dimensions must be positive"):
            self.adapter.forward(input_ids, attention_mask, dimensions=dimensions)

    def test_forward_with_large_dimensions(self):
        """Test forward pass with dimensions larger than embedding size."""
        hidden_states = torch.randn(2, 10, 768)
        self.mock_model.return_value = {"last_hidden_state": hidden_states}
        embeddings = torch.randn(2, 768)
        self.mock_pooling.return_value = embeddings

        input_ids = torch.randint(0, 1000, (2, 10))
        attention_mask = torch.ones(2, 10)
        dimensions = torch.tensor([1000, 2000])  # Larger than 768

        result = self.adapter.forward(input_ids, attention_mask, dimensions=dimensions)

        # Should clip to embedding size and return full embeddings
        assert result.shape == (2, 768)


class TestGetLlamaBidirectionalHFModel:
    """Test cases for the get_llama_bidirectional_hf_model function."""

    @patch("nemo_export.model_adapters.embedding.embedding_adapter.AutoTokenizer")
    @patch("nemo_export.model_adapters.embedding.embedding_adapter.AutoModel")
    def test_get_model_basic(self, mock_auto_model, mock_auto_tokenizer):
        """Test basic model loading."""
        # Setup mocks
        mock_tokenizer = Mock()
        mock_tokenizer.padding_side = "right"
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        mock_model = Mock()
        mock_model.eval.return_value = mock_model
        mock_auto_model.from_pretrained.return_value = mock_model

        model_path = "test/model"
        adapted_model, tokenizer = get_llama_bidirectional_hf_model(
            model_name_or_path=model_path, normalize=True, pooling_mode="avg"
        )

        # Verify function calls
        mock_auto_tokenizer.from_pretrained.assert_called_once_with(model_path, trust_remote_code=False)
        mock_auto_model.from_pretrained.assert_called_once_with(model_path, torch_dtype=None, trust_remote_code=False)
        mock_model.eval.assert_called_once()

        # Verify return types
        assert isinstance(adapted_model, LlamaBidirectionalHFAdapter)
        assert tokenizer == mock_tokenizer
        assert adapted_model.normalize is True

    @patch("nemo_export.model_adapters.embedding.embedding_adapter.AutoTokenizer")
    @patch("nemo_export.model_adapters.embedding.embedding_adapter.AutoModel")
    def test_get_model_with_torch_dtype(self, mock_auto_model, mock_auto_tokenizer):
        """Test model loading with specific torch dtype."""
        mock_tokenizer = Mock()
        mock_tokenizer.padding_side = "right"
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        mock_model = Mock()
        mock_model.eval.return_value = mock_model
        mock_auto_model.from_pretrained.return_value = mock_model

        adapted_model, tokenizer = get_llama_bidirectional_hf_model(
            model_name_or_path="test/model", normalize=False, torch_dtype=torch.float16
        )

        mock_auto_model.from_pretrained.assert_called_once_with(
            "test/model", torch_dtype=torch.float16, trust_remote_code=False
        )

    @patch("nemo_export.model_adapters.embedding.embedding_adapter.AutoTokenizer")
    @patch("nemo_export.model_adapters.embedding.embedding_adapter.AutoModel")
    def test_get_model_with_trust_remote_code(self, mock_auto_model, mock_auto_tokenizer):
        """Test model loading with trust_remote_code=True."""
        mock_tokenizer = Mock()
        mock_tokenizer.padding_side = "right"
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        mock_model = Mock()
        mock_model.eval.return_value = mock_model
        mock_auto_model.from_pretrained.return_value = mock_model

        adapted_model, tokenizer = get_llama_bidirectional_hf_model(
            model_name_or_path="test/model", normalize=True, trust_remote_code=True
        )

        mock_auto_tokenizer.from_pretrained.assert_called_once_with("test/model", trust_remote_code=True)
        mock_auto_model.from_pretrained.assert_called_once_with("test/model", torch_dtype=None, trust_remote_code=True)

    @patch("nemo_export.model_adapters.embedding.embedding_adapter.AutoTokenizer")
    @patch("nemo_export.model_adapters.embedding.embedding_adapter.AutoModel")
    def test_get_model_pooling_mode_adjustment_last(self, mock_auto_model, mock_auto_tokenizer):
        """Test pooling mode adjustment for 'last' with right padding."""
        mock_tokenizer = Mock()
        mock_tokenizer.padding_side = "right"
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        mock_model = Mock()
        mock_model.eval.return_value = mock_model
        mock_auto_model.from_pretrained.return_value = mock_model

        adapted_model, tokenizer = get_llama_bidirectional_hf_model(
            model_name_or_path="test/model", normalize=True, pooling_mode="last"
        )

        # Should use "last__right" mode for right padding
        assert adapted_model.pooling_module.pooling_mode == "last__right"

    @patch("nemo_export.model_adapters.embedding.embedding_adapter.AutoTokenizer")
    @patch("nemo_export.model_adapters.embedding.embedding_adapter.AutoModel")
    def test_get_model_pooling_mode_adjustment_cls(self, mock_auto_model, mock_auto_tokenizer):
        """Test pooling mode adjustment for 'cls' with left padding."""
        mock_tokenizer = Mock()
        mock_tokenizer.padding_side = "left"
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        mock_model = Mock()
        mock_model.eval.return_value = mock_model
        mock_auto_model.from_pretrained.return_value = mock_model

        adapted_model, tokenizer = get_llama_bidirectional_hf_model(
            model_name_or_path="test/model", normalize=True, pooling_mode="cls"
        )

        # Should use "cls__left" mode for left padding
        assert adapted_model.pooling_module.pooling_mode == "cls__left"

    @patch("nemo_export.model_adapters.embedding.embedding_adapter.AutoTokenizer")
    @patch("nemo_export.model_adapters.embedding.embedding_adapter.AutoModel")
    def test_get_model_default_pooling(self, mock_auto_model, mock_auto_tokenizer):
        """Test default pooling mode when none specified."""
        mock_tokenizer = Mock()
        mock_tokenizer.padding_side = "right"
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        mock_model = Mock()
        mock_model.eval.return_value = mock_model
        mock_auto_model.from_pretrained.return_value = mock_model

        adapted_model, tokenizer = get_llama_bidirectional_hf_model(model_name_or_path="test/model", normalize=True)

        # Should default to "avg"
        assert adapted_model.pooling_module.pooling_mode == "avg"

    @patch("nemo_export.model_adapters.embedding.embedding_adapter.AutoTokenizer")
    @patch("nemo_export.model_adapters.embedding.embedding_adapter.AutoModel")
    def test_get_model_nvembed_special_case(self, mock_auto_model, mock_auto_tokenizer):
        """Test special handling for NVEmbedModel."""
        mock_tokenizer = Mock()
        mock_tokenizer.padding_side = "right"
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        # Create mock NVEmbedModel
        mock_nvembed_model = Mock()
        mock_nvembed_model.__class__.__name__ = "NVEmbedModel"
        mock_nvembed_model.eval.return_value = mock_nvembed_model

        # Add the required attributes
        mock_embedding_model = Mock()
        mock_latent_attention_model = Mock()
        mock_nvembed_model.embedding_model = mock_embedding_model
        mock_nvembed_model.latent_attention_model = mock_latent_attention_model

        mock_auto_model.from_pretrained.return_value = mock_nvembed_model

        adapted_model, tokenizer = get_llama_bidirectional_hf_model(
            model_name_or_path="test/nvembed", normalize=True, pooling_mode="avg"
        )

        # Should use the embedding_model as the main model
        assert adapted_model.model == mock_embedding_model
        # Should use latent_attention_model as pooling module
        assert adapted_model.pooling_module == mock_latent_attention_model

    @patch("nemo_export.model_adapters.embedding.embedding_adapter.AutoTokenizer")
    @patch("nemo_export.model_adapters.embedding.embedding_adapter.AutoModel")
    def test_get_model_with_pathlike(self, mock_auto_model, mock_auto_tokenizer):
        """Test model loading with os.PathLike input."""
        mock_tokenizer = Mock()
        mock_tokenizer.padding_side = "right"
        mock_auto_tokenizer.from_pretrained.return_value = mock_tokenizer

        mock_model = Mock()
        mock_model.eval.return_value = mock_model
        mock_auto_model.from_pretrained.return_value = mock_model

        # Use a pathlib Path object
        from pathlib import Path

        model_path = Path("test/model")

        adapted_model, tokenizer = get_llama_bidirectional_hf_model(model_name_or_path=model_path, normalize=True)

        mock_auto_tokenizer.from_pretrained.assert_called_once_with(model_path, trust_remote_code=False)
        mock_auto_model.from_pretrained.assert_called_once_with(model_path, torch_dtype=None, trust_remote_code=False)
