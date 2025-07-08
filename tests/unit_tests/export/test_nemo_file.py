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

import pickle
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch
import yaml

from nemo_export.trt_llm.nemo_ckpt_loader.nemo_file import (
    build_tokenizer,
    get_model_type,
    get_tokenizer,
    get_weights_dtype,
    load_distributed_model_weights,
    load_extra_state_from_bytes,
    load_nemo_config,
    load_nemo_model,
    rename_extra_states,
    update_tokenizer_paths,
)


class TestLoadExtraStateFromBytes:
    """Test cases for load_extra_state_from_bytes function."""

    def test_load_extra_state_from_bytes_none(self):
        """Test loading extra state from None."""
        result = load_extra_state_from_bytes(None)
        assert result is None

    def test_load_extra_state_from_bytes_empty_tensor(self):
        """Test loading extra state from empty tensor."""
        empty_tensor = torch.tensor([])
        result = load_extra_state_from_bytes(empty_tensor)
        assert result is None

    def test_load_extra_state_from_bytes_tensor(self):
        """Test loading extra state from tensor."""
        test_data = {"test_key": "test_value"}
        serialized_data = pickle.dumps(test_data)
        tensor_data = torch.tensor(list(serialized_data), dtype=torch.uint8)

        result = load_extra_state_from_bytes(tensor_data)
        assert result == test_data


class TestRenameExtraStates:
    """Test cases for rename_extra_states function."""

    def test_rename_extra_states_no_extra_state(self):
        """Test renaming with no extra state keys."""
        state_dict = {"layer1.weight": torch.randn(10, 10)}
        result = rename_extra_states(state_dict)
        assert result == state_dict

    def test_rename_extra_states_with_valid_keys(self):
        """Test renaming with valid extra state keys."""
        state_dict = {
            "model.layers.attention._extra_state/shard_0_2": torch.randn(10),
            "model.layers.attention._extra_state/shard_1_2": torch.randn(10),
            "normal_layer.weight": torch.randn(10, 10),
        }

        result = rename_extra_states(state_dict)

        # Check that normal layers are preserved
        assert "normal_layer.weight" in result
        # Check that extra states are renamed
        assert "model.layers.0.attention._extra_state" in result
        assert "model.layers.1.attention._extra_state" in result

    def test_rename_extra_states_with_list_values(self):
        """Test renaming with list values."""
        state_dict = {
            "model.layers.attention._extra_state/shard_0_2": [torch.randn(10)],
            "normal_layer.weight": torch.randn(10, 10),
        }

        result = rename_extra_states(state_dict)
        assert "model.layers.0.attention._extra_state" in result
        assert isinstance(result["model.layers.0.attention._extra_state"], torch.Tensor)


class TestUpdateTokenizerPaths:
    """Test cases for update_tokenizer_paths function."""

    def test_update_tokenizer_paths(self):
        """Test updating tokenizer paths."""
        tokenizer_config = {
            "model": "/old/path/tokenizer.model",
            "vocab_file": "/old/path/vocab.txt",
            "merge_file": "/old/path/merges.txt",
        }

        mock_unpacked_dir = Mock()
        mock_unpacked_dir.get_tokenizer_file_path.side_effect = lambda key, file_key, pattern: f"/new/path/{file_key}"

        result = update_tokenizer_paths(tokenizer_config, mock_unpacked_dir)

        assert result["model"] == "/new/path/model"
        assert result["vocab_file"] == "/new/path/vocab_file"
        assert result["merge_file"] == "/new/path/merge_file"


class TestBuildTokenizer:
    """Test cases for build_tokenizer function."""

    def test_build_tokenizer_sentencepiece(self):
        """Test building SentencePiece tokenizer."""
        config = {"library": "sentencepiece", "model": "/path/to/tokenizer.model"}

        with patch("nemo_export.trt_llm.nemo_ckpt_loader.nemo_file.SentencePieceTokenizer") as mock_sp:
            mock_tokenizer = Mock()
            mock_sp.return_value = mock_tokenizer

            result = build_tokenizer(config)

            mock_sp.assert_called_once_with(model_path="/path/to/tokenizer.model")
            assert result == mock_tokenizer

    def test_build_tokenizer_tiktoken(self):
        """Test building Tiktoken tokenizer."""
        config = {"library": "tiktoken", "vocab_file": "/path/to/vocab.json"}

        with patch("nemo_export.trt_llm.nemo_ckpt_loader.nemo_file.TiktokenTokenizer") as mock_tiktoken:
            mock_tokenizer = Mock()
            mock_tiktoken.return_value = mock_tokenizer

            result = build_tokenizer(config)

            mock_tiktoken.assert_called_once_with(vocab_file="/path/to/vocab.json")
            assert result == mock_tokenizer


class TestLoadNemoConfig:
    """Test cases for load_nemo_config function."""

    def test_load_nemo_config_nemo2_structure(self, tmp_path):
        """Test loading config from NeMo 2.0 structure."""
        # Create NeMo 2.0 directory structure
        nemo_dir = tmp_path / "nemo2_checkpoint"
        weights_dir = nemo_dir / "weights"
        context_dir = nemo_dir / "context"
        weights_dir.mkdir(parents=True)
        context_dir.mkdir(parents=True)

        config_data = {"model_type": "llama", "hidden_size": 4096}
        with open(context_dir / "model.yaml", "w") as f:
            yaml.dump(config_data, f)

        result = load_nemo_config(nemo_dir)
        assert result == config_data


class TestGetModelType:
    """Test cases for get_model_type function."""

    def test_get_model_type_nemo2_llama(self):
        """Test getting model type for NeMo 2.0 Llama model."""
        config = {"_target_": "nemo.collections.llm.gpt.model.llama.LlamaModel"}

        with patch("nemo_export.trt_llm.nemo_ckpt_loader.nemo_file.load_nemo_config") as mock_load:
            mock_load.return_value = config

            result = get_model_type("/path/to/checkpoint")
            assert result == "llama"

    def test_get_model_type_nemo2_mistral(self):
        """Test getting model type for NeMo 2.0 Mistral model."""
        config = {"_target_": "nemo.collections.llm.gpt.model.mistral.MistralModel"}

        with patch("nemo_export.trt_llm.nemo_ckpt_loader.nemo_file.load_nemo_config") as mock_load:
            mock_load.return_value = config

            result = get_model_type("/path/to/checkpoint")
            assert result == "llama"

    def test_get_model_type_nemo2_mixtral_vllm(self):
        """Test getting model type for NeMo 2.0 Mixtral model with vLLM type."""
        config = {"_target_": "nemo.collections.llm.gpt.model.mixtral.MixtralModel"}

        with patch("nemo_export.trt_llm.nemo_ckpt_loader.nemo_file.load_nemo_config") as mock_load:
            mock_load.return_value = config

            result = get_model_type("/path/to/checkpoint", use_vllm_type=True)
            assert result == "mixtral"

    def test_get_model_type_unknown_model(self):
        """Test getting model type for unknown model."""
        config = {"_target_": "nemo.collections.llm.gpt.model.unknown.UnknownModel"}

        with patch("nemo_export.trt_llm.nemo_ckpt_loader.nemo_file.load_nemo_config") as mock_load:
            mock_load.return_value = config

            with pytest.raises(KeyError):
                get_model_type("/path/to/checkpoint")


class TestGetWeightsDtype:
    """Test cases for get_weights_dtype function."""

    def test_get_weights_dtype_nemo2(self):
        """Test getting weights dtype for NeMo 2.0 model."""
        config = {
            "_target_": "nemo.collections.llm.gpt.model.llama.LlamaModel",
            "config": {"params_dtype": {"_target_": "torch.float16"}},
        }

        with patch("nemo_export.trt_llm.nemo_ckpt_loader.nemo_file.load_nemo_config") as mock_load:
            mock_load.return_value = config

            result = get_weights_dtype("/path/to/checkpoint")
            assert result == "float16"

    def test_get_weights_dtype_nemo1(self):
        """Test getting weights dtype for NeMo 1.0 model."""
        config = {"precision": "16-mixed"}

        with patch("nemo_export.trt_llm.nemo_ckpt_loader.nemo_file.load_nemo_config") as mock_load:
            mock_load.return_value = config

            with patch("nemo_export.trt_llm.nemo_ckpt_loader.nemo_file.torch_dtype_from_precision") as mock_convert:
                mock_convert.return_value = torch.float16

                result = get_weights_dtype("/path/to/checkpoint")
                assert result == "float16"

    def test_get_weights_dtype_not_found(self):
        """Test getting weights dtype when not found."""
        config = {}

        with patch("nemo_export.trt_llm.nemo_ckpt_loader.nemo_file.load_nemo_config") as mock_load:
            mock_load.return_value = config

            result = get_weights_dtype("/path/to/checkpoint")
            assert result is None


class TestLoadDistributedModelWeights:
    """Test cases for load_distributed_model_weights function."""

    def test_load_distributed_model_weights_torch_tensor(self):
        """Test loading distributed model weights as torch tensors."""
        mock_state_dict = {"layer1.weight": torch.randn(10, 10), "layer2.bias": torch.randn(10)}

        with patch("nemo_export.trt_llm.nemo_ckpt_loader.nemo_file.load_model_weights") as mock_load:
            mock_load.return_value = mock_state_dict

            with patch("nemo_export.trt_llm.nemo_ckpt_loader.nemo_file.rename_extra_states") as mock_rename:
                mock_rename.return_value = mock_state_dict

                result = load_distributed_model_weights("/path/to/checkpoint", True)

                assert result == mock_state_dict
                mock_load.assert_called_once_with("/path/to/checkpoint", load_extra_states=True)


class TestLoadNemoModel:
    """Test cases for load_nemo_model function."""

    def test_load_nemo_model_nemo2_structure(self, tmp_path):
        """Test loading NeMo 2.0 model."""
        nemo_ckpt = tmp_path / "nemo2_checkpoint"
        nemo_ckpt.mkdir()
        (nemo_ckpt / "weights").mkdir()
        (nemo_ckpt / "context").mkdir()

        export_dir = tmp_path / "export"
        export_dir.mkdir()

        config_data = {
            "config": {
                "activation_func": {"_target_": "torch.nn.functional.silu"},
                "num_moe_experts": 8,
                "add_bias_linear": True,
            }
        }

        with open(nemo_ckpt / "context" / "model.yaml", "w") as f:
            yaml.dump(config_data, f)

        mock_state_dict = {"layer1.weight": torch.randn(10, 10)}

        with patch(
            "nemo_export.trt_llm.nemo_ckpt_loader.nemo_file.load_distributed_model_weights"
        ) as mock_load_weights:
            mock_load_weights.return_value = mock_state_dict

            model, config, tokenizer = load_nemo_model(nemo_ckpt, export_dir)

            assert model == mock_state_dict
            assert config["activation"] == "fast-swiglu"
            assert config["bias"] is True
            assert config["num_moe_experts"] == 8

    def test_load_nemo_model_nonexistent_path(self):
        """Test loading model with nonexistent path."""
        with pytest.raises(TypeError):
            load_nemo_model("/nonexistent/path", "/export/path")


class TestGetTokenizer:
    """Test cases for get_tokenizer function."""

    def test_get_tokenizer_nemo2_context(self, tmp_path):
        """Test getting tokenizer from NeMo 2.0 context."""
        tokenizer_dir = tmp_path / "tokenizer"
        tokenizer_dir.mkdir()
        (tokenizer_dir / "nemo_context").mkdir()

        with patch("nemo_export.trt_llm.nemo_ckpt_loader.nemo_file.get_tokenizer_from_nemo2_context") as mock_get:
            mock_tokenizer = Mock()
            mock_get.return_value = mock_tokenizer

            result = get_tokenizer(tokenizer_dir)

            assert result == mock_tokenizer

    def test_get_tokenizer_huggingface(self, tmp_path):
        """Test getting HuggingFace tokenizer."""
        tokenizer_dir = tmp_path / "tokenizer"
        tokenizer_dir.mkdir()
        (tokenizer_dir / "tokenizer_config.json").touch()

        with patch("nemo_export.trt_llm.nemo_ckpt_loader.nemo_file.AutoTokenizer") as mock_auto:
            mock_tokenizer = Mock()
            mock_auto.from_pretrained.return_value = mock_tokenizer

            result = get_tokenizer(tokenizer_dir)

            assert result == mock_tokenizer

    def test_get_tokenizer_tiktoken(self, tmp_path):
        """Test getting Tiktoken tokenizer."""
        tokenizer_dir = tmp_path / "tokenizer"
        tokenizer_dir.mkdir()
        (tokenizer_dir / "vocab.json").touch()

        with patch("nemo_export.trt_llm.nemo_ckpt_loader.nemo_file.build_tokenizer") as mock_build:
            mock_tokenizer = Mock()
            mock_build.return_value = mock_tokenizer

            result = get_tokenizer(tokenizer_dir)

            assert result == mock_tokenizer

    def test_get_tokenizer_sentencepiece(self, tmp_path):
        """Test getting SentencePiece tokenizer."""
        tokenizer_dir = tmp_path / "tokenizer"
        tokenizer_dir.mkdir()
        (tokenizer_dir / "tokenizer.model").touch()

        with patch("nemo_export.trt_llm.nemo_ckpt_loader.nemo_file.build_tokenizer") as mock_build:
            mock_tokenizer = Mock()
            mock_build.return_value = mock_tokenizer

            result = get_tokenizer(tokenizer_dir)

            assert result == mock_tokenizer


if __name__ == "__main__":
    pytest.main([__file__])
