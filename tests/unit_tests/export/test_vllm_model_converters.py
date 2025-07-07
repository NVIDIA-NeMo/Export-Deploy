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

from unittest.mock import MagicMock, patch
import pytest
import torch
import numpy as np


class TestModelConverter:
    """Test the abstract ModelConverter class"""

    @pytest.mark.run_only_on("GPU")
    def test_init(self):
        from nemo_export.vllm.model_converters import ModelConverter
        """Test ModelConverter initialization"""
        # Can't instantiate abstract class directly
        with pytest.raises(TypeError):
            ModelConverter("test")

    @pytest.mark.run_only_on("GPU")
    def test_concrete_converter_init(self):
        """Test concrete converter initialization"""
        from nemo_export.vllm.model_converters import LlamaConverter

        converter = LlamaConverter("llama")
        assert converter.model_type == "llama"


class TestLlamaConverter:
    """Test the LlamaConverter class"""

    @pytest.mark.run_only_on("GPU")
    def test_get_architecture_llama(self):
        """Test get_architecture for llama model"""
        from nemo_export.vllm.model_converters import LlamaConverter
        
        converter = LlamaConverter("llama")
        assert converter.get_architecture() == "LlamaForCausalLM"

    @pytest.mark.run_only_on("GPU")
    def test_get_architecture_mistral(self):
        """Test get_architecture for mistral model"""
        from nemo_export.vllm.model_converters import LlamaConverter
        
        converter = LlamaConverter("mistral")
        assert converter.get_architecture() == "MistralForCausalLM"

    @pytest.mark.run_only_on("GPU")
    def test_get_architecture_unknown(self):
        """Test get_architecture for unknown model"""
        from nemo_export.vllm.model_converters import LlamaConverter
        
        converter = LlamaConverter("unknown")
        assert converter.get_architecture() is None

    @pytest.mark.run_only_on("GPU")
    def test_requires_bos_token(self):
        """Test requires_bos_token method"""
        from nemo_export.vllm.model_converters import LlamaConverter
        
        converter = LlamaConverter("llama")
        assert converter.requires_bos_token() is True

    @pytest.mark.run_only_on("GPU")
    def test_convert_config_default(self):
        """Test convert_config with default implementation"""
        from nemo_export.vllm.model_converters import LlamaConverter
        
        converter = LlamaConverter("llama")
        nemo_config = {"test": "value"}
        hf_config = {"existing": "config"}
        
        # Should not modify hf_config for base LlamaConverter
        converter.convert_config(nemo_config, hf_config)
        assert hf_config == {"existing": "config"}

    @pytest.mark.run_only_on("GPU")
    def test_convert_weights(self):
        """Test convert_weights method"""
        from nemo_export.vllm.model_converters import LlamaConverter
        
        converter = LlamaConverter("llama")
        
        # Mock configuration
        nemo_config = {
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_query_groups": 8,
            "num_layers": 2,
            "share_embeddings_and_output_weights": False,
        }
        
        # Create mock state dict with realistic tensor shapes
        state_dict = {
            "model.embedding.word_embeddings.weight": torch.randn(32000, 4096),
            "model.decoder.final_layernorm.weight": torch.randn(4096),
            "model.output_layer.weight": torch.randn(32000, 4096),
            "model.decoder.layers.self_attention.linear_qkv.weight": [
                torch.randn(48 * 128, 4096),  # layer 0: (32 + 2*8) * 128
                torch.randn(48 * 128, 4096),  # layer 1
            ],
            "model.decoder.layers.self_attention.linear_proj.weight": [
                torch.randn(4096, 4096),  # layer 0
                torch.randn(4096, 4096),  # layer 1
            ],
            "model.decoder.layers.mlp.linear_fc1.weight": [
                torch.randn(2 * 11008, 4096),  # layer 0: gate_proj + up_proj
                torch.randn(2 * 11008, 4096),  # layer 1
            ],
            "model.decoder.layers.mlp.linear_fc2.weight": [
                torch.randn(4096, 11008),  # layer 0
                torch.randn(4096, 11008),  # layer 1
            ],
            "model.decoder.layers.self_attention.linear_qkv.layer_norm_weight": [
                torch.randn(4096),  # layer 0
                torch.randn(4096),  # layer 1
            ],
            "model.decoder.layers.mlp.linear_fc1.layer_norm_weight": [
                torch.randn(4096),  # layer 0
                torch.randn(4096),  # layer 1
            ],
        }
        
        # Convert weights and collect results
        converted_weights = list(converter.convert_weights(nemo_config, state_dict))
        
        # Check that we have the expected number of weights
        expected_weights = [
            "model.embed_tokens.weight",
            "model.norm.weight",
            "lm_head.weight",
        ]
        
        # Add layer-specific weights
        for layer in range(2):
            expected_weights.extend([
                f"model.layers.{layer}.self_attn.q_proj.weight",
                f"model.layers.{layer}.self_attn.k_proj.weight",
                f"model.layers.{layer}.self_attn.v_proj.weight",
                f"model.layers.{layer}.self_attn.o_proj.weight",
                f"model.layers.{layer}.mlp.gate_proj.weight",
                f"model.layers.{layer}.mlp.up_proj.weight",
                f"model.layers.{layer}.mlp.down_proj.weight",
                f"model.layers.{layer}.input_layernorm.weight",
                f"model.layers.{layer}.post_attention_layernorm.weight",
            ])
        
        # Check that all expected weights are present
        weight_names = [name for name, _ in converted_weights]
        assert len(weight_names) == len(expected_weights)
        
        for expected_name in expected_weights:
            assert expected_name in weight_names

    @pytest.mark.run_only_on("GPU")
    def test_convert_weights_shared_embeddings(self):
        """Test convert_weights with shared embeddings"""
        from nemo_export.vllm.model_converters import LlamaConverter
        
        converter = LlamaConverter("llama")
        
        nemo_config = {
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_query_groups": 8,
            "num_layers": 1,
            "share_embeddings_and_output_weights": True,
        }
        
        state_dict = {
            "model.embedding.word_embeddings.weight": torch.randn(32000, 4096),
            "model.decoder.final_layernorm.weight": torch.randn(4096),
            "model.decoder.layers.self_attention.linear_qkv.weight": [torch.randn(48 * 128, 4096)],
            "model.decoder.layers.self_attention.linear_proj.weight": [torch.randn(4096, 4096)],
            "model.decoder.layers.mlp.linear_fc1.weight": [torch.randn(2 * 11008, 4096)],
            "model.decoder.layers.mlp.linear_fc2.weight": [torch.randn(4096, 11008)],
            "model.decoder.layers.self_attention.linear_qkv.layer_norm_weight": [torch.randn(4096)],
            "model.decoder.layers.mlp.linear_fc1.layer_norm_weight": [torch.randn(4096)],
        }
        
        converted_weights = list(converter.convert_weights(nemo_config, state_dict))
        weight_names = [name for name, _ in converted_weights]
        
        # Should not have lm_head.weight when embeddings are shared
        assert "lm_head.weight" not in weight_names
        assert "model.embed_tokens.weight" in weight_names


class TestMixtralConverter:
    """Test the MixtralConverter class"""

    @pytest.mark.run_only_on("GPU")
    def test_get_architecture(self):
        """Test get_architecture for mixtral model"""
        from nemo_export.vllm.model_converters import MixtralConverter
        
        converter = MixtralConverter("mixtral")
        assert converter.get_architecture() == "MixtralForCausalLM"

    @pytest.mark.run_only_on("GPU")
    def test_get_architecture_unknown(self):
        """Test get_architecture for unknown model"""
        from nemo_export.vllm.model_converters import MixtralConverter
        
        converter = MixtralConverter("unknown")
        assert converter.get_architecture() is None

    @pytest.mark.run_only_on("GPU")
    def test_requires_bos_token(self):
        """Test requires_bos_token method"""
        from nemo_export.vllm.model_converters import MixtralConverter
        
        converter = MixtralConverter("mixtral")
        assert converter.requires_bos_token() is True

    @pytest.mark.run_only_on("GPU")
    def test_convert_weights(self):
        """Test convert_weights method"""
        from nemo_export.vllm.model_converters import MixtralConverter
        
        converter = MixtralConverter("mixtral")
        
        nemo_config = {
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_query_groups": 8,
            "num_layers": 1,
            "num_moe_experts": 8,
        }
        
        state_dict = {
            "model.embedding.word_embeddings.weight": torch.randn(32000, 4096),
            "model.decoder.final_layernorm.weight": torch.randn(4096),
            "model.output_layer.weight": torch.randn(32000, 4096),
            "model.decoder.layers.self_attention.linear_qkv.weight": [torch.randn(48 * 128, 4096)],
            "model.decoder.layers.self_attention.linear_proj.weight": [torch.randn(4096, 4096)],
            "model.decoder.layers.mlp.router.weight": [torch.randn(8, 4096)],
            "model.decoder.layers.mlp.experts.experts.linear_fc1.weight": [
                [torch.randn(2 * 14336, 4096) for _ in range(8)]  # 8 experts
            ],
            "model.decoder.layers.mlp.experts.experts.linear_fc2.weight": [
                [torch.randn(4096, 14336) for _ in range(8)]  # 8 experts
            ],
            "model.decoder.layers.self_attention.linear_qkv.layer_norm_weight": [torch.randn(4096)],
            "model.decoder.layers.pre_mlp_layernorm.weight": [torch.randn(4096)],
        }
        
        converted_weights = list(converter.convert_weights(nemo_config, state_dict))
        weight_names = [name for name, _ in converted_weights]
        
        # Check MoE-specific weights
        assert "model.layers.0.block_sparse_moe.gate.weight" in weight_names
        
        # Check expert weights
        for expert in range(8):
            assert f"model.layers.0.block_sparse_moe.experts.{expert}.w1.weight" in weight_names
            assert f"model.layers.0.block_sparse_moe.experts.{expert}.w2.weight" in weight_names
            assert f"model.layers.0.block_sparse_moe.experts.{expert}.w3.weight" in weight_names


class TestGemmaConverter:
    """Test the GemmaConverter class"""

    @pytest.mark.run_only_on("GPU")
    def test_get_architecture(self):
        """Test get_architecture for gemma model"""
        from nemo_export.vllm.model_converters import GemmaConverter
        
        converter = GemmaConverter("gemma")
        assert converter.get_architecture() == "GemmaForCausalLM"

    @pytest.mark.run_only_on("GPU")
    def test_get_architecture_unknown(self):
        """Test get_architecture for unknown model"""
        from nemo_export.vllm.model_converters import GemmaConverter
        
        converter = GemmaConverter("unknown")
        assert converter.get_architecture() is None

    @pytest.mark.run_only_on("GPU")
    def test_requires_bos_token(self):
        """Test requires_bos_token method"""
        from nemo_export.vllm.model_converters import GemmaConverter
        
        converter = GemmaConverter("gemma")
        assert converter.requires_bos_token() is True

    @pytest.mark.run_only_on("GPU")
    def test_convert_weights(self):
        """Test convert_weights method"""
        from nemo_export.vllm.model_converters import GemmaConverter
        
        converter = GemmaConverter("gemma")
        
        nemo_config = {
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_query_groups": 8,
            "num_layers": 1,
            "kv_channels": 128,
            "layernorm_zero_centered_gamma": False,
        }
        
        state_dict = {
            "model.embedding.word_embeddings.weight": torch.randn(32000, 4096),
            "model.decoder.final_layernorm.weight": torch.randn(4096),
            "model.decoder.layers.self_attention.linear_qkv.weight": [torch.randn(48 * 128, 4096)],
            "model.decoder.layers.self_attention.linear_proj.weight": [torch.randn(4096, 4096)],
            "model.decoder.layers.mlp.linear_fc1.weight": [torch.randn(2 * 11008, 4096)],
            "model.decoder.layers.mlp.linear_fc2.weight": [torch.randn(4096, 11008)],
            "model.decoder.layers.self_attention.linear_qkv.layer_norm_weight": [torch.randn(4096)],
            "model.decoder.layers.mlp.linear_fc1.layer_norm_weight": [torch.randn(4096)],
        }
        
        converted_weights = list(converter.convert_weights(nemo_config, state_dict))
        weight_names = [name for name, _ in converted_weights]
        
        # Check Gemma-specific weight names
        assert "model.layers.0.mlp.gate_proj.weight" in weight_names
        assert "model.layers.0.mlp.up_proj.weight" in weight_names
        assert "model.layers.0.mlp.down_proj.weight" in weight_names

    @pytest.mark.run_only_on("GPU")
    def test_convert_weights_zero_centered_gamma(self):
        """Test convert_weights with zero-centered gamma"""
        from nemo_export.vllm.model_converters import GemmaConverter
        
        converter = GemmaConverter("gemma")
        
        nemo_config = {
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_query_groups": 8,
            "num_layers": 1,
            "kv_channels": 128,
            "layernorm_zero_centered_gamma": True,
        }
        
        state_dict = {
            "model.embedding.word_embeddings.weight": torch.randn(32000, 4096),
            "model.decoder.final_layernorm.weight": torch.ones(4096),
            "model.decoder.layers.self_attention.linear_qkv.weight": [torch.randn(48 * 128, 4096)],
            "model.decoder.layers.self_attention.linear_proj.weight": [torch.randn(4096, 4096)],
            "model.decoder.layers.mlp.linear_fc1.weight": [torch.randn(2 * 11008, 4096)],
            "model.decoder.layers.mlp.linear_fc2.weight": [torch.randn(4096, 11008)],
            "model.decoder.layers.self_attention.linear_qkv.layer_norm_weight": [torch.ones(4096)],
            "model.decoder.layers.mlp.linear_fc1.layer_norm_weight": [torch.ones(4096)],
        }
        
        converted_weights = list(converter.convert_weights(nemo_config, state_dict))
        
        # When zero_centered_gamma is True, we shouldn't subtract 1.0 from layer norm weights
        # Find the final layernorm weight
        final_norm_weight = None
        for name, weight in converted_weights:
            if name == "model.norm.weight":
                final_norm_weight = weight
                break
        
        assert final_norm_weight is not None
        # Original weight is all ones, so should remain ones when zero_centered_gamma is True
        assert torch.allclose(final_norm_weight, torch.ones(4096))


class TestStarcoder2Converter:
    """Test the Starcoder2Converter class"""

    @pytest.mark.run_only_on("GPU")
    def test_get_architecture(self):
        """Test get_architecture for starcoder2 model"""
        from nemo_export.vllm.model_converters import Starcoder2Converter
        
        converter = Starcoder2Converter("starcoder2")
        assert converter.get_architecture() == "Starcoder2ForCausalLM"

    @pytest.mark.run_only_on("GPU")
    def test_get_architecture_unknown(self):
        """Test get_architecture for unknown model"""
        from nemo_export.vllm.model_converters import Starcoder2Converter
        
        converter = Starcoder2Converter("unknown")
        assert converter.get_architecture() is None

    @pytest.mark.run_only_on("GPU")
    def test_requires_bos_token(self):
        """Test requires_bos_token method"""
        from nemo_export.vllm.model_converters import Starcoder2Converter
        
        converter = Starcoder2Converter("starcoder2")
        assert converter.requires_bos_token() is False

    @pytest.mark.run_only_on("GPU")
    def test_convert_config(self):
        """Test convert_config method"""
        from nemo_export.vllm.model_converters import Starcoder2Converter
        
        converter = Starcoder2Converter("starcoder2")
        
        nemo_config = {"window_size": [2048, 2048]}
        hf_config = {}
        
        converter.convert_config(nemo_config, hf_config)
        
        assert hf_config["sliding_window"] == 2048
        assert hf_config["tie_word_embeddings"] is False

    @pytest.mark.run_only_on("GPU")
    def test_convert_config_no_window_size(self):
        """Test convert_config without window_size"""
        from nemo_export.vllm.model_converters import Starcoder2Converter
        
        converter = Starcoder2Converter("starcoder2")
        
        nemo_config = {}
        hf_config = {}
        
        converter.convert_config(nemo_config, hf_config)
        
        assert "sliding_window" not in hf_config
        assert hf_config["tie_word_embeddings"] is False

    @pytest.mark.run_only_on("GPU")
    def test_convert_weights_with_bias(self):
        """Test convert_weights with bias enabled"""
        from nemo_export.vllm.model_converters import Starcoder2Converter
        
        converter = Starcoder2Converter("starcoder2")
        
        nemo_config = {
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_query_groups": 8,
            "num_layers": 1,
            "bias": True,
        }
        
        state_dict = {
            "model.embedding.word_embeddings.weight": torch.randn(32000, 4096),
            "model.decoder.final_layernorm.weight": torch.randn(4096),
            "model.decoder.final_layernorm.bias": torch.randn(4096),
            "model.output_layer.weight": torch.randn(32000, 4096),
            "model.decoder.layers.self_attention.linear_qkv.weight": [torch.randn(48 * 128, 4096)],
            "model.decoder.layers.self_attention.linear_qkv.bias": [torch.randn(48 * 128)],
            "model.decoder.layers.self_attention.linear_proj.weight": [torch.randn(4096, 4096)],
            "model.decoder.layers.self_attention.linear_proj.bias": [torch.randn(4096)],
            "model.decoder.layers.mlp.linear_fc1.weight": [torch.randn(11008, 4096)],
            "model.decoder.layers.mlp.linear_fc1.bias": [torch.randn(11008)],
            "model.decoder.layers.mlp.linear_fc2.weight": [torch.randn(4096, 11008)],
            "model.decoder.layers.mlp.linear_fc2.bias": [torch.randn(4096)],
            "model.decoder.layers.self_attention.linear_qkv.layer_norm_weight": [torch.randn(4096)],
            "model.decoder.layers.self_attention.linear_qkv.layer_norm_bias": [torch.randn(4096)],
            "model.decoder.layers.mlp.linear_fc1.layer_norm_weight": [torch.randn(4096)],
            "model.decoder.layers.mlp.linear_fc1.layer_norm_bias": [torch.randn(4096)],
        }
        
        converted_weights = list(converter.convert_weights(nemo_config, state_dict))
        weight_names = [name for name, _ in converted_weights]
        
        # Check that bias terms are included
        assert "model.norm.bias" in weight_names
        assert "model.layers.0.self_attn.q_proj.bias" in weight_names
        assert "model.layers.0.self_attn.k_proj.bias" in weight_names
        assert "model.layers.0.self_attn.v_proj.bias" in weight_names
        assert "model.layers.0.self_attn.o_proj.bias" in weight_names
        assert "model.layers.0.mlp.c_fc.bias" in weight_names
        assert "model.layers.0.mlp.c_proj.bias" in weight_names
        assert "model.layers.0.input_layernorm.bias" in weight_names
        assert "model.layers.0.post_attention_layernorm.bias" in weight_names

    @pytest.mark.run_only_on("GPU")
    def test_convert_weights_without_bias(self):
        """Test convert_weights without bias"""
        from nemo_export.vllm.model_converters import Starcoder2Converter

        converter = Starcoder2Converter("starcoder2")
        
        nemo_config = {
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_query_groups": 8,
            "num_layers": 1,
            "add_bias_linear": False,
        }
        
        state_dict = {
            "model.embedding.word_embeddings.weight": torch.randn(32000, 4096),
            "model.decoder.final_layernorm.weight": torch.randn(4096),
            "model.output_layer.weight": torch.randn(32000, 4096),
            "model.decoder.layers.self_attention.linear_qkv.weight": [torch.randn(48 * 128, 4096)],
            "model.decoder.layers.self_attention.linear_proj.weight": [torch.randn(4096, 4096)],
            "model.decoder.layers.mlp.linear_fc1.weight": [torch.randn(11008, 4096)],
            "model.decoder.layers.mlp.linear_fc2.weight": [torch.randn(4096, 11008)],
            "model.decoder.layers.self_attention.linear_qkv.layer_norm_weight": [torch.randn(4096)],
            "model.decoder.layers.mlp.linear_fc1.layer_norm_weight": [torch.randn(4096)],
        }
        
        converted_weights = list(converter.convert_weights(nemo_config, state_dict))
        weight_names = [name for name, _ in converted_weights]
        
        # Check that bias terms are NOT included
        assert "model.norm.bias" not in weight_names
        assert "model.layers.0.self_attn.q_proj.bias" not in weight_names
        assert "model.layers.0.mlp.c_fc.bias" not in weight_names


class TestModelConverterRegistry:
    """Test the model converter registry functions"""

    @pytest.mark.run_only_on("GPU")
    def test_register_model_converter(self):
        """Test register_model_converter function"""
        from nemo_export.vllm.model_converters import ModelConverter, _MODEL_CONVERTERS, register_model_converter
        
        class TestConverter(ModelConverter):
            def get_architecture(self):
                return "TestForCausalLM"
            
            def convert_weights(self, nemo_model_config, state_dict):
                yield ("test.weight", torch.randn(10, 10))
        
        # Register the converter
        register_model_converter("test_model", TestConverter)
        
        # Check that it was registered
        assert "test_model" in _MODEL_CONVERTERS
        assert _MODEL_CONVERTERS["test_model"] == TestConverter
        
        # Clean up
        del _MODEL_CONVERTERS["test_model"]

    @pytest.mark.run_only_on("GPU")
    def test_get_model_converter_existing(self):
        """Test get_model_converter for existing model type"""
        from nemo_export.vllm.model_converters import get_model_converter, LlamaConverter
        
        converter = get_model_converter("llama")
        assert isinstance(converter, LlamaConverter)
        assert converter.model_type == "llama"

    @pytest.mark.run_only_on("GPU")
    def test_get_model_converter_nonexistent(self):
        """Test get_model_converter for non-existent model type"""
        from nemo_export.vllm.model_converters import get_model_converter
        
        converter = get_model_converter("nonexistent")
        assert converter is None

    @pytest.mark.run_only_on("GPU")
    def test_get_model_converter_all_types(self):
        """Test get_model_converter for all registered types"""
        from nemo_export.vllm.model_converters import LlamaConverter, MixtralConverter, GemmaConverter, Starcoder2Converter, get_model_converter
        
        expected_types = {
            "llama": LlamaConverter,
            "mistral": LlamaConverter,
            "mixtral": MixtralConverter,
            "gemma": GemmaConverter,
            "starcoder2": Starcoder2Converter,
        }
        
        for model_type, expected_class in expected_types.items():
            converter = get_model_converter(model_type)
            assert isinstance(converter, expected_class)
            assert converter.model_type == model_type


class TestModelConverterIntegration:
    """Integration tests for model converters"""

    @pytest.mark.run_only_on("GPU")
    def test_all_converters_have_required_methods(self):
        """Test that all registered converters implement required methods"""
        from nemo_export.vllm.model_converters import _MODEL_CONVERTERS, get_model_converter
        
        for model_type in _MODEL_CONVERTERS:
            converter = get_model_converter(model_type)
            assert hasattr(converter, "get_architecture")
            assert hasattr(converter, "convert_weights")
            assert hasattr(converter, "requires_bos_token")
            assert hasattr(converter, "convert_config")
            
            # Test that methods return expected types
            arch = converter.get_architecture()
            assert arch is None or isinstance(arch, str)
            
            assert isinstance(converter.requires_bos_token(), bool)

    @pytest.mark.run_only_on("GPU")
    def test_weight_conversion_tensor_types(self):
        """Test that weight conversion yields proper tensor types"""
        from nemo_export.vllm.model_converters import LlamaConverter
        
        converter = LlamaConverter("llama")
        
        nemo_config = {
            "hidden_size": 64,
            "num_attention_heads": 8,
            "num_query_groups": 2,
            "num_layers": 1,
            "share_embeddings_and_output_weights": False,
        }
        
        # Minimal state dict
        state_dict = {
            "model.embedding.word_embeddings.weight": torch.randn(100, 64),
            "model.decoder.final_layernorm.weight": torch.randn(64),
            "model.output_layer.weight": torch.randn(100, 64),
            "model.decoder.layers.self_attention.linear_qkv.weight": [torch.randn(12 * 8, 64)],
            "model.decoder.layers.self_attention.linear_proj.weight": [torch.randn(64, 64)],
            "model.decoder.layers.mlp.linear_fc1.weight": [torch.randn(2 * 256, 64)],
            "model.decoder.layers.mlp.linear_fc2.weight": [torch.randn(64, 256)],
            "model.decoder.layers.self_attention.linear_qkv.layer_norm_weight": [torch.randn(64)],
            "model.decoder.layers.mlp.linear_fc1.layer_norm_weight": [torch.randn(64)],
        }
        
        for name, tensor in converter.convert_weights(nemo_config, state_dict):
            assert isinstance(name, str)
            assert isinstance(tensor, torch.Tensor)
            assert tensor.dtype == torch.float32  # Default dtype 