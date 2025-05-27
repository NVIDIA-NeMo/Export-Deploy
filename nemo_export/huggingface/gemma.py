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

from pathlib import Path
from typing import TYPE_CHECKING

import torch

from nemo_export.huggingface.lightning import (
    ModelConnector,
    TransformFns,
    _ModelState,
    apply_transforms,
    state_transform,
)
from nemo_export.huggingface.utils import (
    ckpt_load,
    get_model,
    get_tokenizer,
    io_model_exporter,
    load_config,
    torch_dtype_from_mcore_config,
)

if TYPE_CHECKING:
    from transformers import GemmaConfig, GemmaForCausalLM

GemmaModel = get_model("GemmaModel")


@io_model_exporter(GemmaModel, "hf", register=False)
class HFGemmaExporter(ModelConnector["GemmaModel", "GemmaForCausalLM"]):
    """ """

    def init(self, torch_dtype: torch.dtype = torch.bfloat16) -> "GemmaForCausalLM":
        """
        Initializes the target HF model.
        """
        from transformers import AutoModelForCausalLM
        from transformers.modeling_utils import no_init_weights

        with no_init_weights():
            return AutoModelForCausalLM.from_config(self.config, torch_dtype=torch_dtype)

    def apply(self, output_path: Path) -> Path:
        """
        Transforms the source model state into the target HF model state.
        """
        source, source_config = ckpt_load(str(self))
        source = _ModelState(source, source_config)

        target = self.init(torch_dtype_from_mcore_config(source_config))
        target = self.convert_state(source, target)

        target = target.cpu()
        target.save_pretrained(output_path)
        self.tokenizer.tokenizer.save_pretrained(output_path)

        return output_path

    def convert_state(self, source, target):
        """
        State conversion definition.
        """
        mapping = {
            "embedding.word_embeddings.weight": "model.embed_tokens.weight",
            "decoder.layers.*.self_attention.linear_proj.weight": "model.layers.*.self_attn.o_proj.weight",
            "decoder.layers.*.mlp.linear_fc2.weight": "model.layers.*.mlp.down_proj.weight",
            "decoder.layers.*.self_attention.linear_qkv.layer_norm_weight": "model.layers.*.input_layernorm.weight",
            "decoder.layers.*.mlp.linear_fc1.layer_norm_weight": "model.layers.*.post_attention_layernorm.weight",
            "decoder.final_layernorm.weight": "model.norm.weight",
        }

        transforms = [
            state_transform(
                source_key="decoder.layers.*.self_attention.linear_qkv.weight",
                target_key=(
                    "model.layers.*.self_attn.q_proj.weight",
                    "model.layers.*.self_attn.k_proj.weight",
                    "model.layers.*.self_attn.v_proj.weight",
                ),
                fn=TransformFns.split_qkv,
            ),
            state_transform(
                source_key="decoder.layers.*.mlp.linear_fc1.weight",
                target_key=("model.layers.*.mlp.gate_proj.weight", "model.layers.*.mlp.up_proj.weight"),
                fn=TransformFns.split_fc1,
            ),
        ]

        return apply_transforms(source, target, mapping=mapping, transforms=transforms)

    @property
    def tokenizer(self):
        """
        Tokenizer initialization.
        """
        return get_tokenizer(str(self))

    @property
    def config(self) -> "GemmaConfig":
        """
        Target HF model config.
        """
        source = load_config(str(self))

        from transformers import GemmaConfig as HFGemmaConfig

        return HFGemmaConfig(
            architectures=["GemmaForCausalLM"],
            num_hidden_layers=source.num_layers,
            hidden_size=source.hidden_size,
            intermediate_size=source.ffn_hidden_size,
            num_attention_heads=source.num_attention_heads,
            head_dim=(
                source.kv_channels
                if source.kv_channels is not None
                else source.hidden_size // source.num_attention_heads
            ),
            max_position_embeddings=source.seq_length,
            initializer_range=source.init_method_std,
            rms_norm_eps=source.layernorm_epsilon,
            num_key_value_heads=source.num_query_groups,
            vocab_size=self.tokenizer.vocab_size,
        )
