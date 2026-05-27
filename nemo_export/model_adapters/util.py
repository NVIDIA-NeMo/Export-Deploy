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


import types

import torch


def replace_onnx_safe_bidirectional_mask(model: torch.nn.Module) -> None:
    """Replace `_create_bidirectional_mask` on a Llama-bidirectional remote-code model.

    transformers>=5.0 routes bidirectional mask construction through
    `masking_utils.sdpa_mask`, which under `torch.onnx.export` tracing collapses
    `q_length` to a 0-d symbolic tensor and raises
    `IndexError: tuple index out of range`. The override builds an SDPA-compatible
    4D mask directly so `_preprocess_mask_arguments` early-exits before reaching
    the broken path. The same mask shape is used in eager mode, so behavior is
    unchanged outside of ONNX export.
    """
    target = getattr(model, "model", model)
    if not hasattr(target, "_create_bidirectional_mask"):
        return

    def _create_bidirectional_mask(self, input_embeds, attention_mask):
        if attention_mask is None:
            return None
        if attention_mask.dim() == 4:
            return attention_mask
        if getattr(self.config, "_attn_implementation", None) == "flash_attention_2":
            has_masked = (attention_mask == 0).any()
            return attention_mask if has_masked else None
        return attention_mask.to(torch.bool)[:, None, None, :]

    target._create_bidirectional_mask = types.MethodType(_create_bidirectional_mask, target)
