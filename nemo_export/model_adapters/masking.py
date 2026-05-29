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


def patch_bidirectional_mask_for_export(model: torch.nn.Module) -> bool:
    """Override LlamaBidirectional ``_create_bidirectional_mask`` with a trace-friendly version.

    The ``create_bidirectional_mask`` helper in transformers>=5.0 is not traceable by the
    TorchScript ONNX exporter: under tracing it dispatches into ``sdpa_mask`` (even with
    ``attn_implementation="eager"``, since ``eager_mask`` reuses ``sdpa_mask``) and crashes with
    ``IndexError: tuple index out of range`` while converting the deprecated ``cache_position``
    argument. Since ONNX export uses eager attention, we build the additive 4D mask directly,
    which is numerically equivalent for fully-bidirectional attention and traces cleanly.

    The replacement is bound to every submodule that defines ``_create_bidirectional_mask`` so it
    works whether the method lives on the top-level model (embedding) or a nested backbone
    (reranker: ``LlamaBidirectionalForSequenceClassification.model``).

    Args:
        model: The loaded HuggingFace model (or wrapper) to patch in place.

    Returns:
        bool: True if at least one module was patched, False otherwise.
    """

    def _create_bidirectional_mask(self, input_embeds, attention_mask):
        if attention_mask is None:
            return None
        dtype = input_embeds.dtype
        expanded = attention_mask[:, None, None, :].to(dtype)  # (batch, 1, 1, seq_len)
        return (1.0 - expanded) * torch.finfo(dtype).min

    patched = False
    for module in model.modules():
        if hasattr(type(module), "_create_bidirectional_mask"):
            module._create_bidirectional_mask = types.MethodType(_create_bidirectional_mask, module)
            patched = True
    return patched
