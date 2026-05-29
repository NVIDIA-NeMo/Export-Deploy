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

import torch

from nemo_export.model_adapters.masking import patch_bidirectional_mask_for_export


class _BidirectionalModule(torch.nn.Module):
    """Minimal stand-in for a LlamaBidirectionalModel exposing the patched method."""

    def _create_bidirectional_mask(self, input_embeds, attention_mask):
        # Original (sentinel) implementation that the patch must replace.
        return "original"


class _Wrapper(torch.nn.Module):
    """Stand-in for the reranker layout where the method lives on a nested backbone."""

    def __init__(self):
        super().__init__()
        self.model = _BidirectionalModule()


class TestPatchBidirectionalMaskForExport:
    """Test cases for patch_bidirectional_mask_for_export."""

    def test_patches_top_level_module(self):
        model = _BidirectionalModule()
        assert patch_bidirectional_mask_for_export(model) is True

        input_embeds = torch.zeros(1, 3, 4)
        # The replacement no longer returns the sentinel.
        assert model._create_bidirectional_mask(input_embeds, None) is None

    def test_patches_nested_module(self):
        wrapper = _Wrapper()
        assert patch_bidirectional_mask_for_export(wrapper) is True

        input_embeds = torch.zeros(1, 3, 4)
        assert wrapper.model._create_bidirectional_mask(input_embeds, None) is None

    def test_returns_false_when_method_absent(self):
        model = torch.nn.Linear(4, 4)
        assert patch_bidirectional_mask_for_export(model) is False

    def test_mask_none_returns_none(self):
        model = _BidirectionalModule()
        patch_bidirectional_mask_for_export(model)
        assert model._create_bidirectional_mask(torch.zeros(1, 2, 4), None) is None

    def test_additive_mask_values(self):
        model = _BidirectionalModule()
        patch_bidirectional_mask_for_export(model)

        dtype = torch.float32
        input_embeds = torch.zeros(1, 3, 4, dtype=dtype)
        attention_mask = torch.tensor([[1, 1, 0]])

        mask = model._create_bidirectional_mask(input_embeds, attention_mask)

        assert mask.shape == (1, 1, 1, 3)
        assert mask.dtype == dtype
        # Real positions are unmasked (0.0); the padded position gets the dtype minimum.
        assert mask[0, 0, 0, 0].item() == 0.0
        assert mask[0, 0, 0, 1].item() == 0.0
        assert mask[0, 0, 0, 2].item() == torch.finfo(dtype).min

    def test_additive_mask_matches_input_dtype(self):
        model = _BidirectionalModule()
        patch_bidirectional_mask_for_export(model)

        input_embeds = torch.zeros(1, 2, 4, dtype=torch.float16)
        attention_mask = torch.tensor([[1, 0]])

        mask = model._create_bidirectional_mask(input_embeds, attention_mask)
        assert mask.dtype == torch.float16
        assert mask[0, 0, 0, 1].item() == torch.finfo(torch.float16).min
