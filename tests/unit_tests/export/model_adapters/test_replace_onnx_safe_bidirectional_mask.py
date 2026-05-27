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

from types import SimpleNamespace

import torch

from nemo_export.model_adapters.util import replace_onnx_safe_bidirectional_mask


class _FakeBidirectionalModel(torch.nn.Module):
    def __init__(self, attn_implementation: str = "sdpa"):
        super().__init__()
        self.config = SimpleNamespace(_attn_implementation=attn_implementation)

    def _create_bidirectional_mask(self, input_embeds, attention_mask):
        raise RuntimeError("original implementation should have been replaced")


class _Wrapper(torch.nn.Module):
    def __init__(self, inner: torch.nn.Module):
        super().__init__()
        self.model = inner


def test_override_returns_4d_mask_for_sdpa():
    model = _FakeBidirectionalModel()
    replace_onnx_safe_bidirectional_mask(model)

    batch, seq = 2, 5
    input_embeds = torch.zeros(batch, seq, 8)
    attention_mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 0]])

    mask = model._create_bidirectional_mask(input_embeds, attention_mask)
    assert mask.dim() == 4
    assert mask.shape == (batch, 1, 1, seq)
    assert mask.dtype == torch.bool
    assert mask[0, 0, 0].tolist() == [True, True, True, False, False]


def test_override_passes_through_4d_mask_unchanged():
    model = _FakeBidirectionalModel()
    replace_onnx_safe_bidirectional_mask(model)

    input_embeds = torch.zeros(1, 3, 4)
    existing = torch.ones(1, 1, 3, 3, dtype=torch.bool)
    assert model._create_bidirectional_mask(input_embeds, existing) is existing


def test_override_returns_none_when_mask_is_none():
    model = _FakeBidirectionalModel()
    replace_onnx_safe_bidirectional_mask(model)
    assert model._create_bidirectional_mask(torch.zeros(1, 2, 3), None) is None


def test_override_preserves_2d_mask_for_flash_attention_2():
    model = _FakeBidirectionalModel(attn_implementation="flash_attention_2")
    replace_onnx_safe_bidirectional_mask(model)

    input_embeds = torch.zeros(1, 4, 3)
    padded = torch.tensor([[1, 1, 0, 0]])
    assert torch.equal(model._create_bidirectional_mask(input_embeds, padded), padded)

    unpadded = torch.tensor([[1, 1, 1, 1]])
    assert model._create_bidirectional_mask(input_embeds, unpadded) is None


def test_override_targets_inner_model_when_wrapped():
    inner = _FakeBidirectionalModel()
    wrapper = _Wrapper(inner)
    replace_onnx_safe_bidirectional_mask(wrapper)

    input_embeds = torch.zeros(1, 2, 3)
    attention_mask = torch.tensor([[1, 0]])

    mask = wrapper.model._create_bidirectional_mask(input_embeds, attention_mask)
    assert mask.shape == (1, 1, 1, 2)
    assert mask[0, 0, 0].tolist() == [True, False]


def test_override_noop_when_method_missing():
    class _PlainModel(torch.nn.Module):
        pass

    model = _PlainModel()
    replace_onnx_safe_bidirectional_mask(model)
    assert not hasattr(model, "_create_bidirectional_mask")
