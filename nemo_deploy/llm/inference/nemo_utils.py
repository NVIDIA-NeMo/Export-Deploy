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
"""NeMo utility code copied from the NeMo project.

All utilities here are copied directly from NeMo and have no static
dependency on the nemo package.  When a NeMo checkpoint is loaded at
runtime, NeMo classes are imported transitively through pydoc.locate
inside nemo_io.load_context — NeMo must therefore still be installed
to read NeMo checkpoints.

Sources:
  - MCoreTokenizerWrappper  : nemo/collections/llm/inference/base.py
  - ckpt_to_dir,
    idempotent_path_append,
    ckpt_to_context_subdir  : nemo/lightning/ckpt_utils.py
  - ckpt_to_weights_subdir  : nemo/lightning/io/pl.py
  - constants               : nemo/lightning/ckpt_utils.py
  - set_modelopt_spec_*     : nemo/collections/llm/modelopt/model_utils.py
  - load_context, io        : nemo_io.py (copied from nemo/lightning/io/)
"""

import inspect
import logging
import types
from functools import partial
from pathlib import Path
from typing import Any, Union

from .nemo_io import load_context as _load_context

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# io namespace — exposes load_context under the same attribute name that
# inference_base.py uses (io.load_context(...)).
# ---------------------------------------------------------------------------

io = types.SimpleNamespace(load_context=_load_context)

# ---------------------------------------------------------------------------
# GPTConfig / T5Config — type stubs used only for annotations.
# The actual runtime objects are NeMo classes deserialized from the
# checkpoint; isinstance() checks use class-name strings instead.
# ---------------------------------------------------------------------------

GPTConfig = Any
T5Config = Any

# ---------------------------------------------------------------------------
# Constants  (from nemo.lightning.ckpt_utils)
# ---------------------------------------------------------------------------

# NeMo-2 checkpoint structure:
#   <ckpt_dir>/weights/  – model weights
#   <ckpt_dir>/context/  – hyper-parameters / IO context
WEIGHTS_PATH: str = "weights"
CONTEXT_PATH: str = "context"
ADAPTER_META_FILENAME: str = "adapter_metadata.json"

# ---------------------------------------------------------------------------
# Checkpoint path utilities  (simplified from nemo.lightning.ckpt_utils and
# nemo.lightning.io.pl – AdapterPath and MultiStorageClient branches removed
# because they are not required for basic NeMo-2 inference).
# ---------------------------------------------------------------------------


def ckpt_to_dir(filepath: Union[str, Path]) -> Path:
    """Return the checkpoint directory path for a given filepath.

    PTL treats checkpoints as ``.ckpt`` files.  This helper strips the
    extension (appending it first when absent) and returns a :class:`Path`
    suitable for use as a distributed-checkpoint directory.

    Copied from ``nemo.lightning.ckpt_utils.ckpt_to_dir`` with the
    ``AdapterPath`` and ``MultiStorageClient`` branches removed.
    """
    filepath = Path(filepath)

    if filepath.suffix != ".ckpt":
        filepath = filepath.with_suffix(filepath.suffix + ".ckpt")

    assert filepath.suffix == ".ckpt", f"filepath: {filepath} must have .ckpt extension"

    # Return path whose name is the original filepath without the .ckpt extension.
    return filepath.with_name(filepath.stem)


def idempotent_path_append(base_dir: Union[str, Path], suffix: str) -> Path:
    """Append *suffix* to *base_dir* only when it is not already the last component.

    Copied from ``nemo.lightning.ckpt_utils.idempotent_path_append`` with the
    ``AdapterPath`` and ``MultiStorageClient`` branches removed.
    """
    base_dir = Path(base_dir)
    if base_dir.parts[-1] != suffix:
        base_dir = base_dir / suffix
    return base_dir


def ckpt_to_context_subdir(filepath: Union[str, Path]) -> Path:
    """Return the ``context`` sub-directory of a NeMo-2 checkpoint.

    Copied from ``nemo.lightning.ckpt_utils.ckpt_to_context_subdir``.
    """
    base_dir = ckpt_to_dir(filepath=filepath)
    return idempotent_path_append(base_dir, CONTEXT_PATH)


def ckpt_to_weights_subdir(filepath: Union[str, Path], is_saving: bool) -> Path:
    """Return the ``weights`` sub-directory of a NeMo-2 checkpoint.

    Copied from ``nemo.lightning.io.pl.ckpt_to_weights_subdir`` with the
    ``AdapterPath`` branch removed.
    """
    filepath = ckpt_to_dir(filepath=filepath)
    base_dir = filepath

    if base_dir.parts[-1] != WEIGHTS_PATH:
        maybe_base_dir = base_dir / WEIGHTS_PATH
        if maybe_base_dir.is_dir() or is_saving:
            base_dir = maybe_base_dir

    if is_saving:
        assert base_dir.parts[-1] == WEIGHTS_PATH
        assert base_dir.parent == filepath

    return base_dir


# ---------------------------------------------------------------------------
# MCoreTokenizerWrappper  (from nemo.collections.llm.inference.base)
# ---------------------------------------------------------------------------


class MCoreTokenizerWrappper:
    """Thin wrapper that adapts a NeMo tokenizer to the MCore generate API.

    MCore's generate pipeline expects ``tokenizer.detokenize``,
    ``tokenizer.tokenize``, ``tokenizer.bos``, and ``tokenizer.pad`` –
    this wrapper maps those calls to the corresponding NeMo tokenizer
    methods/properties.

    Copied verbatim from ``nemo.collections.llm.inference.base.MCoreTokenizerWrappper``.
    """

    def __init__(self, tokenizer, vocab_size=None):
        self.tokenizer = tokenizer
        self.eod = tokenizer.eod
        self.vocab_size = vocab_size or tokenizer.vocab_size

    def detokenize(self, tokens, remove_special_tokens=False):
        """Detokenize *tokens* into a string."""
        if "remove_special_tokens" in inspect.signature(self.tokenizer.ids_to_text).parameters:
            return self.tokenizer.ids_to_text(tokens, remove_special_tokens)
        return self.tokenizer.ids_to_text(tokens)

    def tokenize(self, prompt):
        """Tokenize *prompt* into a list of token IDs."""
        return self.tokenizer.text_to_ids(prompt)

    @property
    def additional_special_tokens_ids(self):
        """IDs of additional special tokens."""
        return self.tokenizer.additional_special_tokens_ids

    @property
    def bos(self):
        """Beginning-of-sequence token ID."""
        return self.tokenizer.bos_id

    @property
    def pad(self):
        """Padding token ID."""
        return self.tokenizer.pad_id


# ---------------------------------------------------------------------------
# set_modelopt_spec_if_exists_in_ckpt
#
# Copied from nemo/collections/llm/modelopt/model_utils.py.
# NeMo model-type isinstance checks are replaced by class-name checks to
# avoid importing nemo at module level.
# ---------------------------------------------------------------------------


def set_modelopt_spec_if_exists_in_ckpt(model, path: str) -> None:
    """Set model.config.transformer_layer_spec to a modelopt spec if the
    checkpoint contains a ``modelopt_state`` directory.

    Copied from ``nemo.collections.llm.modelopt.model_utils.set_modelopt_spec_if_exists_in_ckpt``
    with NeMo isinstance checks replaced by class-name comparisons.
    """
    path = str(path).removeprefix("nemo://")
    modelopt_state_path = ckpt_to_weights_subdir(path, is_saving=False) / "modelopt_state"
    if not modelopt_state_path.exists() or hasattr(model, "module"):
        return

    model_type_name = type(model).__name__
    if model_type_name not in ("GPTModel", "MambaModel"):
        _logger.warning(
            "%s is neither a GPTModel nor MambaModel. Modelopt state will not be loaded.",
            type(model),
        )
        return

    config = model.config
    config_type_name = type(config).__name__

    try:
        from megatron.core.post_training.modelopt.gpt.model_specs import get_gpt_modelopt_spec

        _HAVE_GPT_MODELOPT_SPEC = True
    except ImportError:
        _HAVE_GPT_MODELOPT_SPEC = False

    if config_type_name == "GPTConfig":
        if _HAVE_GPT_MODELOPT_SPEC:
            config.transformer_layer_spec = partial(
                get_gpt_modelopt_spec,
                remap_te_layernorm=True,
                local_core_attention=getattr(config, "softmax_type", "vanilla") != "vanilla",
            )
        else:
            _logger.warning("get_gpt_modelopt_spec not available; skipping modelopt layer spec.")
    elif config_type_name == "SSMConfig":
        try:
            from megatron.core.post_training.modelopt.mamba.model_specs import get_mamba_stack_modelopt_spec

            config.mamba_stack_spec = partial(get_mamba_stack_modelopt_spec, remap_te_layernorm=True)
        except ImportError:
            _logger.warning("get_mamba_stack_modelopt_spec not available; skipping modelopt layer spec.")
    else:
        _logger.warning("No modelopt layer spec supported for config type %s.", type(config))
        return

    config.gradient_accumulation_fusion = False
