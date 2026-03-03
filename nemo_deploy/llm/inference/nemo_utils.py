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

Standalone utilities (MCoreTokenizerWrappper, checkpoint path helpers) are
copied directly and have no dependency on the nemo package.

Complex types that are tightly coupled to NeMo's class hierarchy and
serialization system (GPTConfig, T5Config, io, set_modelopt_spec_if_exists_in_ckpt)
are re-exported here from the nemo package so that inference_base.py and
tron_utils.py do not need to import from nemo directly.

Sources:
  - MCoreTokenizerWrappper  : nemo/collections/llm/inference/base.py
  - ckpt_to_dir,
    idempotent_path_append,
    ckpt_to_context_subdir  : nemo/lightning/ckpt_utils.py
  - ckpt_to_weights_subdir  : nemo/lightning/io/pl.py
  - constants               : nemo/lightning/ckpt_utils.py
"""

import inspect
from pathlib import Path
from typing import Any, Union

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
# NeMo complex types
#
# GPTConfig, T5Config, io, and set_modelopt_spec_if_exists_in_ckpt are
# deeply coupled to NeMo's class hierarchy and serialization system.
# Checkpoints saved by NeMo contain instances of these exact classes, so
# they must originate from the nemo package to preserve isinstance()
# compatibility.  They are re-exported here so that inference_base.py and
# tron_utils.py do not need to import from nemo directly.
# ---------------------------------------------------------------------------

try:
    from nemo.collections.llm.gpt.model.base import GPTConfig
    from nemo.collections.llm.modelopt import set_modelopt_spec_if_exists_in_ckpt
    from nemo.collections.llm.t5.model.t5 import T5Config
    from nemo.lightning import io

    HAVE_NEMO = True
except (ImportError, ModuleNotFoundError):
    GPTConfig = Any
    T5Config = Any
    io = None
    set_modelopt_spec_if_exists_in_ckpt = None
    HAVE_NEMO = False
