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


import json
import logging
import os
import pickle
import shutil
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Optional, Union

import numpy as np
import torch
import yaml
from transformers import AutoTokenizer, GPT2Tokenizer, PreTrainedTokenizer

from nemo_export.sentencepiece_tokenizer import SentencePieceTokenizer
from nemo_export.tarutils import TarPath
from nemo_export.tiktoken_tokenizer import TiktokenTokenizer
from nemo_export.utils import (
    load_model_weights,
    nemo_to_path,
    torch_dtype_from_precision,
)

try:
    from nemo.lightning import io

    HAVE_NEMO2 = True
except (ImportError, ModuleNotFoundError):
    HAVE_NEMO2 = False

LOGGER = logging.getLogger("NeMo")
EXTRA_STATE = "extra_state"


def load_extra_state_from_bytes(
    val: Optional[Union[torch.Tensor, BytesIO]],
) -> Optional[dict]:
    """Loads single extra_state from bytes storage.

    Args:
        val (torch.Tensor | BytesIO): Bytes storage of extra_state
    Returns:
        Optional[dict]: Deserialized extra_state, or None if the bytes storage is empty.
    """
    if val is None:
        return None

    # TransformerEngine shifted from storing extra_states bytes storage from _io.BytesIO to torch.Tensor
    if isinstance(val, torch.Tensor):
        if val.numel() == 0:
            return None

        val = val.detach().numpy(force=True).tobytes()
        return pickle.loads(val)

    val.seek(0)
    return torch.load(val, weights_only=True)


def rename_extra_states(state_dict: Dict[str, Any]) -> Dict[str, Any]:
    """This function preprocesses extra states for Megatron export.

    Args:
        state_dict (dict): Model state dictionary
    Returns:
        dict: Model state dictionary, with extra states consumable by mcore export
    """
    mcore_extra_states = {}

    for key, value in state_dict.items():
        if EXTRA_STATE not in key:
            continue

        # Keys with the extra states have the following format:
        # <prefix>.layers.<layer>._extra_state/shard_<layer_number>_<number_of_layers>
        key_base, shard_key = key.split("/")
        if "_" not in shard_key:
            continue

        shard_layer = shard_key.split("_")[1]
        if not shard_layer.isnumeric():
            continue

        # Renames keys to:
        # <prefix>.layers.<layer_number>.<layer>._extra_state
        mcore_key = key_base.replace("layers", f"layers.{shard_layer}")
        if isinstance(value, list):
            value = value[0]
        mcore_extra_states[mcore_key] = value

    state_dict = {k: v for k, v in state_dict.items() if EXTRA_STATE not in k}
    return state_dict | mcore_extra_states


def update_tokenizer_paths(tokenizer_config: Dict, unpacked_checkpoints_dir):
    """Updates tokenizer paths in the tokenizer config."""

    def _update_config_entry(key, file_pattern):
        old_path = tokenizer_config.get(key, None)
        if old_path is None:
            return
        old_path = Path(old_path)
        new_path = unpacked_checkpoints_dir.get_tokenizer_file_path("tokenizer", key, file_pattern)
        if new_path:
            LOGGER.debug(f"Update tokenizer {key} {old_path} -> {new_path}")
            tokenizer_config[key] = new_path
        elif not old_path.exists():
            LOGGER.warning(f"Tokenizer {key}'s path {old_path} does not exists: set it to None")
            tokenizer_config[key] = None

    _update_config_entry("model", "*.model")
    _update_config_entry("vocab_file", "*vocab*")
    _update_config_entry("merge_file", "*merge*.txt")

    return tokenizer_config


def get_tokenizer_from_nemo2_context(model_context_dir: Path):
    """Retrieve tokenizer configuration from NeMo 2.0 context and instantiate the tokenizer.

    Args:
        model_context_dir (Path): Path to the model context directory.

    Returns:
        The instantiated tokenizer (various classes possible).
    """
    if HAVE_NEMO2:
        # Use NeMo tokenizer loaded from the NeMo 2.0 model context
        tokenizer_spec = io.load_context(model_context_dir, subpath="model.tokenizer")
        return build_tokenizer(tokenizer_spec)
    else:
        # Use local nemo_export SentencePieceTokenizer implementation
        # or directly a HuggingFace tokenizer based on the model config
        with (model_context_dir / "model.yaml").open("r") as stream:
            model_config = yaml.safe_load(stream)

        tokenizer_config = model_config["tokenizer"]
        target_class = tokenizer_config["_target_"]
        tokenizer_module = "nemo.collections.common.tokenizers."
        assert target_class.startswith(tokenizer_module)
        target_class = target_class.removeprefix(tokenizer_module)

        if target_class == "sentencepiece_tokenizer.SentencePieceTokenizer":
            tokenizer = SentencePieceTokenizer(
                model_path=str(model_context_dir / tokenizer_config["model_path"]),
                special_tokens=tokenizer_config.get("special_tokens", None),
                legacy=tokenizer_config.get("legacy", False),
            )
        elif target_class == "huggingface.auto_tokenizer.AutoTokenizer":
            tokenizer = AutoTokenizer.from_pretrained(
                str(model_context_dir / tokenizer_config["pretrained_model_name"])
            )
        else:
            raise ValueError(f"Unsupported tokenizer type: {tokenizer_module}{target_class}.")

    return tokenizer


def get_tokenizer(tokenizer_dir_or_path: Union[str, Path]) -> PreTrainedTokenizer:
    """Loads the tokenizer from the decoded NeMo weights dir."""
    tokenizer_dir_or_path = Path(tokenizer_dir_or_path)
    if (tokenizer_dir_or_path / "nemo_context").exists():
        return get_tokenizer_from_nemo2_context(tokenizer_dir_or_path / "nemo_context")
    elif (tokenizer_dir_or_path / "tokenizer_config.json").exists():
        return AutoTokenizer.from_pretrained(tokenizer_dir_or_path)
    elif os.path.exists(os.path.join(tokenizer_dir_or_path, "vocab.json")):
        vocab_path = tokenizer_dir_or_path / "vocab.json" if tokenizer_dir_or_path.is_dir() else tokenizer_dir_or_path
        tokenizer_config = {"library": "tiktoken", "vocab_file": str(vocab_path)}
        return build_tokenizer(tokenizer_config)
    else:
        model_path = (
            tokenizer_dir_or_path / "tokenizer.model" if tokenizer_dir_or_path.is_dir() else tokenizer_dir_or_path
        )
        tokenizer_config = {"library": "sentencepiece", "model": str(model_path)}
        return build_tokenizer(tokenizer_config)


def build_tokenizer(tokenizer):
    """Builds tokenizer for trt-llm export."""
    if isinstance(tokenizer, dict):
        tokenizer_config = tokenizer
        if tokenizer_config["library"] == "sentencepiece":
            return SentencePieceTokenizer(model_path=tokenizer_config["model"])
        elif tokenizer_config["library"] == "tiktoken":
            return TiktokenTokenizer(vocab_file=tokenizer_config["vocab_file"])
        elif "GPT2" in tokenizer_config["type"]:
            tokenizer = GPT2Tokenizer(tokenizer_config["vocab_file"], tokenizer_config["merge_file"])
        else:
            raise ValueError(f"Tokenizer type {tokenizer_config['library']} not handled")

        if tokenizer.bos_token_id is None:
            tokenizer.add_special_tokens({"bos_token": "<s>"})
        if tokenizer.eos_token_id is None:
            tokenizer.add_special_tokens({"eos_token": "</s>"})
    else:
        # For NeMo tokenizers, monkey patch encode & batch_decode methods for unified interface
        import nemo.collections.common.tokenizers as nemo_tokenizers

        if isinstance(tokenizer, nemo_tokenizers.TokenizerSpec):
            if isinstance(tokenizer, nemo_tokenizers.AutoTokenizer):
                # Unwrap the original methods of HF tokenizer
                batch_decode = tokenizer.tokenizer.batch_decode
                encode = tokenizer.tokenizer.encode
            elif isinstance(tokenizer, nemo_tokenizers.SentencePieceTokenizer):
                # Define HF equivalents based on available SP methods
                def batch_decode(self, ids):
                    if torch.is_tensor(ids):
                        ids = ids.cpu().numpy()
                    if isinstance(ids, np.ndarray):
                        ids = ids.tolist()
                    return self.tokenizer.decode(ids)

                encode = tokenizer.tokenizer.encode_as_ids
            else:
                raise NotImplementedError(f"Patching tokenizer methods for {type(tokenizer)} is not available")

            tokenizer.bos_token_id = tokenizer.bos_id
            tokenizer.eos_token_id = tokenizer.eos_id
            nemo_tokenizers.TokenizerSpec.encode = encode
            nemo_tokenizers.TokenizerSpec.batch_decode = batch_decode

    return tokenizer


def load_nemo_config(nemo_ckpt: Union[str, Path]) -> Dict[Any, Any]:
    """Load the model configuration from a NeMo checkpoint.

    This function handles both NeMo 1.0 and NeMo 2.0 checkpoint structures.
    For NeMo 2.0, it reads the configuration from the 'context/model.yaml' file.

    Args:
        nemo_ckpt (Union[str, Path]): Path to the NeMo checkpoint file or directory.

    Returns:
        Dict[Any, Any]: The configuration dictionary.
    """
    if Path(nemo_ckpt).is_dir():
        nemo_ckpt = Path(nemo_ckpt)
    else:
        nemo_ckpt = TarPath(nemo_ckpt)

    if (nemo_ckpt / "weights").exists() and (nemo_ckpt / "context").exists():  # Stucture of NeMo 2.0 checkpoints
        with (nemo_ckpt / "context" / "model.yaml").open("r") as stream:
            config = yaml.safe_load(stream)
    else:  # pragma: no cover
        raise Exception("Not supported NeMo checkpoint format.")

    return config


def get_model_type(nemo_ckpt: Union[str, Path], use_vllm_type: bool = False) -> Optional[str]:
    """Determine the model type from a NeMo checkpoint for TensorRT-LLM engine build or vLLM model converters.

    Args:
        nemo_ckpt (Union[str, Path]): Path to the NeMo checkpoint file.
        use_vllm_type (bool): If True, uses vLLM model type names for known model converters.

    Returns:
        Optional[str]: The model type if it can be determined, otherwise None.
    """
    model_config = load_nemo_config(nemo_ckpt)
    model_type = None

    if model_class := model_config.get("_target_"):
        # NeMo 2.0 case
        NEMO2_TO_MODEL_TYPE = {
            "nemo.collections.llm.gpt.model.base.GPTModel": "gpt",
            "nemo.collections.llm.gpt.model.llama.LlamaModel": "llama",
            "nemo.collections.llm.gpt.model.mistral.MistralModel": "llama",
            "nemo.collections.llm.gpt.model.mixtral.MixtralModel": "mixtral" if use_vllm_type else "llama",
            "nemo.collections.llm.gpt.model.starcoder.StarcoderModel": "gpt",
            "nemo.collections.llm.gpt.model.starcoder2.Starcoder2Model": "starcoder2" if use_vllm_type else "gpt",
            "nemo.collections.llm.gpt.model.nemotron.NemotronModel": "gpt",
            "nemo.collections.llm.gpt.model.gemma.GemmaModel": "gemma",
            "nemo.collections.llm.gpt.model.phi3mini.Phi3Model": "phi3",
            "nemo.collections.llm.gpt.model.baichuan.Baichuan2Model": "baichuan",
            "nemo.collections.llm.gpt.model.chatglm.ChatGLMModel": "chatglm",
            "nemo.collections.llm.gpt.model.qwen2.Qwen2Model": "qwen",
        }
        try:
            model_type = NEMO2_TO_MODEL_TYPE[model_class]
            LOGGER.info(f"Determined model_type='{model_type}' for {nemo_ckpt} checkpoint.")

        except KeyError:
            LOGGER.error(
                f"Model {model_class} not found in the NEMO2_TO_MODEL_TYPE mapping, "
                "try providing the model_type explicitely for exporting:\n"
                f"{json.dumps(NEMO2_TO_MODEL_TYPE, indent=2)}"
            )
            raise
    else:
        LOGGER.warning(f"Parameter model_type cannot be determined for {nemo_ckpt} checkpoint.")
    return model_type


def get_weights_dtype(nemo_ckpt: Union[str, Path]) -> Optional[str]:
    """Determine the weights data type from a NeMo checkpoint for TensorRT-LLM engine build.

    Args:
        nemo_ckpt (Union[str, Path]): Path to the NeMo checkpoint file.

    Returns:
        Optional[str]: The dtype if it can be determined, otherwise None.
    """
    model_config = load_nemo_config(nemo_ckpt)
    torch_dtype = None
    dtype = None

    is_nemo2 = "_target_" in model_config
    if is_nemo2:
        torch_dtype = model_config["config"]["params_dtype"]["_target_"]
    elif precision := model_config.get("precision", None):
        torch_dtype = str(torch_dtype_from_precision(precision))

    if torch_dtype is not None:
        dtype = torch_dtype.removeprefix("torch.")
        LOGGER.info(f"Determined weights dtype='{dtype}' for {nemo_ckpt} checkpoint.")
    else:
        LOGGER.warning(
            f"Parameter dtype for model weights cannot be determined for {nemo_ckpt} checkpoint. "
            "There is no 'precision' field specified in the model_config.yaml file."
        )

    return dtype


def load_distributed_model_weights(
    nemo_checkpoint: Union[str, Path],
    mcore_scales_format: Optional[bool] = None,
) -> Dict[str, Any]:
    """Loads model weights in `torch_dist` format from the model path.

    Args:
        nemo_checkpoint (str | Path): Path to the nemo checkpoint.
        mcore_scales_format (bool): Depreacted flag for local vs megatron.core export.

    Returns:
        dict: Model state dictionary.
    """
    if mcore_scales_format is not None:
        LOGGER.warning(
            "The mcore_scales_format parameter is deprecated and setting it does not take any effect. "
            "It will be removed in the future."
        )

    state_dict = load_model_weights(nemo_checkpoint, load_extra_states=True)

    state_dict = rename_extra_states(state_dict)

    return state_dict


def load_nemo_model(
    nemo_ckpt: Union[str, Path],
    nemo_export_dir: Union[str, Path],
):
    """Unified model loading for trt-llm export."""
    if not os.path.exists(nemo_ckpt):
        raise TypeError("%s does not exist", nemo_ckpt)

    nemo_dir = nemo_to_path(nemo_ckpt)

    tokenizer = None
    try:
        if (nemo_dir / "weights").exists():
            model = load_distributed_model_weights(nemo_ckpt)
            io_folder = nemo_dir / "context"

            if (io_folder / "model.yaml").exists():
                with open(io_folder / "model.yaml", "r") as stream:
                    config = yaml.safe_load(stream)

                nemo_model_config = {}
                for k, v in config["config"].items():
                    if isinstance(v, (float, int, str, bool)):
                        nemo_model_config[k] = v
                    elif k == "activation_func":
                        nemo_model_config["activation"] = v["_target_"].rsplit(".", 1)[-1]
            else:
                assert HAVE_NEMO2, "nemo_toolkit>=2.0.0 is required to load the model context."

                config = io.load_context(io_folder, subpath="model.config")

                nemo_model_config = {}
                for k, v in config.__dict__.items():
                    if isinstance(v, (float, int, str, bool)):
                        nemo_model_config[k] = v
                    elif k == "activation_func":
                        if isinstance(v, torch.jit.ScriptFunction):
                            nemo_model_config["activation"] = v.name
                        else:
                            nemo_model_config["activation"] = v.__name__

            if nemo_model_config.get("num_moe_experts") is None:
                nemo_model_config["num_moe_experts"] = 0
                nemo_model_config["moe_router_topk"] = 0
            if nemo_model_config["activation"] == "silu":
                nemo_model_config["activation"] = "fast-swiglu"
            elif nemo_model_config["activation"] == "openai_gelu":
                nemo_model_config["activation"] = "openai-gelu"
            elif nemo_model_config["activation"] == "squared_relu":
                nemo_model_config["activation"] = "squared-relu"

            if nemo_model_config.get("add_bias_linear"):
                nemo_model_config["bias"] = True

            nemo_model_config["mcore_gpt"] = True
            nemo_model_config["max_position_embeddings"] = nemo_model_config.get("seq_length", 4096)
            nemo_model_config["rotary_percentage"] = nemo_model_config.get("rotary_percent", 1.0)

            shutil.copytree(io_folder, nemo_export_dir / "nemo_context")
        else:
            raise Exception("Not a supported NeMo file format: only distributed MCore NeMo checkpoints are supported.")
    finally:
        if isinstance(nemo_dir, TarPath):
            nemo_dir.tarobject.close()

    return model, nemo_model_config, tokenizer
