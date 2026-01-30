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

"""TensorRT-LLM export functionality has been removed.

This module now only contains placeholder functions that raise NotImplementedError.
TensorRT-LLM export support has been deprecated and removed from this codebase.
"""

import logging
from typing import Any, Dict, List, Optional

LOGGER = logging.getLogger("NeMo")


class TensorRTLLM:
    """Placeholder class for TensorRT-LLM export functionality.

    Note: TensorRT-LLM export support has been removed from this codebase.
    All methods will raise NotImplementedError.
    """

    def __init__(
        self,
        model_dir: str,
        lora_ckpt_list: List[str] = None,
        load_model: bool = True,
        use_python_runtime: bool = True,
        enable_chunked_context: bool = None,
        max_tokens_in_paged_kv_cache: int = None,
        multi_block_mode: bool = False,
    ):
        """Initialize TensorRTLLM exporter.

        Raises:
            NotImplementedError: This functionality has been removed.
        """
        raise NotImplementedError(
            "TensorRT-LLM export support has been removed from this codebase. "
            "Please use an earlier version if you need this functionality."
        )

    def _export_nemo_checkpoint(
        self,
        nemo_checkpoint_path: str,
        model_type: Optional[str] = None,
        delete_existing_files: bool = True,
        tensor_parallelism_size: int = 1,
        pipeline_parallelism_size: int = 1,
        max_input_len: int = 256,
        max_output_len: Optional[int] = None,
        max_batch_size: int = 8,
        use_parallel_embedding: bool = False,
        paged_kv_cache: bool = True,
        remove_input_padding: bool = True,
        use_paged_context_fmha: bool = True,
        dtype: Optional[str] = None,
        load_model: bool = True,
        use_lora_plugin: str = None,
        lora_target_modules: List[str] = None,
        max_lora_rank: int = 64,
        max_num_tokens: Optional[int] = None,
        opt_num_tokens: Optional[int] = None,
        max_seq_len: Optional[int] = 512,
        multiple_profiles: bool = False,
        gpt_attention_plugin: str = "auto",
        gemm_plugin: str = "auto",
        reduce_fusion: bool = True,
        fp8_quantized: Optional[bool] = None,
        fp8_kvcache: Optional[bool] = None,
        build_rank: Optional[int] = 0,
    ):
        """Export nemo checkpoints to TensorRT-LLM format.

        Raises:
            NotImplementedError: This functionality has been removed.
        """
        raise NotImplementedError("TensorRT-LLM export support has been removed from this codebase.")

    def export_with_hf(
        self,
        nemo_checkpoint_path: str,
        model_type: Optional[str] = None,
        delete_existing_files: bool = True,
        tensor_parallelism_size: int = 1,
        max_input_len: int = 256,
        max_output_len: Optional[int] = None,
        max_batch_size: int = 8,
        paged_kv_cache: bool = True,
        remove_input_padding: bool = True,
        use_paged_context_fmha: bool = True,
        dtype: Optional[str] = None,
        max_num_tokens: Optional[int] = None,
        opt_num_tokens: Optional[int] = None,
        max_seq_len: Optional[int] = 512,
        multiple_profiles: bool = False,
        gemm_plugin: str = "auto",
        reduce_fusion: bool = False,
    ):
        """Export via HuggingFace conversion fallback.

        Raises:
            NotImplementedError: This functionality has been removed.
        """
        raise NotImplementedError("TensorRT-LLM export support has been removed from this codebase.")

    def export(
        self,
        nemo_checkpoint_path: str,
        model_type: Optional[str] = None,
        delete_existing_files: bool = True,
        tensor_parallelism_size: int = 1,
        pipeline_parallelism_size: int = 1,
        max_input_len: int = 256,
        max_output_len: Optional[int] = None,
        max_batch_size: int = 8,
        use_parallel_embedding: bool = False,
        paged_kv_cache: bool = True,
        remove_input_padding: bool = True,
        use_paged_context_fmha: bool = True,
        dtype: Optional[str] = None,
        load_model: bool = True,
        use_lora_plugin: str = None,
        lora_target_modules: List[str] = None,
        max_lora_rank: int = 64,
        max_num_tokens: Optional[int] = None,
        opt_num_tokens: Optional[int] = None,
        max_seq_len: Optional[int] = 512,
        multiple_profiles: bool = False,
        gpt_attention_plugin: str = "auto",
        gemm_plugin: str = "auto",
        reduce_fusion: bool = True,
        fp8_quantized: Optional[bool] = None,
        fp8_kvcache: Optional[bool] = None,
        build_rank: Optional[int] = 0,
    ):
        """Export nemo checkpoints to TensorRT-LLM.

        Raises:
            NotImplementedError: This functionality has been removed.
        """
        raise NotImplementedError("TensorRT-LLM export support has been removed from this codebase.")

    def _export_to_nim_format(self, model_config: Dict[str, Any], model_type: str):
        """Export model configuration to NIM format.

        Raises:
            NotImplementedError: This functionality has been removed.
        """
        raise NotImplementedError("TensorRT-LLM export support has been removed from this codebase.")

    def get_transformer_config(self, nemo_model_config):
        """Get transformer config from nemo model config.

        Raises:
            NotImplementedError: This functionality has been removed.
        """
        raise NotImplementedError("TensorRT-LLM export support has been removed from this codebase.")

    def forward(
        self,
        input_texts: List[str],
        max_output_len: int = 64,
        top_k: int = 1,
        top_p: float = 0.0,
        temperature: float = 1.0,
        stop_words_list: List[str] = None,
        bad_words_list: List[str] = None,
        no_repeat_ngram_size: int = None,
        lora_uids: List[str] = None,
        output_log_probs: bool = False,
        output_context_logits: bool = False,
        output_generation_logits: bool = False,
        **sampling_kwargs,
    ):
        """Run inference.

        Raises:
            NotImplementedError: This functionality has been removed.
        """
        raise NotImplementedError("TensorRT-LLM export support has been removed from this codebase.")

    @property
    def get_hidden_size(self):
        """Get hidden size.

        Raises:
            NotImplementedError: This functionality has been removed.
        """
        raise NotImplementedError("TensorRT-LLM export support has been removed from this codebase.")

    @property
    def get_triton_input(self):
        """Get triton input configuration.

        Raises:
            NotImplementedError: This functionality has been removed.
        """
        raise NotImplementedError("TensorRT-LLM export support has been removed from this codebase.")

    @property
    def get_triton_output(self):
        """Get triton output configuration.

        Raises:
            NotImplementedError: This functionality has been removed.
        """
        raise NotImplementedError("TensorRT-LLM export support has been removed from this codebase.")

    def _infer_fn(self, prompts, inputs):
        """Shared inference helper function.

        Raises:
            NotImplementedError: This functionality has been removed.
        """
        raise NotImplementedError("TensorRT-LLM export support has been removed from this codebase.")

    def triton_infer_fn(self, **inputs):
        """Triton inference function.

        Raises:
            NotImplementedError: This functionality has been removed.
        """
        raise NotImplementedError("TensorRT-LLM export support has been removed from this codebase.")

    def ray_infer_fn(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Ray inference function.

        Raises:
            NotImplementedError: This functionality has been removed.
        """
        raise NotImplementedError("TensorRT-LLM export support has been removed from this codebase.")

    def _load_config_file(self):
        """Load config file.

        Raises:
            NotImplementedError: This functionality has been removed.
        """
        raise NotImplementedError("TensorRT-LLM export support has been removed from this codebase.")

    def _load(self):
        """Load model.

        Raises:
            NotImplementedError: This functionality has been removed.
        """
        raise NotImplementedError("TensorRT-LLM export support has been removed from this codebase.")

    def unload_engine(self):
        """Unload engine.

        Raises:
            NotImplementedError: This functionality has been removed.
        """
        raise NotImplementedError("TensorRT-LLM export support has been removed from this codebase.")
