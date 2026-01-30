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

"""TensorRT-LLM HuggingFace export functionality has been removed.

This module now only contains placeholder functions that raise NotImplementedError.
TensorRT-LLM export support has been deprecated and removed from this codebase.
"""

import logging
from typing import List, Optional

from nemo_export.tensorrt_llm import TensorRTLLM

LOGGER = logging.getLogger("NeMo")


class TensorRTLLMHF(TensorRTLLM):
    """Placeholder class for TensorRT-LLM HuggingFace export functionality.

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
        """Initialize TensorRTLLMHF exporter.

        Raises:
            NotImplementedError: This functionality has been removed.
        """
        raise NotImplementedError(
            "TensorRT-LLM HuggingFace export support has been removed from this codebase. "
            "Please use an earlier version if you need this functionality."
        )

    def export_hf_model(
        self,
        hf_model_path: str,
        max_batch_size: int = 8,
        tensor_parallelism_size: int = 1,
        max_input_len: int = 256,
        max_output_len: Optional[int] = None,
        max_num_tokens: Optional[int] = None,
        opt_num_tokens: Optional[int] = None,
        dtype: Optional[str] = None,
        max_seq_len: Optional[int] = 512,
        gemm_plugin: str = "auto",
        remove_input_padding: bool = True,
        use_paged_context_fmha: bool = True,
        paged_kv_cache: bool = True,
        multiple_profiles: bool = False,
        reduce_fusion: bool = False,
        model_type: Optional[str] = None,
        delete_existing_files: bool = True,
    ):
        """Export HuggingFace model to TensorRT-LLM.

        Raises:
            NotImplementedError: This functionality has been removed.
        """
        raise NotImplementedError("TensorRT-LLM HuggingFace export support has been removed from this codebase.")

    def get_hf_model_type(self, hf_model_path: str) -> str:
        """Get HuggingFace model type.

        Raises:
            NotImplementedError: This functionality has been removed.
        """
        raise NotImplementedError("TensorRT-LLM HuggingFace export support has been removed from this codebase.")

    def get_hf_model_dtype(self, hf_model_path: str) -> str:
        """Get HuggingFace model dtype.

        Raises:
            NotImplementedError: This functionality has been removed.
        """
        raise NotImplementedError("TensorRT-LLM HuggingFace export support has been removed from this codebase.")

    @property
    def get_supported_hf_model_mapping(self):
        """Get supported HuggingFace model mapping.

        Raises:
            NotImplementedError: This functionality has been removed.
        """
        raise NotImplementedError("TensorRT-LLM HuggingFace export support has been removed from this codebase.")
