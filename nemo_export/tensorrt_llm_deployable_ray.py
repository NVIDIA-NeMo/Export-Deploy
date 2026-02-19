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

"""TensorRT-LLM Ray deployment functionality has been removed.

This module now only contains placeholder functions that raise NotImplementedError.
TensorRT-LLM deployment support has been deprecated and removed from this codebase.
"""

import logging
from typing import List

LOGGER = logging.getLogger("NeMo")


class TensorRTLLMRayDeployable:
    """Placeholder class for TensorRT-LLM Ray deployment functionality.

    Note: TensorRT-LLM deployment support has been removed from this codebase.
    All methods will raise NotImplementedError.
    """

    def __init__(
        self,
        trt_llm_path: str,
        model_id: str = "tensorrt-llm-model",
        use_python_runtime: bool = True,
        enable_chunked_context: bool = None,
        max_tokens_in_paged_kv_cache: int = None,
        multi_block_mode: bool = False,
        lora_ckpt_list: List[str] = None,
    ):
        """Initialize the TensorRT-LLM model deployment.

        Raises:
            NotImplementedError: This functionality has been removed.
        """
        raise NotImplementedError(
            "TensorRT-LLM Ray deployment support has been removed from this codebase. "
            "Please use an earlier version if you need this functionality."
        )

    def generate(self, *args, **kwargs):
        """Generate method.

        Raises:
            NotImplementedError: This functionality has been removed.
        """
        raise NotImplementedError("TensorRT-LLM Ray deployment support has been removed from this codebase.")

    def chat_completions(self, *args, **kwargs):
        """Chat completions method.

        Raises:
            NotImplementedError: This functionality has been removed.
        """
        raise NotImplementedError("TensorRT-LLM Ray deployment support has been removed from this codebase.")

    def completions(self, *args, **kwargs):
        """Completions method.

        Raises:
            NotImplementedError: This functionality has been removed.
        """
        raise NotImplementedError("TensorRT-LLM Ray deployment support has been removed from this codebase.")

    @classmethod
    def options(cls, *args, **kwargs):
        """Options method for Ray deployment.

        Raises:
            NotImplementedError: This functionality has been removed.
        """
        raise NotImplementedError("TensorRT-LLM Ray deployment support has been removed from this codebase.")
