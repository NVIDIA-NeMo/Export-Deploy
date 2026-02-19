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

from typing import List

import numpy as np

from nemo_deploy import ITritonDeployable

try:
    from pytriton.model_config import Tensor
except Exception:
    from unittest.mock import MagicMock

    Tensor = MagicMock()


class TensorRTMMExporter(ITritonDeployable):
    """TensorRT multimodal exporter functionality has been removed.

    This class is kept for backward compatibility but all methods will raise NotImplementedError.
    """

    def __init__(
        self,
        model_dir: str,
        load_model: bool = True,
        modality: str = "vision",
    ):
        raise NotImplementedError("TensorRTMMExporter has been removed. This functionality is no longer supported.")

    def export(
        self,
        visual_checkpoint_path: str,
        llm_checkpoint_path: str = None,
        model_type: str = "neva",
        llm_model_type: str = "llama",
        processor_name: str = None,
        tensor_parallel_size: int = 1,
        max_input_len: int = 4096,
        max_output_len: int = 256,
        max_batch_size: int = 1,
        vision_max_batch_size: int = 1,
        max_multimodal_len: int = 3072,
        dtype: str = "bfloat16",
        delete_existing_files: bool = True,
        load_model: bool = True,
        use_lora_plugin: str = None,
        lora_target_modules: List[str] = None,
        lora_checkpoint_path: str = None,
        max_lora_rank: int = 64,
    ):
        """Export multimodal models to TRTLLM."""
        raise NotImplementedError(
            "TensorRTMMExporter.export has been removed. This functionality is no longer supported."
        )

    def forward(
        self,
        input_text: str,
        input_media: str,
        batch_size: int = 1,
        max_output_len: int = 30,
        top_k: int = 1,
        top_p: float = 0.0,
        temperature: float = 1.0,
        repetition_penalty: float = 1.0,
        num_beams: int = 1,
        lora_uids: List[str] = None,
    ):
        """Run forward with loaded TRTLLM engine."""
        raise NotImplementedError(
            "TensorRTMMExporter.forward has been removed. This functionality is no longer supported."
        )

    def get_input_media_tensors(self):
        """Get input media tensors."""
        raise NotImplementedError(
            "TensorRTMMExporter.get_input_media_tensors has been removed. This functionality is no longer supported."
        )

    @property
    def get_triton_input(self):
        raise NotImplementedError(
            "TensorRTMMExporter.get_triton_input has been removed. This functionality is no longer supported."
        )

    @property
    def get_triton_output(self):
        raise NotImplementedError(
            "TensorRTMMExporter.get_triton_output has been removed. This functionality is no longer supported."
        )

    def triton_infer_fn(self, **inputs: np.ndarray):
        """Triton inference function."""
        raise NotImplementedError(
            "TensorRTMMExporter.triton_infer_fn has been removed. This functionality is no longer supported."
        )

    def _load(self):
        raise NotImplementedError(
            "TensorRTMMExporter._load has been removed. This functionality is no longer supported."
        )
