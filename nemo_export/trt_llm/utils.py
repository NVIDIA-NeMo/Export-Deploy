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

from typing import Any, Dict, Optional, Tuple

import tensorrt_llm


def is_rank(rank: Optional[int]) -> bool:
    """Check if the current MPI rank matches the specified rank.

    Args:
        rank (Optional[int]): The rank to check against.

    Returns:
        bool: True if the current rank matches the specified rank or if rank is None.
    """
    current_rank = tensorrt_llm.mpi_rank()
    if rank is None:
        return True
    if isinstance(rank, int):
        return current_rank == rank
    raise ValueError(f"Invalid rank argument {rank} of type {type(rank)}.")


def determine_quantization_settings(
    nemo_model_config: Dict[str, Any],
    fp8_quantized: Optional[bool] = None,
    fp8_kvcache: Optional[bool] = None,
) -> Tuple[bool, bool]:
    """Determines the exported models quantization settings.

    Reads from NeMo config, with optional override.

    Args:
        nemo_model_config (dict): NeMo model configuration
        fp8_quantized (optional, bool): User-specified quantization flag
        fp8_kvcache (optional, bool): User-specified cache quantization flag
    Returns:
        Tuple[bool, bool]:
            - Model quantization flag
            - Model kv-cache quantization flag
    """
    is_nemo_quantized: bool = nemo_model_config.get("fp8", False)
    if fp8_quantized is None:
        fp8_quantized = is_nemo_quantized
    if fp8_kvcache is None:
        fp8_kvcache = is_nemo_quantized

    return fp8_quantized, fp8_kvcache
