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

import gc
import time

import pytest


@pytest.fixture(autouse=True)
def stabilize_gpu_memory_between_tests():
    """
    Stabilize GPU memory before and after each test to prevent vLLM memory profiling race conditions.

    vLLM V1 takes a snapshot of GPU memory at initialization, then profiles available memory.
    If memory changes between these two steps (e.g., another process releases memory),
    vLLM raises an AssertionError. This fixture ensures memory is stable before each test.
    """

    def _cleanup_and_wait():
        # Force garbage collection
        gc.collect()

        # Try to clean up CUDA memory if torch is available
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
        except ImportError:
            pass

        # Try to clean up Ray resources if Ray is available
        try:
            import ray

            if ray.is_initialized():
                # Give Ray time to clean up any pending tasks
                pass
        except ImportError:
            pass

        # Wait for memory to stabilize - this is critical for K8s time-slicing environments
        time.sleep(5)

    # Before test: ensure clean state
    _cleanup_and_wait()

    yield

    # After test: cleanup
    _cleanup_and_wait()
