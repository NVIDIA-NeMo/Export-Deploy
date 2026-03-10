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

"""Tests for nemo_export/__init__.py.

The module is kept at 1.67% coverage because it is almost entirely a
conditional import block.  These tests exercise:
  1. The always-present __all__ entries (__version__ / __package_name__).
  2. The failure path – TensorRT-LLM not available (normal CPU-only env).
  3. The success path – TensorRT-LLM available (simulated via sys.modules).
"""

import importlib
import sys
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _reload_nemo_export():
    """Force a fresh import of nemo_export so that module-level code re-runs."""
    mod_name = "nemo_export"
    if mod_name in sys.modules:
        del sys.modules[mod_name]
    return importlib.import_module(mod_name)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestNemoExportInit:
    def test_version_in_all(self):
        import nemo_export

        assert "__version__" in nemo_export.__all__

    def test_package_name_in_all(self):
        import nemo_export

        assert "__package_name__" in nemo_export.__all__

    def test_version_is_string(self):
        import nemo_export

        assert isinstance(nemo_export.__version__, str)

    def test_package_name_is_string(self):
        import nemo_export

        assert isinstance(nemo_export.__package_name__, str)

    def test_tensorrt_llm_not_in_all_when_unavailable(self):
        """When TensorRT-LLM is not installed, TensorRTLLM should not appear in __all__."""
        # Ensure the submodules are absent from sys.modules before reload
        modules_to_remove = [k for k in sys.modules if k.startswith("nemo_export.tensorrt_llm")]
        saved = {k: sys.modules.pop(k) for k in modules_to_remove}

        try:
            with patch.dict("sys.modules", {"nemo_export.tensorrt_llm": None, "nemo_export.tensorrt_llm_hf": None}):
                mod = _reload_nemo_export()
                # Assert inside the context so the module object hasn't been reloaded yet
                assert "TensorRTLLM" not in mod.__all__
                assert "TensorRTLLMHF" not in mod.__all__
        finally:
            # Restore modules so we don't break subsequent tests
            sys.modules.update(saved)
            _reload_nemo_export()

    def test_tensorrt_llm_in_all_when_available(self):
        """When TensorRT-LLM is importable, TensorRTLLM and TensorRTLLMHF appear in __all__."""
        fake_trtllm = MagicMock()
        fake_trtllm_hf = MagicMock()
        fake_trtllm_cls = MagicMock()
        fake_trtllm_hf_cls = MagicMock()
        fake_trtllm.TensorRTLLM = fake_trtllm_cls
        fake_trtllm_hf.TensorRTLLMHF = fake_trtllm_hf_cls

        extra_modules = {
            "nemo_export.tensorrt_llm": fake_trtllm,
            "nemo_export.tensorrt_llm_hf": fake_trtllm_hf,
        }

        try:
            with patch.dict("sys.modules", extra_modules):
                mod = _reload_nemo_export()
                # Assertions must be inside the patch.dict context because the
                # finally-block reload will mutate the same module object in place.
                assert "TensorRTLLM" in mod.__all__
                assert "TensorRTLLMHF" in mod.__all__
                assert mod.TensorRTLLM is fake_trtllm_cls
                assert mod.TensorRTLLMHF is fake_trtllm_hf_cls
        finally:
            _reload_nemo_export()

    def test_module_not_found_is_silently_handled(self):
        """ModuleNotFoundError during optional import must not propagate."""
        modules_to_remove = [k for k in sys.modules if k.startswith("nemo_export.tensorrt_llm")]
        saved = {k: sys.modules.pop(k) for k in modules_to_remove}

        try:
            # Setting to None in sys.modules causes ImportError on import
            with patch.dict("sys.modules", {"nemo_export.tensorrt_llm": None}):
                mod = _reload_nemo_export()
                # Should be importable without raising
                assert mod is not None
        finally:
            sys.modules.update(saved)
            _reload_nemo_export()

    def test_all_is_a_list(self):
        import nemo_export

        assert isinstance(nemo_export.__all__, list)

    def test_all_contains_at_least_two_entries(self):
        """__all__ must always have at least __version__ and __package_name__."""
        import nemo_export

        assert len(nemo_export.__all__) >= 2
