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


import os
import shutil
import tempfile
from unittest.mock import MagicMock

import pytest
import torch


class TestUtils:
    @pytest.fixture
    def temp_dir(self):
        # Create a temporary directory
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        # Cleanup after test
        shutil.rmtree(temp_dir)

    @pytest.mark.run_only_on("GPU")
    def test_prepare_directory_for_export(self, temp_dir):
        from nemo_export.utils.utils import prepare_directory_for_export

        # Test creating new directory
        model_dir = os.path.join(temp_dir, "new_dir")
        prepare_directory_for_export(model_dir, delete_existing_files=False)
        assert os.path.exists(model_dir)
        assert os.path.isdir(model_dir)

        # Test with existing empty directory
        prepare_directory_for_export(model_dir, delete_existing_files=False)
        assert os.path.exists(model_dir)

        # Test with existing non-empty directory
        with open(os.path.join(model_dir, "test.txt"), "w") as f:
            f.write("test")

        with pytest.raises(RuntimeError):
            prepare_directory_for_export(model_dir, delete_existing_files=False)

        # Test with delete_existing_files=True
        prepare_directory_for_export(model_dir, delete_existing_files=True)
        assert os.path.exists(model_dir)
        assert not os.path.exists(os.path.join(model_dir, "test.txt"))

        # Test with subdir
        prepare_directory_for_export(model_dir, delete_existing_files=False, subdir="subdir")
        assert os.path.exists(os.path.join(model_dir, "subdir"))

    @pytest.mark.run_only_on("GPU")
    def test_torch_dtype_from_precision(self):
        from nemo_export.utils.utils import torch_dtype_from_precision

        # Test with megatron_amp_O2=False
        assert torch_dtype_from_precision("bf16", megatron_amp_O2=False) == torch.float32

        # Test with different precision values
        assert torch_dtype_from_precision("bf16") == torch.bfloat16
        assert torch_dtype_from_precision("bf16-mixed") == torch.bfloat16
        assert torch_dtype_from_precision(16) == torch.float16
        assert torch_dtype_from_precision("16") == torch.float16
        assert torch_dtype_from_precision("16-mixed") == torch.float16
        assert torch_dtype_from_precision(32) == torch.float32
        assert torch_dtype_from_precision("32") == torch.float32
        assert torch_dtype_from_precision("32-true") == torch.float32

        # Test with invalid precision
        with pytest.raises(ValueError):
            torch_dtype_from_precision("invalid")

    @pytest.mark.run_only_on("GPU")
    def test_get_example_inputs(self):
        from nemo_export.utils.utils import get_example_inputs

        # Mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3], [4, 5, 6]]),
            "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 1]]),
        }

        result = get_example_inputs(mock_tokenizer)

        # Verify tokenizer was called with correct arguments
        mock_tokenizer.assert_called_once_with(
            ["example query one", "example query two"],
            ["example passage one", "example passage two"],
            return_tensors="pt",
        )

        # Verify result structure
        assert isinstance(result, dict)
        assert "input_ids" in result
        assert "attention_mask" in result
        assert isinstance(result["input_ids"], torch.Tensor)
        assert isinstance(result["attention_mask"], torch.Tensor)
