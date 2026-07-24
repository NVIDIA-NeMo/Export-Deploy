# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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

from unittest.mock import MagicMock, patch

import numpy as np
import pytest


@pytest.fixture
def query_llm_module():
    """Import query.py with pytriton clients mocked."""
    with patch.dict(
        "sys.modules",
        {
            "pytriton": MagicMock(),
            "pytriton.client": MagicMock(),
        },
    ):
        from scripts.deploy.nlp import query as query_module

        yield query_module


class TestQueryLLMInputs:
    @patch("scripts.deploy.nlp.query.ModelClient")
    def test_random_seed_uses_int_dtype(self, mock_client, query_llm_module):
        """random_seed must be passed as an integer array, not float."""
        mock_instance = MagicMock()
        mock_client.return_value.__enter__.return_value = mock_instance
        mock_instance.infer_batch.return_value = {"outputs": np.array([b"response"])}
        mock_instance.model_config.outputs = [MagicMock(dtype=np.bytes_)]

        query_llm_module.query_llm(
            url="localhost:8000",
            model_name="test",
            prompts=["hello"],
            random_seed=42,
        )

        call_kwargs = mock_instance.infer_batch.call_args.kwargs
        assert call_kwargs["random_seed"].dtype == np.int_
        assert call_kwargs["random_seed"][0] == 42
