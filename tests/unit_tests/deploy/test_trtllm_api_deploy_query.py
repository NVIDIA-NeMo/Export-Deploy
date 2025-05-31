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

import pytest

from nemo_deploy import DeployPyTriton
from nemo_deploy.nlp import NemoQueryTRTLLMAPI
from nemo_deploy.nlp.trtllm_api_deployable import TensorRTLLMAPIDeployable


@pytest.mark.run_only_on('GPU')
@pytest.mark.unit
def test_trtllm_api_deploy_query():
    """Test the TensorRT-LLM API query interface."""
    model_name = "test_model"
    hf_model_path = "/home/TestData/nlp/megatron_llama/llama-ci-hf"

    model = TensorRTLLMAPIDeployable(
        hf_model_id_path=hf_model_path,
        tensor_parallel_size=2,
    )

    nm = DeployPyTriton(
        model=model,
        triton_model_name=model_name,
        max_batch_size=8,
        http_port=8000,
        address="0.0.0.0",
    )
    nm.deploy()
    nm.run()

    nq = NemoQueryTRTLLMAPI(url="localhost:8000", model_name=model_name)
    output_deployed = nq.query_llm(
        prompts=["What is the meaning of life?"],
        max_length=20,
    )
    nm.stop()

    assert output_deployed is not None, "Output cannot be none."
    assert len(output_deployed) > 0, "Output should not be empty."
    assert isinstance(output_deployed[0][0], str), "Output should be a string."
    assert len(output_deployed[0][0]) > 0, "Output string should not be empty."
