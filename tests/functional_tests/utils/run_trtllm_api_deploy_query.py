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

import argparse

from nemo_deploy import DeployPyTriton
from nemo_deploy.nlp import NemoQueryTRTLLMAPI
from nemo_deploy.nlp.trtllm_api_deployable import TensorRTLLMAPIDeployable


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Deploy nemo models to Triton and benchmark the models",
    )
    parser.add_argument(
        "--hf_model_path",
        type=str,
        required=True,
        help="Path to the HuggingFace model directory",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default=None,
        help="Path to the tokenizer or tokenizer instance",
    )
    parser.add_argument(
        "--tensor_parallel_size",
        type=int,
        default=1,
        help="Tensor parallelism size",
    )
    parser.add_argument(
        "--pipeline_parallel_size",
        type=int,
        default=1,
        help="Pipeline parallelism size",
    )
    parser.add_argument(
        "--moe_expert_parallel_size",
        type=int,
        default=-1,
        help="MOE expert parallelism size",
    )
    parser.add_argument(
        "--moe_tensor_parallel_size",
        type=int,
        default=-1,
        help="MOE tensor parallelism size",
    )
    parser.add_argument(
        "--max_batch_size",
        type=int,
        default=8,
        help="Maximum batch size",
    )
    parser.add_argument(
        "--max_num_tokens",
        type=int,
        default=8192,
        help="Maximum total tokens across all sequences in a batch",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="pytorch",
        help="Backend to use for TRTLLM",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="auto",
        help="Model data type",
    )

    return parser.parse_args()


def test_trtllm_api_deploy_query(args):
    """Test the TensorRT-LLM API query interface."""
    model_name = "test_model"

    model = TensorRTLLMAPIDeployable(
        hf_model_id_path=args.hf_model_path,
        tokenizer=args.tokenizer,
        tensor_parallel_size=args.tensor_parallel_size,
        pipeline_parallel_size=args.pipeline_parallel_size,
        moe_expert_parallel_size=args.moe_expert_parallel_size,
        moe_tensor_parallel_size=args.moe_tensor_parallel_size,
        max_batch_size=args.max_batch_size,
        max_num_tokens=args.max_num_tokens,
        backend=args.backend,
        dtype=args.dtype,
    )

    try:
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
    finally:
        nm.stop()

    assert output_deployed is not None, "Output cannot be none."
    assert len(output_deployed) > 0, "Output should not be empty."
    assert isinstance(output_deployed[0][0], str), "Output should be a string."
    assert len(output_deployed[0][0]) > 0, "Output string should not be empty."


if __name__ == "__main__":
    args = get_args()
    test_trtllm_api_deploy_query(args)
