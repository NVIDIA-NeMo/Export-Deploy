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

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest


@pytest.fixture
def run_batch():
    from nemo_deploy.llm.megatronllm_deployable_ray import MegatronRayDeployable

    async def infer(inputs):
        prompts = inputs["prompts"]
        return {
            "sentences": [f"answer:{prompt}" for prompt in prompts],
            "log_probs": [f"log:{prompt}" for prompt in prompts],
            "top_logprobs": [f"top:{prompt}" for prompt in prompts],
        }

    remote = AsyncMock(side_effect=infer)
    deployment = SimpleNamespace(primary_worker=SimpleNamespace(infer=SimpleNamespace(remote=remote)))
    handler = MegatronRayDeployable.func_or_class._batched_infer_impl

    def run(requests):
        return asyncio.run(handler(deployment, requests)), remote

    return run


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("max_length", 17),
        ("temperature", 0.2),
        ("top_k", 4),
        ("top_p", 0.8),
        ("compute_logprob", True),
        ("apply_chat_template", True),
        ("n_top_logprobs", 3),
        ("echo", True),
        ("stop_words", ["END"]),
    ],
)
def test_different_generation_settings_are_not_shared(run_batch, name, value):
    requests = [{"prompts": ["a"]}, {"prompts": ["b"], name: value}]
    results, remote = run_batch(requests)
    assert remote.await_count == 2
    assert remote.await_args_list[1].args[0][name] == value
    assert [r["sentences"] for r in results] == [["answer:a"], ["answer:b"]]


def test_matching_requests_share_a_call_and_restore_original_order(run_batch):
    requests = [
        {"prompts": ["a", "b"], "stop_words": ["END"]},
        {"prompts": ["c"], "stop_words": ["OTHER"]},
        {"prompts": ["d"], "stop_words": ["END"]},
    ]
    results, remote = run_batch(requests)
    assert remote.await_count == 2
    assert remote.await_args_list[0].args[0]["prompts"] == ["a", "b", "d"]
    assert [r["sentences"] for r in results] == [["answer:a", "answer:b"], ["answer:c"], ["answer:d"]]
    assert [r["log_probs"] for r in results] == [["log:a", "log:b"], ["log:c"], ["log:d"]]
    assert [r["top_logprobs"] for r in results] == [["top:a", "top:b"], ["top:c"], ["top:d"]]
