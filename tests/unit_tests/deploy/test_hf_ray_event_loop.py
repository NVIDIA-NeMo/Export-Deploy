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
import threading
from types import SimpleNamespace

import pytest


@pytest.fixture
def deployment_class():
    from nemo_deploy.llm.hf_deployable_ray import HFRayDeployable

    return HFRayDeployable.func_or_class


@pytest.mark.parametrize(
    ("endpoint", "payload"),
    [("completions", {"prompt": "hello"}), ("chat_completions", {"messages": [{"role": "user", "content": "hello"}]})],
)
def test_inference_does_not_block_event_loop(deployment_class, endpoint, payload):
    started = threading.Event()
    release = threading.Event()

    def infer(inputs):
        started.set()
        release.wait(timeout=2)
        return {"sentences": ["ok"]}

    deployment = deployment_class.__new__(deployment_class)
    deployment.model_id = "test"
    deployment.model = SimpleNamespace(ray_infer_fn=infer)
    deployment._inference_lock = threading.Lock()

    async def run():
        task = asyncio.create_task(getattr(deployment, endpoint)(payload))
        try:
            await asyncio.sleep(0.05)
            assert started.is_set()
            assert not task.done()
        finally:
            release.set()
        result = await task
        assert result["choices"]

    asyncio.run(run())


def test_cancelled_request_keeps_model_calls_serialized(deployment_class):
    started = threading.Event()
    release = threading.Event()
    second_started = threading.Event()
    calls = []

    def infer(inputs):
        calls.append(inputs)
        if len(calls) == 1:
            started.set()
            release.wait(timeout=2)
        else:
            second_started.set()
        return {"sentences": ["ok"]}

    deployment = deployment_class.__new__(deployment_class)
    deployment.model_id = "test"
    deployment.model = SimpleNamespace(ray_infer_fn=infer)
    deployment._inference_lock = threading.Lock()

    async def run():
        first = asyncio.create_task(deployment.completions({"prompt": "one"}))
        second = None
        try:
            await asyncio.sleep(0.05)
            assert started.is_set()
            first.cancel()
            with pytest.raises(asyncio.CancelledError):
                await first
            second = asyncio.create_task(deployment.completions({"prompt": "two"}))
            await asyncio.sleep(0.05)
            assert not second_started.is_set()
        finally:
            release.set()
        await second
        assert second_started.is_set()

    asyncio.run(run())
