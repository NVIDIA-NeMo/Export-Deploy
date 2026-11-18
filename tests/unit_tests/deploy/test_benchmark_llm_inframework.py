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

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest


@pytest.fixture
def benchmark_module(monkeypatch):
    # The benchmark only needs the query class; no Triton deployment is needed.
    fake_llm = ModuleType("nemo_deploy.llm")
    fake_llm.NemoQueryLLMPyTorch = Mock()
    monkeypatch.setitem(sys.modules, "nemo_deploy.llm", fake_llm)
    path = Path(__file__).resolve().parents[3] / "scripts/deploy/nlp/benchmark_llm_inframework.py"
    spec = importlib.util.spec_from_file_location("benchmark_llm_inframework_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize(
    "num_queries,batch_size,ticks,expected_batches,mean,p95,qps",
    [
        (6, 4, [0, 4, 4, 6], [1, 4, 2], 10 / 3, 4.0, 1.0),
        (3, 1, [0, 1, 1, 3, 3, 6], [1, 1, 1, 1], 2.0, 2.9, 0.5),
    ],
)
def test_batch_response_latency_and_throughput(
    benchmark_module, monkeypatch, capsys, num_queries, batch_size, ticks, expected_batches, mean, p95, qps
):
    seen_batches = []

    class FakeQuery:
        def query_llm(self, prompts, **kwargs):
            seen_batches.append(len(prompts))
            return ["ok"] * len(prompts)

    clock = iter(ticks)
    monkeypatch.setattr(benchmark_module, "time", SimpleNamespace(time=clock.__next__, perf_counter=clock.__next__))
    monkeypatch.setattr(benchmark_module, "NemoQueryLLMPyTorch", lambda *args: FakeQuery())

    stats = benchmark_module.run_benchmark("unused", "unused", num_queries, batch_size, warmup=1)

    assert seen_batches == expected_batches
    assert stats["mean_latency"] == pytest.approx(mean)
    assert stats["p95_latency"] == pytest.approx(p95)
    assert stats["queries_per_second"] == pytest.approx(qps)
    assert stats["total_queries"] == num_queries
    if batch_size == 4:
        assert "Query 1/6 completed in 4.00 seconds" in capsys.readouterr().out
