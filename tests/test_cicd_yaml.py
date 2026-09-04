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

from pathlib import Path

import yaml

WORKFLOW_FILE = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "cicd-main.yml"


def test_inframework_e2e_job_has_matching_matrix():
    """Regression test for the inframework e2e job referencing matrix values without a strategy block."""
    workflow = yaml.safe_load(WORKFLOW_FILE.read_text())
    job = workflow["jobs"]["cicd-e2e-tests-inframework"]

    assert "strategy" in job, "inframework e2e job must define a strategy block"
    assert "matrix" in job["strategy"], "strategy block must define a matrix"

    include = job["strategy"]["matrix"].get("include", [])
    assert len(include) == 1, "expected exactly one matrix include for the inframework e2e job"
    matrix = include[0]
    assert matrix["script"] == "L2_Launch_InFramework"
    assert matrix["is_optional"] is False

