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

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
COVERAGE_DATA_FILE = PROJECT_ROOT / ".coverage"


def coverage_args() -> list[str]:
    """Return coverage arguments rooted at the active checkout."""
    return [
        "coverage",
        "run",
        f"--data-file={COVERAGE_DATA_FILE}",
        f"--source={PROJECT_ROOT}",
        "--parallel-mode",
    ]
