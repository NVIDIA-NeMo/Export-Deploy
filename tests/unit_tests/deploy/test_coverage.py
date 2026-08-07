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

import os
import subprocess
import tomllib
from pathlib import Path

import pytest
from coverage.files import GlobMatcher

from tests.coverage import COVERAGE_DATA_FILE, PROJECT_ROOT, coverage_args


def test_coverage_args_follow_checkout():
    args = coverage_args()

    assert PROJECT_ROOT == COVERAGE_DATA_FILE.parent
    assert f"--data-file={PROJECT_ROOT / '.coverage'}" in args
    assert f"--source={PROJECT_ROOT}" in args
    assert "--parallel-mode" in args


@pytest.mark.parametrize("checkout", [Path("/workdir"), Path("/tmp/export-deploy")])
def test_coverage_omit_rules_follow_checkout(checkout):
    config = tomllib.loads((PROJECT_ROOT / "pyproject.toml").read_text())
    matcher = GlobMatcher(config["tool"]["coverage"]["run"]["omit"])

    assert matcher.match(str(checkout / "tests" / "test_example.py"))
    assert matcher.match(str(checkout / "nemo_export_deploy_common" / "package_info.py"))
    assert not matcher.match(str(checkout / "nemo_export" / "example.py"))


@pytest.mark.parametrize("checkout", [Path("/workdir"), Path("/tmp/export-deploy")])
def test_shell_coverage_setup_does_not_require_git(checkout, tmp_path):
    mounted_checkout = tmp_path / checkout.relative_to("/")
    helper = mounted_checkout / "tests" / "coverage.sh"
    helper.parent.mkdir(parents=True)
    helper.write_bytes((PROJECT_ROOT / "tests" / "coverage.sh").read_bytes())

    result = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; printf "%s\\n%s\\n" "$PROJECT_ROOT" "$PWD"',
            "coverage-test",
            str(helper),
        ],
        check=True,
        capture_output=True,
        text=True,
        env={"HOME": str(tmp_path / "home"), "PATH": "/usr/bin:/bin"},
    )

    expected_root = str(mounted_checkout.resolve())
    assert result.stdout.splitlines() == [expected_root, expected_root]


def test_shell_coverage_setup_provides_container_identity(tmp_path):
    env = os.environ.copy()
    env.pop("USER", None)
    env.pop("LOGNAME", None)

    result = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; printf "%s\\n%s\\n" "$USER" "$LOGNAME"',
            "coverage-test",
            str(PROJECT_ROOT / "tests" / "coverage.sh"),
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env=env,
    )

    assert result.stdout.splitlines() == ["nemo-ci", "nemo-ci"]


@pytest.mark.parametrize("configured_cache", [None, Path("/root-owned-cache")])
def test_shell_coverage_setup_provides_writable_cache(configured_cache, tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    env = {"HOME": str(home), "PATH": "/usr/bin:/bin"}
    if configured_cache is not None:
        env["XDG_CACHE_HOME"] = str(configured_cache)

    result = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; printf "%s\\n" "$XDG_CACHE_HOME"',
            "coverage-test",
            str(PROJECT_ROOT / "tests" / "coverage.sh"),
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env=env,
    )

    assert result.stdout.strip() == str(home / ".cache")


def test_shell_coverage_setup_preserves_writable_cache(tmp_path):
    cache = tmp_path / "cache"
    cache.mkdir()
    result = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; printf "%s\\n" "$XDG_CACHE_HOME"',
            "coverage-test",
            str(PROJECT_ROOT / "tests" / "coverage.sh"),
        ],
        check=True,
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env={"HOME": str(tmp_path / "home"), "PATH": "/usr/bin:/bin", "XDG_CACHE_HOME": str(cache)},
    )
    assert result.stdout.strip() == str(cache)
