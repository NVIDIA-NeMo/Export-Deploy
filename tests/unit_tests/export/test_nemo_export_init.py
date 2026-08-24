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

"""Tests for nemo_export/__init__.py."""

import nemo_export


class TestNemoExportInit:
    def test_version_in_all(self):
        assert "__version__" in nemo_export.__all__

    def test_package_name_in_all(self):
        assert "__package_name__" in nemo_export.__all__

    def test_version_is_string(self):
        assert isinstance(nemo_export.__version__, str)

    def test_package_name_is_string(self):
        assert isinstance(nemo_export.__package_name__, str)

    def test_all_is_a_list(self):
        assert isinstance(nemo_export.__all__, list)

    def test_all_contains_at_least_two_entries(self):
        """__all__ must always have at least __version__ and __package_name__."""
        import nemo_export

        assert len(nemo_export.__all__) >= 2
