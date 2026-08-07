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

# Resolve the checkout from this file so launchers also work when a linked
# worktree is bind-mounted without its external Git metadata.
PROJECT_ROOT=$(CDPATH= cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd -P)
export PROJECT_ROOT

# Some container runtimes start the requested numeric UID without adding it to
# /etc/passwd. Python's getpass then needs an explicit identity, including when
# PyTorch initializes its compilation cache during test collection.
export USER="${USER:-nemo-ci}"
export LOGNAME="${LOGNAME:-$USER}"

# The CI image may provide a root-owned cache. Redirect it only when it is not
# writable so existing writable cache configuration remains unchanged.
if [ -n "${XDG_CACHE_HOME:-}" ] && [ ! -w "$XDG_CACHE_HOME" ]; then
    XDG_CACHE_HOME="${HOME:?HOME must be set}/.cache"
elif [ -z "${XDG_CACHE_HOME:-}" ]; then
    XDG_CACHE_HOME="${HOME:?HOME must be set}/.cache"
fi
export XDG_CACHE_HOME

cd "$PROJECT_ROOT"
