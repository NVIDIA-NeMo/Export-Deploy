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

#!/bin/bash
set -xeuo pipefail # Exit immediately if a command exits with a non-zero status

source "$(dirname -- "${BASH_SOURCE[0]}")/../coverage.sh"

# ONNX export only works with an older transformers version. Install it into a
# writable overlay rather than mutating the image's root-owned environment,
# unless the selected CI image already contains the required version.
if ! python -c 'import transformers; assert transformers.__version__ == "4.51.3"'; then
    TRANSFORMERS_OVERLAY="$XDG_CACHE_HOME/export-deploy/transformers-4.51.3"
    mkdir -p "$TRANSFORMERS_OVERLAY"
    uv pip install --target "$TRANSFORMERS_OVERLAY" transformers==4.51.3
    export PYTHONPATH="$TRANSFORMERS_OVERLAY${PYTHONPATH:+:$PYTHONPATH}"
fi

export CUDA_VISIBLE_DEVICES="0,1"

coverage run \
    --data-file="$PROJECT_ROOT/.coverage" \
    --source="$PROJECT_ROOT" \
    --parallel-mode \
    -m pytest \
    -o log_cli=true \
    -o log_cli_level=INFO \
    -vs -m "not pleasefixme" --tb=short tests/functional_tests/tests_onnx_trt
coverage combine -q
