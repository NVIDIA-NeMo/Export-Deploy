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

<<<<<<<< HEAD:tests/functional_tests/L2_NeMo_2_VLLM_VISION.sh
export TRANSFORMERS_OFFLINE=1
export HF_HOME=/home/TestData/vlm/vision/hf/

coverage run --data-file=/workspace/.coverage --source=/workspace --parallel-mode tests/functional_tests/utils/test_llava_next_InternVIT.py \
  --devices=1 \
  --max-steps=5

coverage run --data-file=/workspace/.coverage --source=/workspace --parallel-mode scripts/vlm/import_hf.py --input_name_or_path="OpenGVLab/InternViT-300M-448px-V2_5"

coverage run --data-file=/workspace/.coverage --source=/workspace --parallel-mode scripts/vlm/import_hf.py --input_name_or_path="openai/clip-vit-large-patch14"

coverage run --data-file=/workspace/.coverage --source=/workspace --parallel-mode scripts/vlm/import_hf.py --input_name_or_path="google/siglip-base-patch16-224"
========
export CUDA_VISIBLE_DEVICES="0,1"

coverage run \
    --data-file=/workspace/.coverage \
    --source=/workspace/ \
    --parallel-mode \
    -m pytest \
    -o log_cli=true \
    -o log_cli_level=INFO \
    -vs -m "not pleasefixme" --tb=short tests/functional_tests/tests_inframework
>>>>>>>> d66207772 (ci: Refactor tests (#157)):tests/functional_tests/L2_Launch_InFramework.sh
coverage combine
