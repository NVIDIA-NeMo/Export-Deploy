# Copyright (c) 2020-2025, NVIDIA CORPORATION.
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

coverage run --data-file=/workspace/.coverage --source=/workspace --parallel-mode tests/functional_tests/utils/run_nemo_deploy.py \
  --model_name test_model \
  --checkpoint_dir /home/TestData/llm/models/llama32_1b_nemo2 \
  --backend TensorRT-LLM \
  --min_gpus 1 \
  --max_gpus 2 \
  --run_accuracy True \
  --test_data_path tests/functional_tests/data/lambada.json \
  --test_deployment True \
  --debug

coverage combine
