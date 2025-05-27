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
python tests/functional_tests/utils/test_hf_import.py \
  --hf_model meta-llama/Llama-3.2-1B \
  --output_path /tmp/nemo2_ckpt \
  --config Llama32Config1B

coverage run -a --data-file=/workspace/.coverage --source=/workspace tests/functional_tests/utils/run_nemo_export.py \
  --model_name test \
  --model_type llama \
  --checkpoint_dir /tmp/nemo2_ckpt \
  --min_tps 1 \
  --in_framework True \
  --test_deployment True \
  --run_accuracy True \
  --test_data_path tests/functional_tests/data/lambada.json \
  --accuracy_threshold 0.0 \
  --debug
