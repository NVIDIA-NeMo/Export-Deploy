# Copyright (c) 2025, NVIDIA CORPORATION.
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

python tests/functional_tests/utils/create_hf_model.py \
  --model_name_or_path /home/TestData/hf/Llama-2-7b-hf \
  --output_dir /tmp/llama_tiny_hf \
  --config_updates "{\"num_hidden_layers\": 2, \"hidden_size\": 512, \"intermediate_size\": 384, \"num_attention_heads\": 8, \"num_key_value_heads\": 8}"

python tests/functional_tests/utils/test_hf_import.py \
  --hf_model /tmp/llama_tiny_hf \
  --output_path /tmp/nemo2_ckpt

python tests/functional_tests/utils/create_ptq_ckpt.py \
  --nemo_checkpoint /tmp/nemo2_ckpt \
  --algorithm int8_sq \
  --calibration_dataset tests/functional_tests/data/calibration_dataset.json \
  --calibration_batch_size 2 \
  --calibration_dataset_size 6 \
  --export_format trtllm \
  --export_path /tmp/nemo2_ptq \
  --generate_sample

coverage run -a --data-file=/workspace/.coverage --source=/workspace/nemo tests/functional_tests/utils/run_nemo_export.py \
  --model_name test \
  --model_dir /tmp/trt_llm_model_dir/ \
  --checkpoint_dir /tmp/nemo2_ptq \
  --min_tps 1 \
  --test_deployment True \
  --debug
