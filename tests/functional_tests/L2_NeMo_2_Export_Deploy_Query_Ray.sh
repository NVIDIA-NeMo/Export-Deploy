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

python scripts/deploy/nlp/deploy_ray_inframework.py \
    --nemo_checkpoint /tmp/nemo2_ckpt \
    --num_gpus 1 \
    --num_nodes 1 \
    --tensor_model_parallel_size 1 \
    --pipeline_model_parallel_size 1 \
    --context_parallel_size 1 \
    --model_id nemo-model \
    --enable_cuda_graphs \
    --enable_flash_decode \
    --cuda_visible_devices 0 \
    --num_replicas 1 &

coverage run -a --data-file=/workspace/.coverage --source=/workspace scripts/deploy/nlp/query_ray_deployment.py