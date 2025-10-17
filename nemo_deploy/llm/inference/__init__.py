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

try:
    from nemo_deploy.llm.inference.inference_base import (
        create_mcore_engine,
        setup_megatron_model_and_tokenizer_for_inference,
        setup_model_and_tokenizer_for_inference,
    )
    from nemo_deploy.llm.inference.tron_utils import DistributedInitConfig, RNGConfig

    __all__ = [
        "create_mcore_engine",
        "setup_model_and_tokenizer_for_inference",
        "setup_megatron_model_and_tokenizer_for_inference",
        "DistributedInitConfig",
        "RNGConfig",
    ]
except (ImportError, ModuleNotFoundError):
    # If imports fail, export empty list
    __all__ = []
