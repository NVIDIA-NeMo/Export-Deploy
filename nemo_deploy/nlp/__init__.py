# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import warnings

warnings.warn(
    "The 'nemo_deploy.nlp' module is deprecated and will be renamed to 'nemo_deploy.llm' in the next release. "
    "To ensure compatibility with future versions, please change your imports from 'nemo_deploy.nlp' to 'nemo_deploy.llm' as soon as possible.",
    DeprecationWarning,
    stacklevel=2,
)

from nemo_deploy.nlp.query_llm import (
    NemoQueryLLM,
    NemoQueryLLMHF,
    NemoQueryLLMPyTorch,
    NemoQueryTRTLLMAPI,
    NemoQueryvLLM,
)

__all__ = ["NemoQueryLLM", "NemoQueryLLMHF", "NemoQueryLLMPyTorch", "NemoQueryTRTLLMAPI", "NemoQueryvLLM"]
