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


from pathlib import Path
import yaml
from .utils import dot_dict


class NeMoConfig:

    def __init__(self, path):
        if path.exists():
            with open(path, 'r') as stream:
                self.config = yaml.safe_load(stream)
                if not "config" in self.config:
                    raise FileNotFoundError(
                        "A model.yaml with valid format that includes model config could not be found."
                    )
        else:
            raise FileNotFoundError("model.yaml could not be found in the checkpoint.")

    def model_config(self, return_dict=True):
        if return_dict:
            return self.config["config"]
        else:
            return dot_dict(self.config["config"])

    def tokenizer(self, return_dict=True):
        if return_dict:
            return self.config["tokenizer"]
        else:
            return dot_dict(self.config["tokenizer"])
