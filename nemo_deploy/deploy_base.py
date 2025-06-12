# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

import logging
from abc import (
    ABC,
    abstractmethod,
)
from nemo_deploy.triton_deployable import (
    ITritonDeployable,
)


LOGGER = logging.getLogger("NeMo")


class DeployBase(ABC):
    def __init__(
        self,
        triton_model_name: str,
        triton_model_version: int = 1,
        model=None,
        max_batch_size: int = 128,
        http_port: int = 8000,
        grpc_port: int = 8001,
        address="0.0.0.0",
        allow_grpc=True,
        allow_http=True,
        streaming=False,
    ):
        self.triton_model_name = triton_model_name
        self.triton_model_version = triton_model_version
        self.max_batch_size = max_batch_size
        self.model = model
        self.http_port = http_port
        self.grpc_port = grpc_port
        self.address = address
        self.triton = None
        self.allow_grpc = allow_grpc
        self.allow_http = allow_http
        self.streaming = streaming

    @abstractmethod
    def deploy(
        self,
    ):
        pass

    @abstractmethod
    def serve(
        self,
    ):
        pass

    @abstractmethod
    def run(
        self,
    ):
        pass

    @abstractmethod
    def stop(
        self,
    ):
        pass

    def _is_model_deployable(
        self,
    ):
        if not issubclass(
            type(self.model),
            ITritonDeployable,
        ):
            raise Exception(
                "This model is not deployable to Triton.nemo_deploy.ITritonDeployable class should be inherited"
            )
        else:
            return True
