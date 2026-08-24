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


import logging

from nemo_deploy.deploy_base import DeployBase
from nemo_export_deploy_common.import_utils import MISSING_TRITON_MSG, UnavailableError

LOGGER = logging.getLogger("NeMo")

try:
    from pytriton.model_config import ModelConfig
    from pytriton.triton import Triton, TritonConfig

    HAVE_TRITON = True
except (ImportError, ModuleNotFoundError):
    HAVE_TRITON = False


class DeployPyTriton(DeployBase):
    """Deploys any models to Triton Inference Server that implements ITritonDeployable interface in nemo_deploy."""

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
        pytriton_log_verbose=0,
    ):
        """A nemo checkpoint or model is expected for serving on Triton Inference Server.

        Args:
            triton_model_name (str): Name for the service
            triton_model_version(int): Version for the service
            checkpoint_path (str): path of the nemo file
            model (ITritonDeployable): A model that implements the ITritonDeployable from nemo_deploy import ITritonDeployable
            max_batch_size (int): max batch size
            port (int) : port for the Triton server
            address (str): http address for Triton server to bind.
        """
        super().__init__(
            triton_model_name=triton_model_name,
            triton_model_version=triton_model_version,
            model=model,
            max_batch_size=max_batch_size,
            http_port=http_port,
            grpc_port=grpc_port,
            address=address,
            allow_grpc=allow_grpc,
            allow_http=allow_http,
            streaming=streaming,
        )
        self.pytriton_log_verbose = pytriton_log_verbose

    def deploy(self):
        """Deploys any models to Triton Inference Server."""
        if not HAVE_TRITON:
            raise UnavailableError(MISSING_TRITON_MSG)

        try:
            if self.streaming:
                triton_config = TritonConfig(
                    log_verbose=self.pytriton_log_verbose,
                    allow_grpc=self.allow_grpc,
                    allow_http=self.allow_http,
                    grpc_address=self.address,
                )
                self.triton = Triton(config=triton_config)
                self.triton.bind(
                    model_name=self.triton_model_name,
                    model_version=self.triton_model_version,
                    infer_func=self.model.triton_infer_fn_streaming,
                    inputs=self.model.get_triton_input,
                    outputs=self.model.get_triton_output,
                    config=ModelConfig(decoupled=True),
                )
            else:
                triton_config = TritonConfig(
                    http_address=self.address,
                    http_port=self.http_port,
                    grpc_address=self.address,
                    grpc_port=self.grpc_port,
                    allow_grpc=self.allow_grpc,
                    allow_http=self.allow_http,
                )
                self.triton = Triton(config=triton_config)
                self.triton.bind(
                    model_name=self.triton_model_name,
                    model_version=self.triton_model_version,
                    infer_func=self.model.triton_infer_fn,
                    inputs=self.model.get_triton_input,
                    outputs=self.model.get_triton_output,
                    config=ModelConfig(max_batch_size=self.max_batch_size),
                )
        except Exception as e:
            self.triton = None
            LOGGER.error(e)

    def serve(self):
        """Starts serving the model and waits for the requests."""
        if self.triton is None:
            raise Exception("deploy should be called first.")

        try:
            self.triton.serve()
        except Exception as e:
            self.triton = None
            LOGGER.error(e)

    def run(self):
        """Starts serving the model asynchronously."""
        if self.triton is None:
            raise Exception("deploy should be called first.")

        self.triton.run()

    def stop(self):
        """Stops serving the model."""
        if self.triton is None:
            raise Exception("deploy should be called first.")

        self.triton.stop()
