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


import numpy as np
import pytest
from unittest.mock import Mock, patch
from nemo_deploy.deploy_base import DeployBase, ITritonDeployable


# Create a mock implementation of ITritonDeployable
class MockTritonDeployable(ITritonDeployable):
    def get_triton_input(self):
        pass

    def get_triton_output(self):
        pass

    def triton_infer_fn(self, **inputs: np.ndarray):
        pass


# Create a concrete implementation of DeployBase for testing
class TestDeployBase(DeployBase):
    def deploy(self):
        pass

    def serve(self):
        pass

    def run(self):
        pass

    def stop(self):
        pass


@pytest.fixture
def deploy_base():
    return TestDeployBase(
        triton_model_name="test_model",
        triton_model_version=1,
        model=MockTritonDeployable(),
        max_batch_size=64,
        http_port=8000,
        grpc_port=8001,
        address="0.0.0.0",
        allow_grpc=True,
        allow_http=True,
        streaming=False,
    )


def test_init_default_values():
    """Test initialization with default values"""
    deploy = TestDeployBase(triton_model_name="test_model")
    assert deploy.triton_model_name == "test_model"
    assert deploy.triton_model_version == 1
    assert deploy.max_batch_size == 128
    assert deploy.model is None
    assert deploy.http_port == 8000
    assert deploy.grpc_port == 8001
    assert deploy.address == "0.0.0.0"
    assert deploy.triton is None
    assert deploy.allow_grpc is True
    assert deploy.allow_http is True
    assert deploy.streaming is False


def test_init_custom_values():
    """Test initialization with custom values"""
    model = MockTritonDeployable()
    deploy = TestDeployBase(
        triton_model_name="custom_model",
        triton_model_version=2,
        model=model,
        max_batch_size=32,
        http_port=9000,
        grpc_port=9001,
        address="127.0.0.1",
        allow_grpc=False,
        allow_http=False,
        streaming=True,
    )
    assert deploy.triton_model_name == "custom_model"
    assert deploy.triton_model_version == 2
    assert deploy.max_batch_size == 32
    assert deploy.model == model
    assert deploy.http_port == 9000
    assert deploy.grpc_port == 9001
    assert deploy.address == "127.0.0.1"
    assert deploy.allow_grpc is False
    assert deploy.allow_http is False
    assert deploy.streaming is True


def test_is_model_deployable_valid(deploy_base):
    """Test _is_model_deployable with a valid model"""
    assert deploy_base._is_model_deployable() is True


def test_is_model_deployable_invalid():
    """Test _is_model_deployable with an invalid model"""
    deploy = TestDeployBase(triton_model_name="test_model", model=object())
    with pytest.raises(Exception) as exc_info:
        deploy._is_model_deployable()
    assert "This model is not deployable to Triton" in str(exc_info.value)


def test_abstract_methods():
    """Test that DeployBase cannot be instantiated directly"""
    with pytest.raises(TypeError):
        DeployBase(triton_model_name="test_model")


def test_abstract_methods_implementation():
    """Test that all abstract methods must be implemented"""

    class IncompleteDeployBase(DeployBase):
        pass

    with pytest.raises(TypeError):
        IncompleteDeployBase(triton_model_name="test_model")
