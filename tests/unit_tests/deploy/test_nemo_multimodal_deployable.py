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

import json
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from megatron.core.inference.common_inference_params import CommonInferenceParams
from PIL import Image

from nemo_deploy.multimodal.nemo_multimodal_deployable import NeMoMultimodalDeployable
from nemo_export_deploy_common.import_utils import UnavailableError


class MockProcessor:
    def __init__(self):
        self.tokenizer = MagicMock()
        self.image_processor = MagicMock()


class MockInferenceWrappedModel:
    def __init__(self):
        pass


class MockResult:
    def __init__(self, generated_text):
        self.generated_text = generated_text


@pytest.fixture
def mock_setup_model_and_tokenizer():
    with patch("nemo_deploy.multimodal.nemo_multimodal_deployable.setup_model_and_tokenizer") as mock:
        mock.return_value = (MockInferenceWrappedModel(), MockProcessor())
        yield mock


@pytest.fixture
def mock_triton_imports():
    with patch("nemo_deploy.multimodal.nemo_multimodal_deployable.HAVE_TRITON", True):
        with patch("nemo_deploy.multimodal.nemo_multimodal_deployable.batch") as mock_batch:
            with patch("nemo_deploy.multimodal.nemo_multimodal_deployable.first_value") as mock_first_value:
                with patch("nemo_deploy.multimodal.nemo_multimodal_deployable.Tensor") as mock_tensor:
                    mock_batch.return_value = lambda x: x
                    mock_first_value.return_value = lambda x: x

                    # Create a proper mock Tensor class that can be instantiated with parameters
                    def create_tensor(**kwargs):
                        tensor_mock = MagicMock()
                        tensor_mock.name = kwargs.get("name")
                        tensor_mock.shape = kwargs.get("shape")
                        tensor_mock.dtype = kwargs.get("dtype")
                        tensor_mock.optional = kwargs.get("optional", False)

                        return tensor_mock

                    mock_tensor.side_effect = create_tensor
                    yield mock_batch, mock_first_value, mock_tensor


@pytest.fixture
def mock_utils():
    with patch("nemo_deploy.multimodal.nemo_multimodal_deployable.str_ndarray2list") as mock_str2list:
        with patch("nemo_deploy.multimodal.nemo_multimodal_deployable.ndarray2img") as mock_ndarray2img:
            with patch("nemo_deploy.multimodal.nemo_multimodal_deployable.cast_output") as mock_cast:
                mock_str2list.return_value = ["test prompt 1", "test prompt 2"]
                mock_ndarray2img.return_value = [Image.new("RGB", (224, 224)), Image.new("RGB", (224, 224))]
                mock_cast.return_value = np.array([b"Generated text 1", b"Generated text 2"])
                yield mock_str2list, mock_ndarray2img, mock_cast


@pytest.fixture
def sample_image():
    return Image.new("RGB", (224, 224))


@pytest.fixture
def deployable(mock_setup_model_and_tokenizer, mock_triton_imports):
    return NeMoMultimodalDeployable(
        nemo_checkpoint_filepath="test_checkpoint.nemo",
        tensor_parallel_size=1,
        pipeline_parallel_size=1,
        params_dtype=torch.bfloat16,
        inference_batch_times_seqlen_threshold=1000,
    )


class TestNeMoMultimodalDeployable:
    def test_initialization_success(self, mock_setup_model_and_tokenizer, mock_triton_imports):
        """Test successful initialization of NeMoMultimodalDeployable."""
        deployable = NeMoMultimodalDeployable(nemo_checkpoint_filepath="test_checkpoint.nemo")

        assert deployable.nemo_checkpoint_filepath == "test_checkpoint.nemo"
        assert deployable.tensor_parallel_size == 1
        assert deployable.pipeline_parallel_size == 1
        assert deployable.params_dtype == torch.bfloat16
        assert deployable.inference_batch_times_seqlen_threshold == 1000
        assert deployable.inference_wrapped_model is not None
        assert deployable.processor is not None

    def test_initialization_with_custom_params(self, mock_setup_model_and_tokenizer, mock_triton_imports):
        """Test initialization with custom parameters."""
        deployable = NeMoMultimodalDeployable(
            nemo_checkpoint_filepath="custom_checkpoint.nemo",
            tensor_parallel_size=2,
            pipeline_parallel_size=2,
            params_dtype=torch.float16,
            inference_batch_times_seqlen_threshold=2000,
        )

        assert deployable.tensor_parallel_size == 2
        assert deployable.pipeline_parallel_size == 2
        assert deployable.params_dtype == torch.float16
        assert deployable.inference_batch_times_seqlen_threshold == 2000

    def test_initialization_calls_setup_model(self, mock_setup_model_and_tokenizer, mock_triton_imports):
        """Test that initialization calls setup_model_and_tokenizer with correct parameters."""
        NeMoMultimodalDeployable(
            nemo_checkpoint_filepath="test_checkpoint.nemo",
            tensor_parallel_size=2,
            pipeline_parallel_size=2,
            params_dtype=torch.float16,
            inference_batch_times_seqlen_threshold=1500,
        )

        mock_setup_model_and_tokenizer.assert_called_once_with(
            path="test_checkpoint.nemo",
            tp_size=2,
            pp_size=2,
            params_dtype=torch.float16,
            inference_batch_times_seqlen_threshold=1500,
        )

    def test_generate_method(self, deployable, sample_image):
        """Test the generate method."""
        prompts = ["Test prompt 1", "Test prompt 2"]
        images = [sample_image, sample_image]
        inference_params = CommonInferenceParams(temperature=0.7, top_k=10, top_p=0.9, num_tokens_to_generate=100)

        with patch("nemo_deploy.multimodal.nemo_multimodal_deployable.generate") as mock_generate:
            mock_generate.return_value = [MockResult("Generated text 1"), MockResult("Generated text 2")]

            results = deployable.generate(
                prompts=prompts, images=images, inference_params=inference_params, max_batch_size=2, random_seed=42
            )

            mock_generate.assert_called_once_with(
                wrapped_model=deployable.inference_wrapped_model,
                tokenizer=deployable.processor.tokenizer,
                image_processor=deployable.processor.image_processor,
                prompts=prompts,
                images=images,
                processor=deployable.processor,
                max_batch_size=2,
                random_seed=42,
                inference_params=inference_params,
            )

            assert len(results) == 2
            assert results[0].generated_text == "Generated text 1"
            assert results[1].generated_text == "Generated text 2"

    def test_generate_method_default_params(self, deployable, sample_image):
        """Test the generate method with default parameters."""
        prompts = ["Test prompt"]
        images = [sample_image]

        with patch("nemo_deploy.multimodal.nemo_multimodal_deployable.generate") as mock_generate:
            mock_generate.return_value = [MockResult("Generated text")]

            deployable.generate(prompts=prompts, images=images)

            mock_generate.assert_called_once_with(
                wrapped_model=deployable.inference_wrapped_model,
                tokenizer=deployable.processor.tokenizer,
                image_processor=deployable.processor.image_processor,
                prompts=prompts,
                images=images,
                processor=deployable.processor,
                max_batch_size=4,
                random_seed=None,
                inference_params=None,
            )

    def test_get_triton_input(self, deployable):
        """Test the get_triton_input property."""
        inputs = deployable.get_triton_input

        assert len(inputs) == 8

        # Check prompts input
        assert inputs[0].name == "prompts"
        assert inputs[0].shape == (-1,)
        assert inputs[0].dtype is not None  # Just check that dtype is set

        # Check images input
        assert inputs[1].name == "images"
        assert inputs[1].shape == (-1, -1, -1, 3)
        assert inputs[1].dtype is not None  # Just check that dtype is set

        # Check optional inputs
        assert inputs[2].name == "max_length"
        assert inputs[2].optional is True

        assert inputs[3].name == "max_batch_size"
        assert inputs[3].optional is True

        assert inputs[4].name == "top_k"
        assert inputs[4].optional is True

        assert inputs[5].name == "top_p"
        assert inputs[5].optional is True

        assert inputs[6].name == "temperature"
        assert inputs[6].optional is True

        assert inputs[7].name == "random_seed"
        assert inputs[7].optional is True

    def test_get_triton_output(self, deployable):
        """Test the get_triton_output property."""
        outputs = deployable.get_triton_output

        assert len(outputs) == 1
        assert outputs[0].name == "sentences"
        assert outputs[0].shape == (-1,)
        assert outputs[0].dtype is not None  # Just check that dtype is set

    def test_infer_fn(self, deployable, sample_image):
        """Test the _infer_fn method."""
        prompts = ["Test prompt 1", "Test prompt 2"]
        images = [sample_image, sample_image]

        with patch.object(deployable, "generate") as mock_generate:
            mock_generate.return_value = [MockResult("Generated text 1"), MockResult("Generated text 2")]

            result = deployable._infer_fn(
                prompts=prompts,
                images=images,
                temperature=0.8,
                top_k=20,
                top_p=0.95,
                num_tokens_to_generate=150,
                random_seed=123,
                max_batch_size=3,
            )

            # Check that generate was called with the right parameters
            assert mock_generate.call_count == 1
            call_args = mock_generate.call_args
            # Check positional arguments: prompts, images, inference_params
            assert call_args[0][0] == prompts
            assert call_args[0][1] == images
            # Check that inference_params is a CommonInferenceParams object (3rd positional arg)
            assert isinstance(call_args[0][2], CommonInferenceParams)
            assert call_args[0][2].temperature == 0.8
            assert call_args[0][2].top_k == 20
            assert call_args[0][2].top_p == 0.95
            assert call_args[0][2].num_tokens_to_generate == 150
            # Check keyword arguments
            assert call_args[1]["max_batch_size"] == 3
            assert call_args[1]["random_seed"] == 123

            assert "sentences" in result
            assert len(result["sentences"]) == 2
            assert result["sentences"] == ["Generated text 1", "Generated text 2"]

    def test_infer_fn_default_params(self, deployable, sample_image):
        """Test the _infer_fn method with default parameters."""
        prompts = ["Test prompt"]
        images = [sample_image]

        with patch.object(deployable, "generate") as mock_generate:
            mock_generate.return_value = [MockResult("Generated text 1")]

            result = deployable._infer_fn(prompts=prompts, images=images)

            # Check that generate was called with the right parameters
            assert mock_generate.call_count == 1
            call_args = mock_generate.call_args
            # Check positional arguments: prompts, images, inference_params
            assert call_args[0][0] == prompts
            assert call_args[0][1] == images
            # Check that inference_params is a CommonInferenceParams object (3rd positional arg)
            assert isinstance(call_args[0][2], CommonInferenceParams)
            assert call_args[0][2].temperature == 1.0
            assert call_args[0][2].top_k == 1
            assert call_args[0][2].top_p == 0.0
            assert call_args[0][2].num_tokens_to_generate == 256
            # Check keyword arguments
            assert call_args[1]["max_batch_size"] == 4
            assert call_args[1]["random_seed"] is None

            assert result["sentences"] == ["Generated text 1"]

    def test_dict_to_str_function(self):
        """Test the dict_to_str utility function."""
        from nemo_deploy.multimodal.nemo_multimodal_deployable import dict_to_str

        test_dict = {"key1": "value1", "key2": "value2"}
        result = dict_to_str(test_dict)

        assert isinstance(result, str)
        assert json.loads(result) == test_dict

    @patch("nemo_deploy.multimodal.nemo_multimodal_deployable.HAVE_TRITON", False)
    def test_initialization_no_triton(self):
        """Test that initialization fails when Triton is not available."""
        with pytest.raises(UnavailableError):
            NeMoMultimodalDeployable(nemo_checkpoint_filepath="test_checkpoint.nemo")

    @patch("nemo_deploy.multimodal.nemo_multimodal_deployable.HAVE_NEMO", False)
    def test_initialization_no_nemo(self):
        """Test that initialization fails when NeMo is not available."""
        with pytest.raises(UnavailableError, match="nemo is not available. Please install it with `pip install nemo`."):
            NeMoMultimodalDeployable(nemo_checkpoint_filepath="test_checkpoint.nemo")

    def test_initialization_missing_checkpoint(self, mock_triton_imports):
        """Test initialization with missing checkpoint filepath."""
        with pytest.raises(TypeError):
            NeMoMultimodalDeployable()

    def test_generate_empty_inputs(self, deployable):
        """Test generate method with empty inputs."""
        with patch("nemo_deploy.multimodal.nemo_multimodal_deployable.generate") as mock_generate:
            mock_generate.return_value = []

            results = deployable.generate(prompts=[], images=[])
            assert len(results) == 0

    def test_generate_mismatched_inputs(self, deployable, sample_image):
        """Test generate method with mismatched prompt and image counts."""
        prompts = ["prompt1", "prompt2"]
        images = [sample_image]  # Only one image for two prompts

        with patch("nemo_deploy.multimodal.nemo_multimodal_deployable.generate") as mock_generate:
            mock_generate.return_value = [MockResult("Generated text 1"), MockResult("Generated text 2")]

            # This should work as the mock handles it, but in real scenario it might fail
            result = deployable.generate(prompts=prompts, images=images)
            assert len(result) == 2
