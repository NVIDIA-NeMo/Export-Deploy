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

from nemo_deploy.multimodal.megatron_multimodal_deployable import MegatronMultimodalDeployable
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
    with patch("nemo_deploy.multimodal.megatron_multimodal_deployable.setup_model_and_tokenizer") as mock:
        mock.return_value = (MockInferenceWrappedModel(), MockProcessor())
        yield mock


@pytest.fixture
def mock_triton_imports():
    with patch("nemo_deploy.multimodal.megatron_multimodal_deployable.HAVE_TRITON", True):
        with patch("nemo_deploy.multimodal.megatron_multimodal_deployable.HAVE_MBRIDGE", True):
            with patch("nemo_deploy.multimodal.megatron_multimodal_deployable.batch") as mock_batch:
                with patch("nemo_deploy.multimodal.megatron_multimodal_deployable.first_value") as mock_first_value:
                    with patch("nemo_deploy.multimodal.megatron_multimodal_deployable.Tensor") as mock_tensor:
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
    with patch("nemo_deploy.multimodal.megatron_multimodal_deployable.str_ndarray2list") as mock_str2list:
        with patch("nemo_deploy.multimodal.megatron_multimodal_deployable.cast_output") as mock_cast:
            mock_str2list.return_value = ["test prompt 1", "test prompt 2"]
            mock_cast.return_value = np.array([b"Generated text 1", b"Generated text 2"])
            yield mock_str2list, mock_cast


@pytest.fixture
def sample_image():
    return Image.new("RGB", (224, 224))


@pytest.fixture
def sample_image_base64():
    """Return a mock base64-encoded image string."""
    return "mock_base64_image_string_123"


@pytest.fixture
def deployable(mock_setup_model_and_tokenizer, mock_triton_imports):
    return MegatronMultimodalDeployable(
        megatron_checkpoint_filepath="test_checkpoint.nemo",
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        params_dtype=torch.bfloat16,
        inference_batch_times_seqlen_threshold=1000,
    )


class TestMegatronMultimodalDeployable:
    def test_initialization_success(self, mock_setup_model_and_tokenizer, mock_triton_imports):
        """Test successful initialization of MegatronMultimodalDeployable."""
        deployable = MegatronMultimodalDeployable(megatron_checkpoint_filepath="test_checkpoint.nemo")

        assert deployable.megatron_checkpoint_filepath == "test_checkpoint.nemo"
        assert deployable.tensor_model_parallel_size == 1
        assert deployable.pipeline_model_parallel_size == 1
        assert deployable.params_dtype == torch.bfloat16
        assert deployable.inference_batch_times_seqlen_threshold == 1000
        assert deployable.inference_wrapped_model is not None
        assert deployable.processor is not None

    def test_initialization_with_custom_params(self, mock_setup_model_and_tokenizer, mock_triton_imports):
        """Test initialization with custom parameters."""
        deployable = MegatronMultimodalDeployable(
            megatron_checkpoint_filepath="custom_checkpoint.nemo",
            tensor_model_parallel_size=2,
            pipeline_model_parallel_size=2,
            params_dtype=torch.float16,
            inference_batch_times_seqlen_threshold=2000,
            inference_max_batch_size=8,
        )

        assert deployable.tensor_model_parallel_size == 2
        assert deployable.pipeline_model_parallel_size == 2
        assert deployable.params_dtype == torch.float16
        assert deployable.inference_batch_times_seqlen_threshold == 2000

    def test_initialization_calls_setup_model(self, mock_setup_model_and_tokenizer, mock_triton_imports):
        """Test that initialization calls setup_model_and_tokenizer with correct parameters."""
        MegatronMultimodalDeployable(
            megatron_checkpoint_filepath="test_checkpoint.nemo",
            tensor_model_parallel_size=2,
            pipeline_model_parallel_size=2,
            params_dtype=torch.float16,
            inference_batch_times_seqlen_threshold=1500,
            inference_max_seq_length=4096,
        )

        mock_setup_model_and_tokenizer.assert_called_once_with(
            megatron_model_path="test_checkpoint.nemo",
            tp=2,
            pp=2,
            params_dtype=torch.float16,
            inference_batch_times_seqlen_threshold=1500,
            inference_max_seq_length=4096,
            inference_max_batch_size=4,
        )

    def test_generate_method(self, deployable, sample_image):
        """Test the generate method."""
        prompts = ["Test prompt 1", "Test prompt 2"]
        images = [sample_image, sample_image]
        inference_params = CommonInferenceParams(temperature=0.7, top_k=10, top_p=0.9, num_tokens_to_generate=100)

        with patch("nemo_deploy.multimodal.megatron_multimodal_deployable.generate") as mock_generate:
            with patch.object(deployable, "apply_chat_template", side_effect=lambda x: x):
                mock_generate.return_value = [MockResult("Generated text 1"), MockResult("Generated text 2")]

                results = deployable.generate(
                    prompts=prompts,
                    images=images,
                    inference_params=inference_params,
                    random_seed=42,
                    apply_chat_template=True,
                )

                mock_generate.assert_called_once_with(
                    wrapped_model=deployable.inference_wrapped_model,
                    tokenizer=deployable.processor.tokenizer,
                    image_processor=deployable.processor.image_processor,
                    prompts=prompts,
                    images=images,
                    processor=deployable.processor,
                    random_seed=42,
                    sampling_params=inference_params,
                )

                assert len(results) == 2
                assert results[0].generated_text == "Generated text 1"
                assert results[1].generated_text == "Generated text 2"

    def test_generate_method_default_params(self, deployable, sample_image):
        """Test the generate method with default parameters."""
        prompts = ["Test prompt"]
        images = [sample_image]

        with patch("nemo_deploy.multimodal.megatron_multimodal_deployable.generate") as mock_generate:
            mock_generate.return_value = [MockResult("Generated text")]

            deployable.generate(prompts=prompts, images=images)

            mock_generate.assert_called_once_with(
                wrapped_model=deployable.inference_wrapped_model,
                tokenizer=deployable.processor.tokenizer,
                image_processor=deployable.processor.image_processor,
                prompts=prompts,
                images=images,
                processor=deployable.processor,
                random_seed=None,
                sampling_params=None,
            )

    def test_get_triton_input(self, deployable):
        """Test the get_triton_input property."""
        inputs = deployable.get_triton_input

        assert len(inputs) == 8

        # Check prompts input
        assert inputs[0].name == "prompts"
        assert inputs[0].shape == (-1,)
        assert inputs[0].dtype is not None  # Just check that dtype is set

        # Check images input (now base64 strings)
        assert inputs[1].name == "images"
        assert inputs[1].shape == (-1,)
        assert inputs[1].dtype is not None  # Just check that dtype is set

        # Check optional inputs
        assert inputs[2].name == "max_length"
        assert inputs[2].optional is True

        assert inputs[3].name == "top_k"
        assert inputs[3].optional is True

        assert inputs[4].name == "top_p"
        assert inputs[4].optional is True

        assert inputs[5].name == "temperature"
        assert inputs[5].optional is True

        assert inputs[6].name == "random_seed"
        assert inputs[6].optional is True

        assert inputs[7].name == "apply_chat_template"
        assert inputs[7].optional is True

    def test_get_triton_output(self, deployable):
        """Test the get_triton_output property."""
        outputs = deployable.get_triton_output

        assert len(outputs) == 1
        assert outputs[0].name == "sentences"
        assert outputs[0].shape == (-1,)
        assert outputs[0].dtype is not None  # Just check that dtype is set

    def test_infer_fn(self, deployable, sample_image_base64, sample_image):
        """Test the _infer_fn method."""
        prompts = ["Test prompt 1", "Test prompt 2"]
        images = [sample_image_base64, sample_image_base64]

        with patch.object(deployable, "process_image_input") as mock_process_image_input:
            with patch.object(deployable, "generate") as mock_generate:
                # Mock process_image_input to return PIL Images
                mock_process_image_input.return_value = sample_image
                mock_generate.return_value = [MockResult("Generated text 1"), MockResult("Generated text 2")]

                result = deployable._infer_fn(
                    prompts=prompts,
                    images=images,
                    temperature=0.8,
                    top_k=20,
                    top_p=0.95,
                    num_tokens_to_generate=150,
                    random_seed=123,
                )

                # Check that process_image_input was called for each image
                assert mock_process_image_input.call_count == 2

                # Check that generate was called with the right parameters
                assert mock_generate.call_count == 1
                call_args = mock_generate.call_args
                # Check positional arguments: prompts, images, inference_params
                assert call_args[0][0] == prompts
                # Images should be converted from base64
                assert len(call_args[0][1]) == 2
                # Check that inference_params is a CommonInferenceParams object (3rd positional arg)
                assert isinstance(call_args[0][2], CommonInferenceParams)
                assert call_args[0][2].temperature == 0.8
                assert call_args[0][2].top_k == 20
                assert call_args[0][2].top_p == 0.95
                assert call_args[0][2].num_tokens_to_generate == 150
                # Check keyword arguments
                assert call_args[1]["random_seed"] == 123

                assert "sentences" in result
                assert len(result["sentences"]) == 2
                assert result["sentences"] == ["Generated text 1", "Generated text 2"]

    def test_infer_fn_default_params(self, deployable, sample_image_base64, sample_image):
        """Test the _infer_fn method with default parameters."""
        prompts = ["Test prompt"]
        images = [sample_image_base64]

        with patch.object(deployable, "process_image_input") as mock_process_image_input:
            with patch.object(deployable, "generate") as mock_generate:
                # Mock process_image_input to return PIL Images
                mock_process_image_input.return_value = sample_image
                mock_generate.return_value = [MockResult("Generated text 1")]

                result = deployable._infer_fn(prompts=prompts, images=images)

                # Check that process_image_input was called
                assert mock_process_image_input.call_count == 1

                # Check that generate was called with the right parameters
                assert mock_generate.call_count == 1
                call_args = mock_generate.call_args
                # Check positional arguments: prompts, images, inference_params
                assert call_args[0][0] == prompts
                # Images should be converted from base64
                assert len(call_args[0][1]) == 1
                # Check that inference_params is a CommonInferenceParams object (3rd positional arg)
                assert isinstance(call_args[0][2], CommonInferenceParams)
                assert call_args[0][2].temperature == 1.0
                assert call_args[0][2].top_k == 1
                assert call_args[0][2].top_p == 0.0
                assert call_args[0][2].num_tokens_to_generate == 256
                # Check keyword arguments
                assert call_args[1]["random_seed"] is None

                assert result["sentences"] == ["Generated text 1"]

    def test_infer_fn_with_temperature_zero(self, deployable):
        """Test _infer_fn with temperature=0.0 for greedy decoding."""
        sample_image = Image.new("RGB", (100, 100))
        sample_image_base64 = "data:image;base64,test_base64_string"

        prompts = ["Test prompt"]
        images = [sample_image_base64]

        with patch.object(deployable, "process_image_input") as mock_process_image:
            with patch.object(deployable, "generate") as mock_generate:
                # Mock process_image_input to return PIL Images
                mock_process_image.return_value = sample_image
                mock_generate.return_value = [MockResult("Generated text")]

                result = deployable._infer_fn(
                    prompts=prompts,
                    images=images,
                    temperature=0.0,  # Should trigger greedy sampling handling
                    top_k=5,  # Should be overridden to 1
                    top_p=0.5,  # Should be overridden to 0.0
                    num_tokens_to_generate=100,
                )

                # Check that generate was called with the right parameters
                assert mock_generate.call_count == 1
                call_args = mock_generate.call_args

                # Check that inference_params has greedy sampling parameters
                assert isinstance(call_args[0][2], CommonInferenceParams)
                assert call_args[0][2].temperature == 0.0  # Kept as 0.0
                assert call_args[0][2].top_k == 1  # Overridden for greedy sampling
                assert call_args[0][2].top_p == 0.0  # Overridden for greedy sampling
                assert call_args[0][2].num_tokens_to_generate == 100

                assert result["sentences"] == ["Generated text"]

    def test_dict_to_str_function(self):
        """Test the dict_to_str utility function."""
        from nemo_deploy.multimodal.megatron_multimodal_deployable import dict_to_str

        test_dict = {"key1": "value1", "key2": "value2"}
        result = dict_to_str(test_dict)

        assert isinstance(result, str)
        assert json.loads(result) == test_dict

    @patch("nemo_deploy.multimodal.megatron_multimodal_deployable.HAVE_TRITON", False)
    def test_initialization_no_triton(self):
        """Test that initialization fails when Triton is not available."""
        with pytest.raises(UnavailableError):
            MegatronMultimodalDeployable(megatron_checkpoint_filepath="test_checkpoint.nemo")

    @patch("nemo_deploy.multimodal.megatron_multimodal_deployable.HAVE_MBRIDGE", False)
    def test_initialization_no_mbridge(self):
        """Test that initialization fails when Megatron Bridge is not available."""
        with pytest.raises(
            UnavailableError,
            match="megatron.bridge is not available. Please install it from https://github.com/NVIDIA-NeMo/Megatron-Bridge",
        ):
            MegatronMultimodalDeployable(megatron_checkpoint_filepath="test_checkpoint.nemo")

    def test_initialization_missing_checkpoint(self, mock_setup_model_and_tokenizer, mock_triton_imports):
        """Test initialization with missing checkpoint filepath."""
        with pytest.raises(TypeError):
            MegatronMultimodalDeployable()

    def test_generate_empty_inputs(self, deployable):
        """Test generate method with empty inputs."""
        with patch("nemo_deploy.multimodal.megatron_multimodal_deployable.generate") as mock_generate:
            mock_generate.return_value = []

            results = deployable.generate(prompts=[], images=[])
            assert len(results) == 0

    def test_generate_mismatched_inputs(self, deployable, sample_image):
        """Test generate method with mismatched prompt and image counts."""
        prompts = ["prompt1", "prompt2"]
        images = [sample_image]  # Only one image for two prompts

        with patch("nemo_deploy.multimodal.megatron_multimodal_deployable.generate") as mock_generate:
            mock_generate.return_value = [MockResult("Generated text 1"), MockResult("Generated text 2")]

            # This should work as the mock handles it, but in real scenario it might fail
            result = deployable.generate(prompts=prompts, images=images)
            assert len(result) == 2

    def test_triton_infer_fn_without_decorators(self, deployable, sample_image_base64):
        """Test triton_infer_fn using __wrapped__ to bypass decorators."""
        # Create sample inputs that would normally be processed by decorators
        inputs = {
            "prompts": np.array([b"test prompt 1", b"test prompt 2"]),
            "images": np.array([b"mock_base64_1", b"mock_base64_2"]),
            "temperature": np.array([0.7]),
            "top_k": np.array([10]),
            "top_p": np.array([0.9]),
            "max_length": np.array([100]),
            "random_seed": np.array([42]),
            "apply_chat_template": np.array([False]),
        }

        with patch("nemo_deploy.multimodal.megatron_multimodal_deployable.str_ndarray2list") as mock_str2list:
            with patch("nemo_deploy.multimodal.megatron_multimodal_deployable.cast_output") as mock_cast:
                with patch.object(deployable, "_infer_fn") as mock_infer:
                    # Setup mocks
                    mock_str2list.side_effect = [["test prompt 1", "test prompt 2"], ["mock_base64_1", "mock_base64_2"]]
                    mock_cast.return_value = np.array([b"Generated text 1", b"Generated text 2"])
                    mock_infer.return_value = {"sentences": ["Generated text 1", "Generated text 2"]}

                    # Use __wrapped__ twice to access the original function without both decorators
                    # First decorator: @first_value, Second decorator: @batch
                    original_function = deployable.triton_infer_fn.__wrapped__.__wrapped__

                    # Call the original function directly
                    result = original_function(**inputs)

                    # Verify the function was called with correct parameters
                    assert mock_str2list.call_count == 2
                    # Check the calls were made with prompts and images
                    call_args_list = [call[0][0] for call in mock_str2list.call_args_list]
                    np.testing.assert_array_equal(call_args_list[0], inputs["prompts"])
                    np.testing.assert_array_equal(call_args_list[1], inputs["images"])

                    # Verify _infer_fn was called with correct parameters
                    mock_infer.assert_called_once_with(
                        prompts=["test prompt 1", "test prompt 2"],
                        images=["mock_base64_1", "mock_base64_2"],
                        temperature=0.7,
                        top_k=10,
                        top_p=0.9,
                        num_tokens_to_generate=100,
                        random_seed=42,
                        apply_chat_template=False,
                    )

                    # Verify output formatting
                    mock_cast.assert_called_once_with(["Generated text 1", "Generated text 2"], np.bytes_)

                    # Verify final result
                    assert "sentences" in result
                    assert len(result["sentences"]) == 2
                    np.testing.assert_array_equal(
                        result["sentences"], np.array([b"Generated text 1", b"Generated text 2"])
                    )

    def test_apply_chat_template_with_dict(self, deployable):
        """Test apply_chat_template with dict input."""
        messages = [{"role": "user", "content": "Hello"}]
        expected_text = "User: Hello\n"

        # Mock the processor's apply_chat_template method
        deployable.processor.apply_chat_template = MagicMock(return_value=expected_text)

        result = deployable.apply_chat_template(messages)

        deployable.processor.apply_chat_template.assert_called_once_with(
            messages, tokenizer=False, add_generation_prompt=True
        )
        assert result == expected_text

    def test_apply_chat_template_with_json_string(self, deployable):
        """Test apply_chat_template with JSON string input."""
        messages_dict = [{"role": "user", "content": "Hello"}]
        messages_json = json.dumps(messages_dict)
        expected_text = "User: Hello\n"

        # Mock the processor's apply_chat_template method
        deployable.processor.apply_chat_template = MagicMock(return_value=expected_text)

        result = deployable.apply_chat_template(messages_json)

        deployable.processor.apply_chat_template.assert_called_once_with(
            messages_dict, tokenizer=False, add_generation_prompt=True
        )
        assert result == expected_text

    def test_apply_chat_template_without_generation_prompt(self, deployable):
        """Test apply_chat_template with add_generation_prompt=False."""
        messages = [{"role": "user", "content": "Hello"}]
        expected_text = "User: Hello"

        # Mock the processor's apply_chat_template method
        deployable.processor.apply_chat_template = MagicMock(return_value=expected_text)

        result = deployable.apply_chat_template(messages, add_generation_prompt=False)

        deployable.processor.apply_chat_template.assert_called_once_with(
            messages, tokenizer=False, add_generation_prompt=False
        )
        assert result == expected_text

    def test_process_image_input_with_qwenvl_wrapper(self, deployable):
        """Test process_image_input with QwenVLInferenceWrapper using base64 image."""
        # Create a mock QwenVLInferenceWrapper class
        mock_qwenvl_class = MagicMock()

        # Make deployable.inference_wrapped_model an instance of the mock class
        # Use isinstance check to return True for QwenVLInferenceWrapper
        deployable.inference_wrapped_model = MagicMock()

        # Image source with data URI prefix (new format)
        image_source = "data:image;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M9QDwADhgGAWjR9awAAAABJRU5ErkJggg=="
        expected_image = Image.new("RGB", (100, 100))

        with patch("nemo_deploy.multimodal.megatron_multimodal_deployable.QwenVLInferenceWrapper", mock_qwenvl_class):
            # Make isinstance return True for our mock
            with patch("nemo_deploy.multimodal.megatron_multimodal_deployable.isinstance") as mock_isinstance:
                mock_isinstance.return_value = True

                with patch("qwen_vl_utils.process_vision_info") as mock_process:
                    mock_process.return_value = (expected_image, None)

                    result = deployable.process_image_input(image_source)

                    # Verify isinstance was called to check the model type
                    mock_isinstance.assert_called_once_with(deployable.inference_wrapped_model, mock_qwenvl_class)

                    # Verify process_vision_info was called with correct format
                    call_args = mock_process.call_args[0][0]
                    assert len(call_args) == 1
                    assert call_args[0]["role"] == "user"
                    assert call_args[0]["content"][0]["type"] == "image"
                    assert call_args[0]["content"][0]["image"] == image_source

                    assert result == expected_image

    def test_process_image_input_with_http_url(self, deployable):
        """Test process_image_input with HTTP URL."""
        # Create a mock QwenVLInferenceWrapper class
        mock_qwenvl_class = MagicMock()

        # Make deployable.inference_wrapped_model an instance of the mock class
        deployable.inference_wrapped_model = MagicMock()

        # HTTP URL as image source
        image_source = "https://example.com/image.jpg"
        expected_image = Image.new("RGB", (100, 100))

        with patch("nemo_deploy.multimodal.megatron_multimodal_deployable.QwenVLInferenceWrapper", mock_qwenvl_class):
            # Make isinstance return True for our mock
            with patch("nemo_deploy.multimodal.megatron_multimodal_deployable.isinstance") as mock_isinstance:
                mock_isinstance.return_value = True

                with patch("qwen_vl_utils.process_vision_info") as mock_process:
                    mock_process.return_value = (expected_image, None)

                    result = deployable.process_image_input(image_source)

                    # Verify process_vision_info was called with URL
                    call_args = mock_process.call_args[0][0]
                    assert len(call_args) == 1
                    assert call_args[0]["role"] == "user"
                    assert call_args[0]["content"][0]["type"] == "image"
                    assert call_args[0]["content"][0]["image"] == image_source

                    assert result == expected_image

    def test_process_image_input_with_unsupported_model(self, deployable):
        """Test process_image_input with unsupported model raises ValueError."""
        # Create a mock QwenVLInferenceWrapper class
        mock_qwenvl_class = MagicMock()

        # Make sure the wrapped model is NOT a QwenVLInferenceWrapper
        deployable.inference_wrapped_model = MagicMock()

        image_source = "data:image;base64,test_base64_string"

        with patch("nemo_deploy.multimodal.megatron_multimodal_deployable.QwenVLInferenceWrapper", mock_qwenvl_class):
            # Make isinstance return False for our mock (not a QwenVLInferenceWrapper)
            with patch("nemo_deploy.multimodal.megatron_multimodal_deployable.isinstance") as mock_isinstance:
                mock_isinstance.return_value = False

                with pytest.raises(ValueError, match="not supported"):
                    deployable.process_image_input(image_source)

    def test_ray_infer_fn(self, deployable):
        """Test ray_infer_fn method."""
        inputs = {
            "prompts": ["test prompt 1", "test prompt 2"],
            "images": ["base64_image_1", "base64_image_2"],
            "temperature": 0.8,
            "top_k": 15,
            "top_p": 0.95,
            "max_length": 200,
            "random_seed": 999,
            "apply_chat_template": True,
        }

        with patch.object(deployable, "_infer_fn") as mock_infer:
            mock_infer.return_value = {"sentences": ["Generated 1", "Generated 2"]}

            result = deployable.ray_infer_fn(inputs)

            mock_infer.assert_called_once_with(
                prompts=["test prompt 1", "test prompt 2"],
                images=["base64_image_1", "base64_image_2"],
                temperature=0.8,
                top_k=15,
                top_p=0.95,
                num_tokens_to_generate=200,
                random_seed=999,
                apply_chat_template=True,
            )

            assert result == {"sentences": ["Generated 1", "Generated 2"]}

    def test_ray_infer_fn_with_default_params(self, deployable):
        """Test ray_infer_fn with default parameters."""
        inputs = {
            "prompts": ["test prompt"],
            "images": ["base64_image"],
        }

        with patch.object(deployable, "_infer_fn") as mock_infer:
            mock_infer.return_value = {"sentences": ["Generated text"]}

            result = deployable.ray_infer_fn(inputs)

            mock_infer.assert_called_once_with(
                prompts=["test prompt"],
                images=["base64_image"],
                temperature=1.0,
                top_k=1,
                top_p=0.0,
                num_tokens_to_generate=50,
                random_seed=None,
                apply_chat_template=False,
            )

            assert result == {"sentences": ["Generated text"]}

    def test_ray_infer_fn_with_empty_inputs(self, deployable):
        """Test ray_infer_fn with empty inputs dict."""
        inputs = {}

        with patch.object(deployable, "_infer_fn") as mock_infer:
            mock_infer.return_value = {"sentences": []}

            result = deployable.ray_infer_fn(inputs)

            mock_infer.assert_called_once_with(
                prompts=[],
                images=[],
                temperature=1.0,
                top_k=1,
                top_p=0.0,
                num_tokens_to_generate=50,
                random_seed=None,
                apply_chat_template=False,
            )

            assert result == {"sentences": []}
