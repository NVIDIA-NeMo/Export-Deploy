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

import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

from nemo_deploy.multimodal.query_multimodal import NemoQueryMultimodal, NemoQueryMultimodalPytorch
from nemo_export_deploy_common.import_utils import UnavailableError


class TestNemoQueryMultimodal:
    @pytest.fixture
    def query_multimodal(self):
        return NemoQueryMultimodal(url="localhost", model_name="test_model", model_type="neva")

    @pytest.fixture
    def mock_image(self):
        # Create a temporary image file
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            img = Image.new("RGB", (100, 100), color="red")
            img.save(tmp.name)
            return tmp.name

    @pytest.fixture
    def mock_video(self):
        # Create a temporary video file
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            # Just create an empty file for testing
            return tmp.name

    @pytest.fixture
    def mock_audio(self):
        # Create a temporary audio file
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            # Just create an empty file for testing
            return tmp.name

    def test_init(self):
        nq = NemoQueryMultimodal(url="localhost", model_name="test_model", model_type="neva")
        assert nq.url == "localhost"
        assert nq.model_name == "test_model"
        assert nq.model_type == "neva"

    def test_setup_media_image_local(self, query_multimodal, mock_image):
        result = query_multimodal.setup_media(mock_image)
        assert isinstance(result, np.ndarray)
        assert result.shape[0] == 1  # Batch dimension
        os.unlink(mock_image)

    @patch("requests.get")
    def test_setup_media_image_url(self, mock_get, query_multimodal):
        # Mock the response from requests.get
        mock_response = MagicMock()
        mock_response.content = b"fake_image_data"
        mock_get.return_value = mock_response

        # Mock Image.open
        with patch("PIL.Image.open") as mock_image_open:
            mock_image = MagicMock()
            mock_image.convert.return_value = mock_image
            mock_image_open.return_value = mock_image

            result = query_multimodal.setup_media("http://example.com/image.jpg")
            assert isinstance(result, np.ndarray)
            assert result.shape[0] == 1

    def test_frame_len(self, query_multimodal):
        # Test with frames less than max_frames
        frames = [np.zeros((100, 100, 3)) for _ in range(100)]
        assert query_multimodal.frame_len(frames) == 100

        # Test with frames more than max_frames
        frames = [np.zeros((100, 100, 3)) for _ in range(300)]
        result = query_multimodal.frame_len(frames)
        assert result <= 256  # Should be less than or equal to max_frames

    def test_get_subsampled_frames(self, query_multimodal):
        frames = [np.zeros((100, 100, 3)) for _ in range(10)]
        subsample_len = 5
        result = query_multimodal.get_subsampled_frames(frames, subsample_len)
        assert len(result) == subsample_len

    @patch("nemo_deploy.multimodal.query_multimodal.ModelClient")
    def test_query(self, mock_model_client, query_multimodal, mock_image):
        # Mock the ModelClient context manager
        mock_client_instance = MagicMock()
        mock_client_instance.infer_batch.return_value = {"outputs": np.array(["test response"])}
        mock_client_instance.model_config.outputs = [MagicMock(dtype=np.bytes_)]
        mock_model_client.return_value.__enter__.return_value = mock_client_instance

        result = query_multimodal.query(
            input_text="test prompt",
            input_media=mock_image,
            max_output_len=30,
            top_k=1,
            top_p=0.0,
            temperature=1.0,
        )

        assert isinstance(result, np.ndarray)
        assert result[0] == "test response"
        os.unlink(mock_image)

    @patch("nemo_deploy.multimodal.query_multimodal.VideoReader")
    def test_setup_media_video(self, mock_video_reader, mock_video):
        nq = NemoQueryMultimodal(url="localhost", model_name="test_model", model_type="video-neva")

        # Mock VideoReader
        mock_frames = [MagicMock(asnumpy=lambda: np.zeros((100, 100, 3))) for _ in range(10)]
        mock_video_reader.return_value = mock_frames

        result = nq.setup_media(mock_video)
        assert isinstance(result, np.ndarray)
        os.unlink(mock_video)


class TestNemoQueryMultimodalPytorch:
    @pytest.fixture
    def query_multimodal_pytorch(self):
        return NemoQueryMultimodalPytorch(url="localhost", model_name="test_model")

    @pytest.fixture
    def mock_images(self):
        # Create sample PIL images for testing
        return [Image.new("RGB", (224, 224), color="red") for _ in range(2)]

    @pytest.fixture
    def mock_prompts(self):
        return ["What is in this image?", "Describe this picture"]

    def test_init(self):
        """Test successful initialization of NemoQueryMultimodalPytorch."""
        nq = NemoQueryMultimodalPytorch(url="localhost:8000", model_name="qwen-vl")
        assert nq.url == "localhost:8000"
        assert nq.model_name == "qwen-vl"

    def test_init_with_different_params(self):
        """Test initialization with different URL and model name."""
        nq = NemoQueryMultimodalPytorch(url="127.0.0.1:8001", model_name="llava")
        assert nq.url == "127.0.0.1:8001"
        assert nq.model_name == "llava"

    @patch("nemo_deploy.multimodal.query_multimodal.HAVE_TRITON", True)
    @patch("nemo_deploy.multimodal.query_multimodal.ModelClient")
    def test_query_multimodal_basic(self, mock_model_client, query_multimodal_pytorch, mock_prompts, mock_images):
        """Test basic query_multimodal functionality with minimal parameters."""
        # Mock the ModelClient context manager
        mock_client_instance = MagicMock()
        mock_client_instance.infer_batch.return_value = {
            "sentences": np.array([b"Generated text 1", b"Generated text 2"])
        }
        mock_client_instance.model_config.outputs = [MagicMock(dtype=np.bytes_)]
        mock_model_client.return_value.__enter__.return_value = mock_client_instance

        result = query_multimodal_pytorch.query_multimodal(prompts=mock_prompts, images=mock_images)

        assert isinstance(result, dict)
        assert "sentences" in result
        assert len(result["sentences"]) == 2
        assert result["sentences"][0] == "Generated text 1"
        assert result["sentences"][1] == "Generated text 2"

    @patch("nemo_deploy.multimodal.query_multimodal.HAVE_TRITON", True)
    @patch("nemo_deploy.multimodal.query_multimodal.ModelClient")
    def test_query_multimodal_with_all_params(
        self, mock_model_client, query_multimodal_pytorch, mock_prompts, mock_images
    ):
        """Test query_multimodal with all optional parameters."""
        # Mock the ModelClient context manager
        mock_client_instance = MagicMock()
        mock_client_instance.infer_batch.return_value = {
            "sentences": np.array([b"Generated text 1", b"Generated text 2"])
        }
        mock_client_instance.model_config.outputs = [MagicMock(dtype=np.bytes_)]
        mock_model_client.return_value.__enter__.return_value = mock_client_instance

        result = query_multimodal_pytorch.query_multimodal(
            prompts=mock_prompts,
            images=mock_images,
            max_length=100,
            max_batch_size=4,
            top_k=10,
            top_p=0.9,
            temperature=0.8,
            random_seed=42,
            init_timeout=120.0,
        )

        assert isinstance(result, dict)
        assert "sentences" in result
        assert len(result["sentences"]) == 2

    @patch("nemo_deploy.multimodal.query_multimodal.HAVE_TRITON", True)
    @patch("nemo_deploy.multimodal.query_multimodal.ModelClient")
    def test_query_multimodal_input_preparation(
        self, mock_model_client, query_multimodal_pytorch, mock_prompts, mock_images
    ):
        """Test that inputs are properly prepared for the ModelClient."""
        # Mock the ModelClient context manager
        mock_client_instance = MagicMock()
        mock_client_instance.infer_batch.return_value = {"sentences": np.array([b"Generated text"])}
        mock_client_instance.model_config.outputs = [MagicMock(dtype=np.bytes_)]
        mock_model_client.return_value.__enter__.return_value = mock_client_instance

        query_multimodal_pytorch.query_multimodal(
            prompts=mock_prompts,
            images=mock_images,
            max_length=50,
            top_k=5,
            top_p=0.8,
            temperature=0.7,
            random_seed=123,
        )

        # Verify that ModelClient was called with the right inputs
        mock_client_instance.infer_batch.assert_called_once()
        call_args = mock_client_instance.infer_batch.call_args[1]  # Get kwargs

        # Check that all expected inputs are present
        assert "prompts" in call_args
        assert "images" in call_args
        assert "max_length" in call_args
        assert "top_k" in call_args
        assert "top_p" in call_args
        assert "temperature" in call_args
        assert "random_seed" in call_args

        # Check data types
        assert call_args["max_length"].dtype == np.int_
        assert call_args["top_k"].dtype == np.int_
        assert call_args["top_p"].dtype == np.single
        assert call_args["temperature"].dtype == np.single
        assert call_args["random_seed"].dtype == np.int_

    @patch("nemo_deploy.multimodal.query_multimodal.HAVE_TRITON", True)
    @patch("nemo_deploy.multimodal.query_multimodal.ModelClient")
    def test_query_multimodal_non_bytes_output(
        self, mock_model_client, query_multimodal_pytorch, mock_prompts, mock_images
    ):
        """Test query_multimodal with non-bytes output type."""
        # Mock the ModelClient context manager
        mock_client_instance = MagicMock()
        mock_client_instance.infer_batch.return_value = {"output": np.array([1, 2, 3])}
        mock_client_instance.model_config.outputs = [MagicMock(dtype=np.int32)]
        mock_model_client.return_value.__enter__.return_value = mock_client_instance

        result = query_multimodal_pytorch.query_multimodal(prompts=mock_prompts, images=mock_images)

        # Should return raw output for non-bytes type
        assert "output" in result
        assert np.array_equal(result["output"], np.array([1, 2, 3]))

    @patch("nemo_deploy.multimodal.query_multimodal.HAVE_TRITON", True)
    @patch("nemo_deploy.multimodal.query_multimodal.ModelClient")
    def test_query_multimodal_missing_sentences_key(
        self, mock_model_client, query_multimodal_pytorch, mock_prompts, mock_images
    ):
        """Test query_multimodal when 'sentences' key is missing from output."""
        # Mock the ModelClient context manager
        mock_client_instance = MagicMock()
        mock_client_instance.infer_batch.return_value = {"other_key": np.array([b"text"])}
        mock_client_instance.model_config.outputs = [MagicMock(dtype=np.bytes_)]
        mock_model_client.return_value.__enter__.return_value = mock_client_instance

        result = query_multimodal_pytorch.query_multimodal(prompts=mock_prompts, images=mock_images)

        # Should return error message when 'sentences' key is missing
        assert result == "Unknown output keyword: sentences not found"

    @patch("nemo_deploy.multimodal.query_multimodal.HAVE_TRITON", False)
    def test_query_multimodal_no_triton(self, query_multimodal_pytorch, mock_prompts, mock_images):
        """Test that query_multimodal raises UnavailableError when Triton is not available."""
        with pytest.raises(UnavailableError):
            query_multimodal_pytorch.query_multimodal(prompts=mock_prompts, images=mock_images)

    @patch("nemo_deploy.multimodal.query_multimodal.HAVE_TRITON", True)
    @patch("nemo_deploy.multimodal.query_multimodal.ModelClient")
    def test_query_multimodal_single_prompt_single_image(self, mock_model_client, query_multimodal_pytorch):
        """Test query_multimodal with single prompt and single image."""
        # Mock the ModelClient context manager
        mock_client_instance = MagicMock()
        mock_client_instance.infer_batch.return_value = {"sentences": np.array([b"Single response"])}
        mock_client_instance.model_config.outputs = [MagicMock(dtype=np.bytes_)]
        mock_model_client.return_value.__enter__.return_value = mock_client_instance

        result = query_multimodal_pytorch.query_multimodal(
            prompts=["Single prompt"], images=[Image.new("RGB", (224, 224), color="blue")]
        )

        assert isinstance(result, dict)
        assert "sentences" in result
        assert len(result["sentences"]) == 1
        assert result["sentences"][0] == "Single response"

    @patch("nemo_deploy.multimodal.query_multimodal.HAVE_TRITON", True)
    @patch("nemo_deploy.multimodal.query_multimodal.ModelClient")
    def test_query_multimodal_timeout_handling(
        self, mock_model_client, query_multimodal_pytorch, mock_prompts, mock_images
    ):
        """Test that init_timeout is properly passed to ModelClient."""
        # Mock the ModelClient context manager
        mock_client_instance = MagicMock()
        mock_client_instance.infer_batch.return_value = {"sentences": np.array([b"Generated text"])}
        mock_client_instance.model_config.outputs = [MagicMock(dtype=np.bytes_)]
        mock_model_client.return_value.__enter__.return_value = mock_client_instance

        custom_timeout = 180.0
        query_multimodal_pytorch.query_multimodal(prompts=mock_prompts, images=mock_images, init_timeout=custom_timeout)

        # Verify ModelClient was called with the custom timeout
        mock_model_client.assert_called_once_with(
            query_multimodal_pytorch.url, query_multimodal_pytorch.model_name, init_timeout_s=custom_timeout
        )

    @patch("nemo_deploy.multimodal.query_multimodal.HAVE_TRITON", True)
    @patch("nemo_deploy.multimodal.query_multimodal.ModelClient")
    def test_query_multimodal_optional_params_none(
        self, mock_model_client, query_multimodal_pytorch, mock_prompts, mock_images
    ):
        """Test that optional parameters set to None are not included in inputs."""
        # Mock the ModelClient context manager
        mock_client_instance = MagicMock()
        mock_client_instance.infer_batch.return_value = {"sentences": np.array([b"Generated text"])}
        mock_client_instance.model_config.outputs = [MagicMock(dtype=np.bytes_)]
        mock_model_client.return_value.__enter__.return_value = mock_client_instance

        query_multimodal_pytorch.query_multimodal(
            prompts=mock_prompts,
            images=mock_images,
            max_length=None,
            top_k=None,
            top_p=None,
            temperature=None,
            random_seed=None,
        )

        # Verify that ModelClient was called with only required inputs
        mock_client_instance.infer_batch.assert_called_once()
        call_args = mock_client_instance.infer_batch.call_args[1]  # Get kwargs

        # Only required inputs should be present
        assert "prompts" in call_args
        assert "images" in call_args
        assert "max_length" not in call_args
        assert "top_k" not in call_args
        assert "top_p" not in call_args
        assert "temperature" not in call_args
        assert "random_seed" not in call_args
