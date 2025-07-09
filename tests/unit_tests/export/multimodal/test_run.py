# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
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
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from PIL import Image

from nemo_export_deploy_common.import_utils import UnavailableError

try:
    import tensorrt_llm  # noqa: F401

    HAVE_TRT_LLM = True
except ImportError:
    HAVE_TRT_LLM = False


@pytest.mark.skipif(not HAVE_TRT_LLM, reason="TensorRT-LLM is not installed")
def test_trt_dtype_to_torch_without_trt():
    from nemo_export.multimodal.run import trt_dtype_to_torch

    with patch("nemo_export.multimodal.run.HAVE_TRT", False), pytest.raises(UnavailableError):
        trt_dtype_to_torch(dtype=MagicMock())


@pytest.mark.skipif(not HAVE_TRT_LLM, reason="TensorRT-LLM is not installed")
@pytest.mark.run_only_on("GPU")
class TestMultimodalModelRunner(unittest.TestCase):
    @pytest.mark.run_only_on("GPU")
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.visual_engine_dir = os.path.join(self.temp_dir, "visual_engine")
        self.llm_engine_dir = os.path.join(self.temp_dir, "llm_engine")

        # Create directories
        os.makedirs(self.visual_engine_dir, exist_ok=True)
        os.makedirs(self.llm_engine_dir, exist_ok=True)
        os.makedirs(os.path.join(self.llm_engine_dir, "trtllm_engine"), exist_ok=True)

        # Mock config for visual engine
        self.mock_visual_config = {
            "builder_config": {
                "model_type": "neva",
                "precision": "float16",
                "num_frames": 4,
                "image_size": 224,
            }
        }

        # Create mock config file
        with open(os.path.join(self.visual_engine_dir, "config.json"), "w") as f:
            json.dump(self.mock_visual_config, f)

        # Mock nemo config
        self.mock_nemo_config = {
            "mm_cfg": {
                "vision_encoder": {
                    "from_pretrained": "test_model",
                    "crop_size": [224, 224],
                },
                "lita": {
                    "sample_frames": 8,
                    "visual_token_format": "im_vid_start_end",
                    "lita_video_arch": "temporal_spatial_pool",
                },
            },
            "data": {"num_frames": 4, "image_aspect_ratio": "pad"},
        }

        # Create mock nemo config file
        with open(os.path.join(self.visual_engine_dir, "nemo_config.yaml"), "w") as f:
            import yaml

            yaml.dump(self.mock_nemo_config, f)

        # Create mock tokenizer config
        tokenizer_config = {"model_type": "llama", "vocab_size": 32000}
        with open(os.path.join(self.llm_engine_dir, "tokenizer_config.json"), "w") as f:
            json.dump(tokenizer_config, f)

    @pytest.mark.run_only_on("GPU")
    def tearDown(self):
        # Clean up temporary directory
        import shutil

        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def _create_runner_with_mocks(self, model_type="vila"):
        """Helper method to create a runner with mocked dependencies"""
        from nemo_export.multimodal.run import MultimodalModelRunner

        # Update config for specific model type
        self.mock_visual_config["builder_config"]["model_type"] = model_type
        with open(os.path.join(self.visual_engine_dir, "config.json"), "w") as f:
            json.dump(self.mock_visual_config, f)

        with (
            patch("nemo_export.multimodal.run.torch.cuda.set_device"),
            patch("nemo_export.multimodal.run.torch.cuda.device_count", return_value=1),
            patch("nemo_export.multimodal.run.tensorrt_llm.mpi_rank", return_value=0),
            patch("nemo_export.multimodal.run.Session.from_serialized_engine") as mock_session,
            patch("nemo_export.multimodal.run.ModelRunner.from_dir") as mock_model_runner,
            patch("transformers.AutoTokenizer.from_pretrained") as mock_tokenizer,
            patch("transformers.SiglipImageProcessor.from_pretrained")
            if model_type in ["vila", "vita"]
            else patch("transformers.AutoProcessor.from_pretrained"),
        ):
            # Create mock visual encoder engine file
            engine_path = os.path.join(self.visual_engine_dir, "visual_encoder.engine")
            with open(engine_path, "wb") as f:
                f.write(b"mock_engine_data")

            # Setup mocks
            mock_tokenizer_instance = MagicMock()
            mock_tokenizer_instance.eos_token = "<eos>"
            mock_tokenizer.return_value = mock_tokenizer_instance

            mock_session_instance = MagicMock()
            mock_session.return_value = mock_session_instance

            mock_model_instance = MagicMock()
            mock_model_config = MagicMock()
            mock_model_config.vocab_size = 32000
            mock_model_config.hidden_size = 4096
            mock_model_config.dtype = "float16"
            mock_model_config.remove_input_padding = False
            mock_model_instance.session._model_config = mock_model_config
            mock_model_instance.session.mapping = MagicMock()
            mock_model_instance.session.mapping.tp_size = 1
            mock_model_runner.return_value = mock_model_instance

            return MultimodalModelRunner(
                visual_engine_dir=self.visual_engine_dir,
                llm_engine_dir=self.llm_engine_dir,
                modality="vision",
            )

    @pytest.mark.run_only_on("GPU")
    def test_setup_fake_prompts_vila_batch_size_1(self):
        """Test setup_fake_prompts_vila with VILA model and batch size 1"""
        runner = self._create_runner_with_mocks("vila")

        # Mock visual features for single batch
        visual_features = [
            torch.randn(256, 4096),  # First visual feature
            torch.randn(128, 4096),  # Second visual feature
        ]

        # Mock split input IDs
        split_input_ids = [
            torch.tensor([[1, 2, 3]]),  # Pre-prompt
            torch.tensor([[4, 5]]),  # Inter-prompt
            torch.tensor([[6, 7, 8]]),  # Post-prompt
        ]

        # Mock input lengths
        input_lengths = torch.tensor([10], dtype=torch.int32)

        # Mock ptuning_setup
        with patch.object(runner, "ptuning_setup") as mock_ptuning:
            mock_ptuning.return_value = ["mock_prompt_table", "mock_tasks", "mock_task_vocab_size"]

            input_ids, ptuning_args = runner.setup_fake_prompts_vila(
                batch_size=1,
                visual_features=visual_features,
                split_input_ids=split_input_ids,
                input_lengths=input_lengths,
            )

            # Verify input_ids shape and type
            self.assertEqual(input_ids.shape[0], 1)  # batch size
            self.assertEqual(input_ids.dtype, torch.int32)

            # Verify ptuning_setup was called
            mock_ptuning.assert_called_once()

            # Verify ptuning_args structure
            self.assertEqual(len(ptuning_args), 3)
            self.assertEqual(ptuning_args[0], "mock_prompt_table")
            self.assertEqual(ptuning_args[1], "mock_tasks")
            self.assertEqual(ptuning_args[2], "mock_task_vocab_size")

    @pytest.mark.run_only_on("GPU")
    def test_setup_fake_prompts_vila_batch_size_greater_than_1(self):
        """Test setup_fake_prompts_vila with VILA model and batch size > 1"""
        runner = self._create_runner_with_mocks("vila")

        # Mock visual features for multiple batches
        visual_features = [
            torch.randn(256, 4096),  # First visual feature
            torch.randn(128, 4096),  # Second visual feature
        ]

        # Mock split input IDs
        split_input_ids = [
            torch.tensor([[1, 2, 3]]),  # Pre-prompt
            torch.tensor([[6, 7, 8]]),  # Post-prompt
        ]

        # Mock input lengths
        input_lengths = torch.tensor([10, 10], dtype=torch.int32)

        # Mock ptuning_setup
        with patch.object(runner, "ptuning_setup") as mock_ptuning:
            mock_ptuning.return_value = ["mock_prompt_table", "mock_tasks", "mock_task_vocab_size"]

            input_ids, ptuning_args = runner.setup_fake_prompts_vila(
                batch_size=2,
                visual_features=visual_features,
                split_input_ids=split_input_ids,
                input_lengths=input_lengths,
            )

            # Verify input_ids shape and type
            self.assertEqual(input_ids.shape[0], 2)  # batch size
            self.assertEqual(input_ids.dtype, torch.int32)

            # Verify ptuning_setup was called
            mock_ptuning.assert_called_once()

    @pytest.mark.run_only_on("GPU")
    def test_setup_fake_prompts_vila_assertion_error(self):
        """Test setup_fake_prompts_vila assertion error when visual features exceed split_input_ids"""
        runner = self._create_runner_with_mocks("vila")

        # Mock visual features with more features than split_input_ids
        visual_features = [
            torch.randn(256, 4096),  # First visual feature
            torch.randn(128, 4096),  # Second visual feature
            torch.randn(64, 4096),  # Third visual feature (exceeds split_input_ids)
        ]

        # Mock split input IDs (only 2 elements)
        split_input_ids = [
            torch.tensor([[1, 2, 3]]),  # Pre-prompt
            torch.tensor([[6, 7, 8]]),  # Post-prompt
        ]

        # Mock input lengths
        input_lengths = torch.tensor([10], dtype=torch.int32)

        # Expect assertion error
        with self.assertRaises(AssertionError) as context:
            runner.setup_fake_prompts_vila(
                batch_size=1,
                visual_features=visual_features,
                split_input_ids=split_input_ids,
                input_lengths=input_lengths,
            )

        self.assertIn("Unexpected number of visual features", str(context.exception))

    @pytest.mark.run_only_on("GPU")
    def test_setup_fake_prompts_vila_fake_prompt_counter_increments(self):
        """Test that fake_prompt_counter increments correctly"""
        runner = self._create_runner_with_mocks("vila")

        # Mock visual features with known sizes
        visual_features = [
            torch.randn(100, 4096),  # First visual feature (100 tokens)
            torch.randn(50, 4096),  # Second visual feature (50 tokens)
        ]

        # Mock split input IDs
        split_input_ids = [
            torch.tensor([[1, 2, 3]]),  # Pre-prompt
            torch.tensor([[4, 5]]),  # Inter-prompt
            torch.tensor([[6, 7, 8]]),  # Post-prompt
        ]

        # Mock input lengths
        input_lengths = torch.tensor([10], dtype=torch.int32)

        # Mock ptuning_setup
        with patch.object(runner, "ptuning_setup") as mock_ptuning:
            mock_ptuning.return_value = ["mock_prompt_table", "mock_tasks", "mock_task_vocab_size"]

            input_ids, ptuning_args = runner.setup_fake_prompts_vila(
                batch_size=1,
                visual_features=visual_features,
                split_input_ids=split_input_ids,
                input_lengths=input_lengths,
            )

            # Verify that fake prompt IDs start from vocab_size
            # and increment by the size of each visual feature
            vocab_size = runner.model_config.vocab_size

            # Check that input_ids contains the expected fake prompt IDs
            # First fake prompt should be [vocab_size, vocab_size+1, ..., vocab_size+99]
            # Second fake prompt should be [vocab_size+100, vocab_size+101, ..., vocab_size+149]

            # Since we can't easily extract the exact fake prompt IDs from the concatenated result,
            # we'll verify that the input_ids contains values >= vocab_size
            unique_ids = torch.unique(input_ids)
            high_ids = unique_ids[unique_ids >= vocab_size]

            # Should have 150 unique high IDs (100 + 50)
            self.assertEqual(len(high_ids), 150)

    @pytest.mark.run_only_on("GPU")
    def test_setup_fake_prompts_vila_empty_split_input_ids(self):
        """Test setup_fake_prompts_vila with minimal split_input_ids"""
        runner = self._create_runner_with_mocks("vila")

        # Mock visual features
        visual_features = [
            torch.randn(256, 4096),  # Single visual feature
        ]

        # Mock split input IDs with only pre-prompt
        split_input_ids = [
            torch.tensor([[1, 2, 3]]),  # Pre-prompt only
        ]

        # Mock input lengths
        input_lengths = torch.tensor([10], dtype=torch.int32)

        # Mock ptuning_setup
        with patch.object(runner, "ptuning_setup") as mock_ptuning:
            mock_ptuning.return_value = ["mock_prompt_table", "mock_tasks", "mock_task_vocab_size"]

            input_ids, ptuning_args = runner.setup_fake_prompts_vila(
                batch_size=1,
                visual_features=visual_features,
                split_input_ids=split_input_ids,
                input_lengths=input_lengths,
            )

            # Should not raise an error and should produce valid output
            self.assertEqual(input_ids.shape[0], 1)  # batch size
            self.assertEqual(input_ids.dtype, torch.int32)

            # Verify ptuning_setup was called
            mock_ptuning.assert_called_once()

    @pytest.mark.run_only_on("GPU")
    @patch("nemo_export.multimodal.run.torch.cuda.set_device")
    @patch("nemo_export.multimodal.run.torch.cuda.device_count", return_value=1)
    @patch("nemo_export.multimodal.run.tensorrt_llm.mpi_rank", return_value=0)
    @patch("nemo_export.multimodal.run.ModelRunner.from_dir")
    @patch("nemo_export.multimodal.run.Session.from_serialized_engine")
    @patch("transformers.AutoTokenizer.from_pretrained")
    def test_init_vision_modality(
        self,
        mock_tokenizer,
        mock_session,
        mock_model_runner,
        mock_mpi_rank,
        mock_device_count,
        mock_set_device,
    ):
        """Test initialization with vision modality"""
        from nemo_export.multimodal.run import MultimodalModelRunner

        # Setup mocks
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.eos_token = "<eos>"
        mock_tokenizer.return_value = mock_tokenizer_instance

        mock_session_instance = MagicMock()
        mock_session.return_value = mock_session_instance

        mock_model_instance = MagicMock()
        mock_model_instance.session._model_config = MagicMock()
        mock_model_instance.session.mapping = MagicMock()
        mock_model_runner.return_value = mock_model_instance

        # Create mock visual encoder engine file
        engine_path = os.path.join(self.visual_engine_dir, "visual_encoder.engine")
        with open(engine_path, "wb") as f:
            f.write(b"mock_engine_data")

        runner = MultimodalModelRunner(
            visual_engine_dir=self.visual_engine_dir,
            llm_engine_dir=self.llm_engine_dir,
            modality="vision",
        )

        self.assertEqual(runner.modality, "vision")
        self.assertEqual(runner.model_type, "neva")
        self.assertEqual(runner.vision_precision, "float16")
        mock_tokenizer.assert_called_once()
        mock_session.assert_called_once()
        mock_model_runner.assert_called_once()

    @pytest.mark.run_only_on("GPU")
    @patch("nemo_export.multimodal.run.torch.cuda.set_device")
    @patch("nemo_export.multimodal.run.torch.cuda.device_count", return_value=1)
    @patch("nemo_export.multimodal.run.tensorrt_llm.mpi_rank", return_value=0)
    @patch("nemo_export.multimodal.run.ModelRunner.from_dir")
    @patch("nemo_export.multimodal.run.Session.from_serialized_engine")
    @patch("transformers.AutoTokenizer.from_pretrained")
    @patch("transformers.SiglipImageProcessor.from_pretrained")
    def test_init_vila_model(
        self,
        mock_processor,
        mock_tokenizer,
        mock_session,
        mock_model_runner,
        mock_mpi_rank,
        mock_device_count,
        mock_set_device,
    ):
        """Test initialization with VILA model type"""
        from nemo_export.multimodal.run import MultimodalModelRunner

        # Update config for VILA
        self.mock_visual_config["builder_config"]["model_type"] = "vila"
        with open(os.path.join(self.visual_engine_dir, "config.json"), "w") as f:
            json.dump(self.mock_visual_config, f)

        # Setup mocks
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.eos_token = "<eos>"
        mock_tokenizer.return_value = mock_tokenizer_instance

        mock_processor_instance = MagicMock()
        mock_processor.return_value = mock_processor_instance

        mock_session_instance = MagicMock()
        mock_session.return_value = mock_session_instance

        mock_model_instance = MagicMock()
        mock_model_instance.session._model_config = MagicMock()
        mock_model_instance.session.mapping = MagicMock()
        mock_model_runner.return_value = mock_model_instance

        # Create mock visual encoder engine file
        engine_path = os.path.join(self.visual_engine_dir, "visual_encoder.engine")
        with open(engine_path, "wb") as f:
            f.write(b"mock_engine_data")

        runner = MultimodalModelRunner(
            visual_engine_dir=self.visual_engine_dir,
            llm_engine_dir=self.llm_engine_dir,
            modality="vision",
        )

        self.assertEqual(runner.model_type, "vila")
        mock_processor.assert_called_once()

    @pytest.mark.run_only_on("GPU")
    @patch("nemo_export.multimodal.run.torch.cuda.set_device")
    @patch("nemo_export.multimodal.run.torch.cuda.device_count", return_value=1)
    @patch("nemo_export.multimodal.run.tensorrt_llm.mpi_rank", return_value=0)
    @patch("nemo_export.multimodal.run.ModelRunner.from_dir")
    @patch("nemo_export.multimodal.run.Session.from_serialized_engine")
    @patch("sentencepiece.SentencePieceProcessor")
    def test_init_sentencepiece_tokenizer(
        self,
        mock_sp,
        mock_session,
        mock_model_runner,
        mock_mpi_rank,
        mock_device_count,
        mock_set_device,
    ):
        """Test initialization with SentencePiece tokenizer"""
        from nemo_export.multimodal.run import MultimodalModelRunner

        # Remove tokenizer_config.json to trigger SentencePiece path
        os.remove(os.path.join(self.llm_engine_dir, "tokenizer_config.json"))

        # Create mock tokenizer.model file
        tokenizer_model_path = os.path.join(self.llm_engine_dir, "tokenizer.model")
        with open(tokenizer_model_path, "wb") as f:
            f.write(b"mock_tokenizer_model")

        # Setup mocks
        mock_sp_instance = MagicMock()
        mock_sp_instance.eos_id.return_value = 2
        mock_sp_instance.bos_id.return_value = 1
        mock_sp_instance.pad_id.return_value = 0
        mock_sp_instance.piece_to_id.return_value = 100
        mock_sp.return_value = mock_sp_instance

        mock_session_instance = MagicMock()
        mock_session.return_value = mock_session_instance

        mock_model_instance = MagicMock()
        mock_model_instance.session._model_config = MagicMock()
        mock_model_instance.session.mapping = MagicMock()
        mock_model_runner.return_value = mock_model_instance

        # Create mock visual encoder engine file
        engine_path = os.path.join(self.visual_engine_dir, "visual_encoder.engine")
        with open(engine_path, "wb") as f:
            f.write(b"mock_engine_data")

        runner = MultimodalModelRunner(
            visual_engine_dir=self.visual_engine_dir,
            llm_engine_dir=self.llm_engine_dir,
            modality="vision",
        )

        self.assertEqual(runner.tokenizer.eos_token_id, 2)
        self.assertEqual(runner.tokenizer.bos_token_id, 1)
        self.assertEqual(runner.tokenizer.pad_token_id, 0)

    @pytest.mark.run_only_on("GPU")
    def test_trt_dtype_to_torch(self):
        """Test TensorRT dtype to torch dtype conversion"""
        import tensorrt as trt

        from nemo_export.multimodal.run import trt_dtype_to_torch

        self.assertEqual(trt_dtype_to_torch(trt.float16), torch.float16)
        self.assertEqual(trt_dtype_to_torch(trt.float32), torch.float32)
        self.assertEqual(trt_dtype_to_torch(trt.int32), torch.int32)
        self.assertEqual(trt_dtype_to_torch(trt.bfloat16), torch.bfloat16)

        with self.assertRaises(TypeError):
            trt_dtype_to_torch(trt.int8)

    @pytest.mark.run_only_on("GPU")
    @patch("nemo_export.multimodal.run.tensorrt_llm._utils.str_dtype_to_torch")
    @patch("decord.VideoReader")
    @patch("transformers.CLIPImageProcessor.from_pretrained")
    @patch("PIL.Image.fromarray")
    def test_video_preprocess_string_path(
        self,
        mock_image_fromarray,
        mock_clip_processor,
        mock_video_reader,
        mock_str_dtype_to_torch,
    ):
        """Test video preprocessing with string path"""
        from nemo_export.multimodal.run import MultimodalModelRunner

        # Mock the str_dtype_to_torch utility function
        mock_str_dtype_to_torch.return_value = torch.float16

        # Setup mock frames and video reader
        mock_frame = MagicMock()
        mock_frame.asnumpy.return_value = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

        mock_vr = MagicMock()
        mock_vr.__len__.return_value = 10  # Total frames in video
        mock_vr.__getitem__.return_value = mock_frame  # For indexed access vr[idx]
        mock_video_reader.return_value = mock_vr

        # Mock PIL.Image.fromarray to return mock images
        mock_pil_image = MagicMock()
        mock_pil_image.convert.return_value = mock_pil_image  # RGB conversion
        mock_image_fromarray.return_value = mock_pil_image

        # Mock CLIP processor
        mock_processor_instance = MagicMock()
        # Create mock tensor with correct shape [num_frames, 3, H, W]
        mock_processed_frames = torch.randn(4, 3, 224, 224, dtype=torch.float32)
        mock_processor_instance.preprocess.return_value = {"pixel_values": mock_processed_frames}
        mock_clip_processor.return_value = mock_processor_instance

        # Create runner instance with required attributes
        runner = MagicMock()
        runner.num_frames = 4  # Want 4 frames from video
        runner.vision_precision = "float16"

        # Call the method directly
        result = MultimodalModelRunner.video_preprocess(runner, "test_video.mp4")

        # Verify the result shape
        self.assertEqual(result.shape[0], 1)  # batch dimension added by unsqueeze(0)
        self.assertEqual(result.shape[1], 4)  # num_frames
        self.assertEqual(result.shape[2], 3)  # channels
        self.assertEqual(result.shape[3], 224)  # height
        self.assertEqual(result.shape[4], 224)  # width
        self.assertEqual(result.dtype, torch.float16)  # converted dtype

        # Verify VideoReader was called correctly
        mock_video_reader.assert_called_once_with("test_video.mp4")

        # Verify CLIPImageProcessor was called with correct parameters
        mock_clip_processor.assert_called_once_with("openai/clip-vit-large-patch14", torch_dtype=torch.bfloat16)

        # Verify frames were processed
        mock_processor_instance.preprocess.assert_called_once()
        processed_frames_call = mock_processor_instance.preprocess.call_args[0][0]
        self.assertEqual(len(processed_frames_call), 4)  # 4 PIL images

        # Verify dtype conversion was called
        mock_str_dtype_to_torch.assert_called_once_with("float16")

    @pytest.mark.run_only_on("GPU")
    @patch("transformers.CLIPImageProcessor.from_pretrained")
    def test_video_preprocess_numpy_array(self, mock_processor):
        """Test video preprocessing with numpy array input"""
        from nemo_export.multimodal.run import MultimodalModelRunner

        # Setup mock processor
        mock_processor_instance = MagicMock()
        mock_processor_instance.preprocess.return_value = {"pixel_values": torch.randn(4, 3, 224, 224)}
        mock_processor.return_value = mock_processor_instance

        # Create mock video data
        video_data = np.random.randint(0, 255, (10, 224, 224, 3), dtype=np.uint8)

        # Create a minimal runner instance
        runner = MagicMock()
        runner.num_frames = 4
        runner.vision_precision = "float16"

        result = MultimodalModelRunner.video_preprocess(runner, video_data)

        self.assertEqual(result.shape[0], 1)  # batch dimension
        self.assertEqual(result.shape[1], 4)  # num_frames

    @pytest.mark.run_only_on("GPU")
    def test_insert_tokens_by_index(self):
        """Test token insertion by index"""
        from nemo_export.multimodal.run import MultimodalModelRunner

        # Create a minimal runner instance with mock tokenizer
        runner = MagicMock()
        runner.tokenizer = MagicMock()
        runner.tokenizer.im_start_id = 10
        runner.tokenizer.im_end_id = 11
        runner.tokenizer.vid_start_id = 12
        runner.tokenizer.vid_end_id = 13

        # Test input with image tokens (represented as 0)
        input_ids = torch.tensor([[1, 2, 0, 3, 4]])  # 0 represents image token
        num_frames = 2

        result = MultimodalModelRunner.insert_tokens_by_index(runner, input_ids, num_frames)

        self.assertIsInstance(result, torch.Tensor)
        self.assertEqual(result.shape[0], 1)  # batch dimension maintained

    @pytest.mark.run_only_on("GPU")
    def test_tokenizer_image_token(self):
        """Test tokenizer image token processing"""
        from nemo_export.multimodal.run import MultimodalModelRunner

        # Mock tokenizer
        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value.input_ids = [1, 2, 3]
        mock_tokenizer.bos_token_id = 1

        prompt = "Hello <image> world"
        batch_size = 2

        result = MultimodalModelRunner.tokenizer_image_token(batch_size, prompt, mock_tokenizer)

        self.assertEqual(result.shape[0], batch_size)

    @pytest.mark.run_only_on("GPU")
    def test_split_prompt_by_images(self):
        """Test splitting prompt by image tokens"""
        from nemo_export.multimodal.run import MultimodalModelRunner

        # Create test tensor with zeros representing image tokens
        test_tensor = torch.tensor([[1, 2, 0, 3, 4, 0, 5, 6]])

        runner = MagicMock()
        result = MultimodalModelRunner.split_prompt_by_images(runner, test_tensor)

        self.assertEqual(len(result), 1)  # One batch
        self.assertEqual(len(result[0]), 3)  # Three segments split by two zeros

    @pytest.mark.run_only_on("GPU")
    def test_expand2square_pt(self):
        """Test expanding images to square"""
        from nemo_export.multimodal.run import MultimodalModelRunner

        runner = MagicMock()

        # Test rectangular image (wider than tall)
        images = torch.randn(2, 3, 100, 200)  # batch=2, channels=3, height=100, width=200
        background_color = [0.5, 0.5, 0.5]

        result = MultimodalModelRunner.expand2square_pt(runner, images, background_color)

        self.assertEqual(result.shape[-1], result.shape[-2])  # Should be square
        self.assertEqual(result.shape[-1], 200)  # Should match the larger dimension

    @pytest.mark.run_only_on("GPU")
    @patch("decord.VideoReader")
    def test_load_video_string_path(self, mock_video_reader):
        """Test loading video from string path"""
        from nemo_export.multimodal.run import MultimodalModelRunner

        # Setup mock video reader
        mock_frame = MagicMock()
        mock_frame.asnumpy.return_value = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

        mock_vr = MagicMock()
        mock_vr.__len__.return_value = 10
        mock_vr.get_batch.return_value = [mock_frame] * 4
        mock_video_reader.return_value = mock_vr

        # Mock processor
        mock_processor = MagicMock()
        mock_processor.preprocess.return_value = {"pixel_values": torch.randn(4, 3, 224, 224)}
        mock_processor.image_mean = [0.5, 0.5, 0.5]

        # Mock config
        mock_config = {"data": {"image_aspect_ratio": "pad"}}

        runner = MagicMock()
        MultimodalModelRunner.load_video(runner, mock_config, "test_video.mp4", mock_processor, num_frames=4)

        mock_video_reader.assert_called_once()

    @pytest.mark.run_only_on("GPU")
    def test_load_video_numpy_array(self):
        """Test loading video from numpy array"""
        from nemo_export.multimodal.run import MultimodalModelRunner

        # Create test video data
        video_data = np.random.randint(0, 255, (10, 224, 224, 3), dtype=np.uint8)

        # Mock processor
        mock_processor = MagicMock()
        mock_processor.preprocess.return_value = {"pixel_values": torch.randn(10, 3, 224, 224)}
        mock_processor.image_mean = [0.5, 0.5, 0.5]

        # Mock config
        mock_config = {"data": {"image_aspect_ratio": "pad"}}

        # Create a mock runner and assign the real load_video method to it
        runner = MagicMock()
        runner.load_video = MultimodalModelRunner.load_video.__get__(runner, MultimodalModelRunner)
        runner.preprocess_frames.return_value = torch.randn(10, 3, 224, 224)

        # Call the method - now it will use the real implementation
        runner.load_video(mock_config, video_data, mock_processor)

        # Should call preprocess_frames on the instance
        runner.preprocess_frames.assert_called_once()

    @pytest.mark.run_only_on("GPU")
    def test_get_num_sample_frames(self):
        """Test getting number of sample frames"""
        from nemo_export.multimodal.run import MultimodalModelRunner

        # Test with im_vid_start_end format
        config1 = {
            "mm_cfg": {"lita": {"visual_token_format": "im_vid_start_end"}},
            "data": {"num_frames": 8},
        }

        runner = MagicMock()

        # When vid_len > max_frames, it subsamples
        # vid_len=10, max_frames=8 -> subsample=ceil(10/8)=2 -> round(10/2)=5
        result1 = MultimodalModelRunner.get_num_sample_frames(runner, config1, vid_len=10)
        self.assertEqual(result1, 5)

        # When vid_len <= max_frames, it returns vid_len
        result2 = MultimodalModelRunner.get_num_sample_frames(runner, config1, vid_len=4)
        self.assertEqual(result2, 4)

        # When vid_len == max_frames, it returns vid_len
        result3 = MultimodalModelRunner.get_num_sample_frames(runner, config1, vid_len=8)
        self.assertEqual(result3, 8)

        # Test with other format (uses sample_frames)
        config2 = {"mm_cfg": {"lita": {"sample_frames": 6}}}

        result4 = MultimodalModelRunner.get_num_sample_frames(runner, config2, vid_len=10)
        self.assertEqual(result4, 6)

    @pytest.mark.run_only_on("GPU")
    def test_process_image_string_path(self):
        """Test processing image from string path"""
        from nemo_export.multimodal.run import MultimodalModelRunner

        # Create a test image
        test_image = Image.new("RGB", (256, 256), color="red")

        # Mock image processor
        mock_processor = MagicMock()
        mock_processor.preprocess.return_value = {"pixel_values": [torch.randn(3, 224, 224)]}
        mock_processor.image_mean = [0.5, 0.5, 0.5]

        # Mock config
        mock_config = {
            "mm_cfg": {"vision_encoder": {"crop_size": [224, 224]}},
            "data": {"image_aspect_ratio": "pad"},
        }

        runner = MagicMock()

        with patch("PIL.Image.open", return_value=test_image):
            MultimodalModelRunner.process_image(runner, "test_image.jpg", mock_processor, mock_config, None)

        mock_processor.preprocess.assert_called_once()

    @pytest.mark.run_only_on("GPU")
    @patch("nemo_export.multimodal.run.torch.cuda.set_device")
    @patch("nemo_export.multimodal.run.torch.cuda.device_count", return_value=1)
    @patch("nemo_export.multimodal.run.tensorrt_llm.mpi_rank", return_value=0)
    @patch("nemo_export.multimodal.run.ModelRunner.from_dir")
    @patch("nemo_export.multimodal.run.Session.from_serialized_engine")
    @patch("transformers.AutoTokenizer.from_pretrained")
    def test_setup_inputs_neva(
        self,
        mock_tokenizer,
        mock_session,
        mock_model_runner,
        mock_mpi_rank,
        mock_device_count,
        mock_set_device,
    ):
        """Test setup inputs for NEVA model"""
        from nemo_export.multimodal.run import MultimodalModelRunner

        # Setup mocks
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.eos_token = "<eos>"
        mock_tokenizer.return_value = mock_tokenizer_instance

        mock_session_instance = MagicMock()
        mock_session.return_value = mock_session_instance

        mock_model_instance = MagicMock()
        mock_model_instance.session._model_config = MagicMock()
        mock_model_instance.session.mapping = MagicMock()
        mock_model_runner.return_value = mock_model_instance

        # Create mock visual encoder engine file
        engine_path = os.path.join(self.visual_engine_dir, "visual_encoder.engine")
        with open(engine_path, "wb") as f:
            f.write(b"mock_engine_data")

        runner = MultimodalModelRunner(
            visual_engine_dir=self.visual_engine_dir,
            llm_engine_dir=self.llm_engine_dir,
            modality="vision",
        )

        # Create test image
        test_image = Image.new("RGB", (256, 256), color="red")

        result = runner.setup_inputs(input_text="What is in this image?", raw_image=test_image, batch_size=1)

        self.assertEqual(len(result), 6)  # Should return 6 elements
        (
            input_text,
            pre_prompt,
            post_prompt,
            image,
            decoder_input_ids,
            attention_mask,
        ) = result

        self.assertIsInstance(pre_prompt, list)
        self.assertIsInstance(post_prompt, list)
        self.assertIsInstance(image, torch.Tensor)

    @pytest.mark.run_only_on("GPU")
    def test_preprocess_lita_visual_im_vid_start_end(self):
        """Test LITA visual preprocessing with im_vid_start_end format"""
        from nemo_export.multimodal.run import MultimodalModelRunner

        visual_features = torch.randn(1, 8, 256, 768)  # batch, time, spatial, dim

        config = {"mm_cfg": {"lita": {"visual_token_format": "im_vid_start_end", "sample_frames": 4}}}

        runner = MagicMock()
        im_features, vid_features, num_frames = MultimodalModelRunner.preprocess_lita_visual(
            runner, visual_features, config
        )

        self.assertEqual(num_frames, 4)
        self.assertEqual(im_features.shape[1], 4)  # Should sample 4 frames
        self.assertEqual(vid_features.shape[1], 8)  # Should have 8 temporal features

    @pytest.mark.run_only_on("GPU")
    def test_preprocess_lita_visual_temporal_spatial_pool(self):
        """Test LITA visual preprocessing with temporal spatial pooling"""
        from nemo_export.multimodal.run import MultimodalModelRunner

        visual_features = torch.randn(1, 8, 256, 768)  # batch, time, spatial, dim

        config = {"mm_cfg": {"lita": {"lita_video_arch": "temporal_spatial_pool"}}}

        runner = MagicMock()
        t_tokens, s_tokens, pool_size_sq = MultimodalModelRunner.preprocess_lita_visual(runner, visual_features, config)

        self.assertEqual(pool_size_sq, 4)  # pool_size^2 = 2^2 = 4
        self.assertEqual(t_tokens.shape[1], 8)  # Should have 8 temporal tokens

    @pytest.mark.run_only_on("GPU")
    def test_load_test_media_neva(self):
        """Test loading test media for NEVA model"""
        from nemo_export.multimodal.run import MultimodalModelRunner

        runner = MagicMock()
        runner.model_type = "neva"

        test_image = Image.new("RGB", (256, 256), color="red")

        with patch("PIL.Image.open", return_value=test_image):
            result = MultimodalModelRunner.load_test_media(runner, "test_image.jpg")

        self.assertIsInstance(result, Image.Image)

    @pytest.mark.run_only_on("GPU")
    def test_load_test_media_video_neva(self):
        """Test loading test media for video NEVA model"""
        from nemo_export.multimodal.run import MultimodalModelRunner

        runner = MagicMock()
        runner.model_type = "video-neva"

        test_video_path = "test_video.mp4"
        result = MultimodalModelRunner.load_test_media(runner, test_video_path)

        self.assertEqual(result, test_video_path)

    @pytest.mark.run_only_on("GPU")
    def test_load_test_media_invalid_model(self):
        """Test loading test media with invalid model type"""
        from nemo_export.multimodal.run import MultimodalModelRunner

        runner = MagicMock()
        runner.model_type = "invalid_model"

        with self.assertRaises(RuntimeError):
            MultimodalModelRunner.load_test_media(runner, "test_media")

    @pytest.mark.run_only_on("GPU")
    @patch("nemo_export.multimodal.run.torch.cuda.set_device")
    @patch("nemo_export.multimodal.run.torch.cuda.device_count", return_value=1)
    @patch("nemo_export.multimodal.run.tensorrt_llm.mpi_rank", return_value=0)
    @patch("nemo_export.multimodal.run.ModelRunner.from_dir")
    @patch("nemo_export.multimodal.run.Session.from_serialized_engine")
    @patch("transformers.AutoTokenizer.from_pretrained")
    def test_print_result(
        self,
        mock_tokenizer,
        mock_session,
        mock_model_runner,
        mock_mpi_rank,
        mock_device_count,
        mock_set_device,
    ):
        """Test print result functionality"""
        from nemo_export.multimodal.run import MultimodalModelRunner

        # Setup mocks
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.eos_token = "<eos>"
        mock_tokenizer_instance.return_value = {"input_ids": [1, 2, 3, 4, 5]}
        mock_tokenizer.return_value = mock_tokenizer_instance

        mock_session_instance = MagicMock()
        mock_session.return_value = mock_session_instance

        mock_model_instance = MagicMock()
        mock_model_instance.session._model_config = MagicMock()
        mock_model_instance.session.mapping = MagicMock()
        mock_model_runner.return_value = mock_model_instance

        # Create mock visual encoder engine file
        engine_path = os.path.join(self.visual_engine_dir, "visual_encoder.engine")
        with open(engine_path, "wb") as f:
            f.write(b"mock_engine_data")

        runner = MultimodalModelRunner(
            visual_engine_dir=self.visual_engine_dir,
            llm_engine_dir=self.llm_engine_dir,
            modality="vision",
        )

        # Test with profiling enabled
        with patch("nemo_export.multimodal.run.profiler.elapsed_time_in_sec", return_value=0.1):
            runner.print_result(
                input_text="Test input",
                output_text=[["Test output"]],
                batch_size=1,
                num_beams=1,
                run_profiling=True,
                check_accuracy=False,
            )

    @pytest.mark.run_only_on("GPU")
    @patch("nemo_export.multimodal.run.torch.cuda.set_device")
    @patch("nemo_export.multimodal.run.torch.cuda.device_count", return_value=1)
    @patch("nemo_export.multimodal.run.tensorrt_llm.mpi_rank", return_value=0)
    @patch("nemo_export.multimodal.run.ModelRunner.from_dir")
    @patch("nemo_export.multimodal.run.Session.from_serialized_engine")
    @patch("transformers.AutoTokenizer.from_pretrained")
    def test_ptuning_setup(
        self,
        mock_tokenizer,
        mock_session,
        mock_model_runner,
        mock_mpi_rank,
        mock_device_count,
        mock_set_device,
    ):
        """Test P-tuning setup"""
        from nemo_export.multimodal.run import MultimodalModelRunner

        # Setup mocks
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_instance.eos_token = "<eos>"
        mock_tokenizer.return_value = mock_tokenizer_instance

        mock_session_instance = MagicMock()
        mock_session.return_value = mock_session_instance

        mock_model_config = MagicMock()
        mock_model_config.hidden_size = 4096
        mock_model_config.dtype = "float16"
        mock_model_config.remove_input_padding = False

        mock_runtime_mapping = MagicMock()
        mock_runtime_mapping.tp_size = 1

        mock_model_instance = MagicMock()
        mock_model_instance.session._model_config = mock_model_config
        mock_model_instance.session.mapping = mock_runtime_mapping
        mock_model_runner.return_value = mock_model_instance

        # Create mock visual encoder engine file
        engine_path = os.path.join(self.visual_engine_dir, "visual_encoder.engine")
        with open(engine_path, "wb") as f:
            f.write(b"mock_engine_data")

        runner = MultimodalModelRunner(
            visual_engine_dir=self.visual_engine_dir,
            llm_engine_dir=self.llm_engine_dir,
            modality="vision",
        )

        # Test P-tuning setup
        prompt_table = torch.randn(1, 256, 4096)
        input_ids = torch.tensor([[1, 2, 3, 4, 5]])
        input_lengths = torch.tensor([5])

        result = runner.ptuning_setup(prompt_table, input_ids, input_lengths)

        self.assertEqual(len(result), 3)  # Should return 3 elements
        self.assertIsInstance(result[0], torch.Tensor)  # prompt_table
        self.assertIsInstance(result[1], torch.Tensor)  # tasks
        self.assertIsInstance(result[2], torch.Tensor)  # task_vocab_size

    @pytest.mark.run_only_on("GPU")
    def test_preprocess_lita_visual_invalid_format(self):
        """Test LITA visual preprocessing with invalid format raises ValueError"""
        from nemo_export.multimodal.run import MultimodalModelRunner

        visual_features = torch.randn(1, 8, 256, 768)

        config = {"mm_cfg": {"lita": {"visual_token_format": "invalid_format"}}}

        runner = MagicMock()

        with self.assertRaises(ValueError):
            MultimodalModelRunner.preprocess_lita_visual(runner, visual_features, config)

    @pytest.mark.run_only_on("GPU")
    def test_preprocess_lita_model_batch_error(self):
        """Test that LITA/VITA models raise error for batch size > 1"""
        from nemo_export.multimodal.run import MultimodalModelRunner

        runner = MagicMock()
        runner.model_type = "lita"
        runner.modality = "vision"

        # Mock the get_visual_features method - called in the for loop for each image
        runner.get_visual_features = MagicMock(return_value=(torch.randn(8, 256, 768), torch.ones(8, 256)))

        # Mock the preprocess_lita_visual method - processes visual features
        im_tokens = torch.randn(1, 4, 256, 768)  # image tokens
        vid_tokens = torch.randn(1, 8, 768)  # video tokens
        num_sample_frames = 4
        runner.preprocess_lita_visual = MagicMock(return_value=(im_tokens, vid_tokens, num_sample_frames))

        # Mock tokenizer_image_token - processes text prompts
        runner.tokenizer_image_token = MagicMock(return_value=torch.tensor([[1, 0, 2, 3]]))

        # Mock insert_tokens_by_index - inserts special tokens for LITA
        runner.insert_tokens_by_index = MagicMock(return_value=torch.tensor([[1, 10, 0, 11, 2, 3]]))

        # Mock split_prompt_by_images - splits prompt by image tokens
        runner.split_prompt_by_images = MagicMock(return_value=[[torch.tensor([[1]]), torch.tensor([[2, 3]])]])

        # Mock setup_fake_prompts_vila - final setup (shouldn't be called due to ValueError)
        runner.setup_fake_prompts_vila = MagicMock()

        # Mock nemo_config
        nemo_config = {"mm_cfg": {"lita": {"visual_token_format": "im_vid_start_end", "sample_frames": 4}}}
        runner.nemo_config = nemo_config

        # Test inputs
        pre_prompt = ["test prompt"]
        post_prompt = ["test post"]

        # Create a mock image tensor with device attribute
        mock_image_tensor = torch.randn(1, 4, 3, 224, 224)

        # Create a mock list that also has a device attribute (needed for image.device access)
        class MockImageList(list):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.device = torch.device("cuda:0")

        image = MockImageList([mock_image_tensor])
        attention_mask = None
        batch_size = 2  # This should trigger the ValueError

        with self.assertRaises(ValueError) as context:
            MultimodalModelRunner.preprocess(
                runner,
                warmup=False,
                pre_prompt=pre_prompt,
                post_prompt=post_prompt,
                image=image,
                attention_mask=attention_mask,
                batch_size=batch_size,
            )

        self.assertIn(
            "Batch size greater than 1 is not supported for LITA and VITA models",
            str(context.exception),
        )

        # Verify that the methods were called in the expected order before the ValueError
        runner.get_visual_features.assert_called_once_with(mock_image_tensor, attention_mask)
        runner.preprocess_lita_visual.assert_called_once()
        runner.tokenizer_image_token.assert_called_once()
        runner.insert_tokens_by_index.assert_called_once()
        runner.split_prompt_by_images.assert_called_once()
        # setup_fake_prompts_vila should NOT be called due to the ValueError
        runner.setup_fake_prompts_vila.assert_not_called()

    @pytest.mark.run_only_on("GPU")
    @patch("nemo_export.multimodal.run.logger")
    def test_print_result_accuracy_check_with_robot(self, mock_logger):
        """Test print result accuracy checking for robot detection"""
        from nemo_export.multimodal.run import MultimodalModelRunner

        runner = MagicMock()
        runner.model_type = "neva"  # Set model type to avoid logger issues
        runner.tokenizer.return_value = {"input_ids": [1, 2, 3, 4, 5]}
        # Bind the real print_result method to the mock runner
        runner.print_result = MultimodalModelRunner.print_result.__get__(runner, MultimodalModelRunner)

        # Should pass with "robot" in output - use batch_size=2 so the for loop runs
        output_text = [
            ["I can see a robot in the image"],
            ["I can see a robot in the image"],
        ]

        # Should not raise any exception
        runner.print_result(
            input_text="Test input",
            output_text=output_text,
            batch_size=2,
            num_beams=1,
            run_profiling=False,
            check_accuracy=True,
        )

        # Should fail without "robot" in output - use batch_size=2 so the for loop runs
        output_text_fail = [
            ["I can see a car in the image"],
            ["I can see a car in the image"],
        ]

        with self.assertRaises(AssertionError):
            runner.print_result(
                input_text="Test input",
                output_text=output_text_fail,
                batch_size=2,
                num_beams=1,
                run_profiling=False,
                check_accuracy=True,
            )

    def test_init_without_trt_llm(self):
        from nemo_export.multimodal.run import MultimodalModelRunner

        with patch("nemo_export.multimodal.run.HAVE_TRT_LLM", False), pytest.raises(UnavailableError):
            MultimodalModelRunner(visual_engine_dir="", llm_engine_dir="")

    def test_init_llm_without_trt_llm(self):
        from nemo_export.multimodal.run import MultimodalModelRunner

        with patch.object(MultimodalModelRunner, "__init__", lambda self: None):
            with patch("nemo_export.multimodal.run.HAVE_TRT_LLM", False), pytest.raises(UnavailableError):
                MultimodalModelRunner().init_llm(llm_engine_dir="")

    def test_generate_without_trt_llm(self):
        from nemo_export.multimodal.run import MultimodalModelRunner

        with patch.object(MultimodalModelRunner, "__init__", lambda self: None):
            with patch("nemo_export.multimodal.run.HAVE_TRT_LLM", False), pytest.raises(UnavailableError):
                MultimodalModelRunner().generate(
                    pre_prompt="",
                    post_prompt="",
                    image="",
                    decoder_input_ids="",
                    max_new_tokens="",
                    attention_mask="",
                    warmup="",
                    batch_size="",
                    top_k="",
                    top_p="",
                    temperature="",
                    repetition_penalty="",
                    num_beams="",
                )

    def test_get_visual_features_without_trt_llm(self):
        from nemo_export.multimodal.run import MultimodalModelRunner

        with patch.object(MultimodalModelRunner, "__init__", lambda self: None):
            with patch("nemo_export.multimodal.run.HAVE_TRT_LLM", False), pytest.raises(UnavailableError):
                MultimodalModelRunner().get_visual_features(image="", attention_mask="")

    def test_video_preprocess_without_pil_str(self):
        from nemo_export.multimodal.run import MultimodalModelRunner

        with patch.object(MultimodalModelRunner, "__init__", lambda self: None):
            with patch("nemo_export.multimodal.run.HAVE_PIL", False), pytest.raises(UnavailableError):
                MultimodalModelRunner().video_preprocess(video_path="")

    def test_video_preprocess_without_pil_ndarray(self):
        from nemo_export.multimodal.run import MultimodalModelRunner

        with patch.object(MultimodalModelRunner, "__init__", lambda self: None):
            with patch("nemo_export.multimodal.run.HAVE_PIL", False), pytest.raises(UnavailableError):
                MultimodalModelRunner().video_preprocess(video_path=np.array(["hello"]))

    def test_load_video_without_decord(self):
        from nemo_export.multimodal.run import MultimodalModelRunner

        with patch.object(MultimodalModelRunner, "__init__", lambda self: None):
            with patch("nemo_export.multimodal.run.HAVE_DECORD", False), pytest.raises(UnavailableError):
                MultimodalModelRunner().load_video(config="", video_path="", processor="")

    def test_process_lita_video_without_decord(self):
        from nemo_export.multimodal.run import MultimodalModelRunner

        with patch.object(MultimodalModelRunner, "__init__", lambda self: None):
            with patch("nemo_export.multimodal.run.HAVE_DECORD", False), pytest.raises(UnavailableError):
                MultimodalModelRunner().process_lita_video(nemo_config="", video_path="", image_processor="")

    @pytest.mark.run_only_on("GPU")
    def test_setup_fake_prompts_with_post_input_ids(self):
        """Test setup_fake_prompts with post_input_ids provided"""
        runner = self._create_runner_with_mocks("neva")

        # Test data
        visual_features = torch.randn(1, 256, 4096)  # [batch_size, seq_len, hidden_size]
        pre_input_ids = torch.tensor([[1, 2, 3]], dtype=torch.int64)
        post_input_ids = torch.tensor([[4, 5, 6]], dtype=torch.int64)
        input_lengths = torch.tensor([10], dtype=torch.int32)

        # Mock ptuning_setup
        expected_ptuning_args = ["mock_prompt_table", "mock_tasks", "mock_task_vocab_size"]
        with patch.object(runner, "ptuning_setup", return_value=expected_ptuning_args) as mock_ptuning:
            input_ids, ptuning_args = runner.setup_fake_prompts(
                visual_features, pre_input_ids, post_input_ids, input_lengths
            )

            # Verify input_ids structure: [pre_input_ids, fake_prompt_id, post_input_ids]
            self.assertEqual(input_ids.dtype, torch.int32)
            self.assertEqual(input_ids.shape[0], 1)  # batch_size
            expected_length = pre_input_ids.shape[1] + visual_features.shape[1] + post_input_ids.shape[1]
            self.assertEqual(input_ids.shape[1], expected_length)

            # Verify fake_prompt_id generation (should be in the middle)
            fake_prompt_start = runner.model_config.vocab_size
            fake_prompt_end = fake_prompt_start + visual_features.shape[1]
            middle_section = input_ids[0, pre_input_ids.shape[1] : pre_input_ids.shape[1] + visual_features.shape[1]]
            expected_fake_ids = torch.arange(fake_prompt_start, fake_prompt_end, dtype=torch.int32)
            torch.testing.assert_close(middle_section, expected_fake_ids)

            # Verify pre and post prompts are correctly placed
            torch.testing.assert_close(input_ids[0, : pre_input_ids.shape[1]], pre_input_ids[0].to(torch.int32))
            torch.testing.assert_close(input_ids[0, -post_input_ids.shape[1] :], post_input_ids[0].to(torch.int32))

            # Verify ptuning_setup was called with correct parameters
            mock_ptuning.assert_called_once_with(visual_features, input_ids, input_lengths)

            # Verify returned ptuning_args
            self.assertEqual(ptuning_args, expected_ptuning_args)

    @pytest.mark.run_only_on("GPU")
    def test_setup_fake_prompts_without_post_input_ids(self):
        """Test setup_fake_prompts with post_input_ids=None"""
        runner = self._create_runner_with_mocks("neva")

        # Test data
        visual_features = torch.randn(1, 128, 4096)  # [batch_size, seq_len, hidden_size]
        pre_input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.int64)
        post_input_ids = None
        input_lengths = torch.tensor([8], dtype=torch.int32)

        # Mock ptuning_setup
        expected_ptuning_args = ["mock_prompt_table", "mock_tasks", "mock_task_vocab_size"]
        with patch.object(runner, "ptuning_setup", return_value=expected_ptuning_args) as mock_ptuning:
            input_ids, ptuning_args = runner.setup_fake_prompts(
                visual_features, pre_input_ids, post_input_ids, input_lengths
            )

            # Verify input_ids structure: [fake_prompt_id, pre_input_ids]
            self.assertEqual(input_ids.dtype, torch.int32)
            self.assertEqual(input_ids.shape[0], 1)  # batch_size
            expected_length = visual_features.shape[1] + pre_input_ids.shape[1]
            self.assertEqual(input_ids.shape[1], expected_length)

            # Verify fake_prompt_id generation (should be at the beginning)
            fake_prompt_start = runner.model_config.vocab_size
            fake_prompt_end = fake_prompt_start + visual_features.shape[1]
            fake_section = input_ids[0, : visual_features.shape[1]]
            expected_fake_ids = torch.arange(fake_prompt_start, fake_prompt_end, dtype=torch.int32)
            torch.testing.assert_close(fake_section, expected_fake_ids)

            # Verify pre_input_ids are correctly placed at the end
            torch.testing.assert_close(input_ids[0, visual_features.shape[1] :], pre_input_ids[0].to(torch.int32))

            # Verify ptuning_setup was called with correct parameters
            mock_ptuning.assert_called_once_with(visual_features, input_ids, input_lengths)

            # Verify returned ptuning_args
            self.assertEqual(ptuning_args, expected_ptuning_args)

    @pytest.mark.run_only_on("GPU")
    def test_setup_fake_prompts_with_num_frames_reshaping(self):
        """Test setup_fake_prompts with num_frames attribute that triggers reshaping"""
        runner = self._create_runner_with_mocks("video-neva")

        # Set num_frames attribute to trigger reshaping
        runner.num_frames = 4

        # Test data - visual_features.shape[1] == num_frames to trigger reshaping
        visual_features = torch.randn(1, 4, 256, 4096)  # [batch_size, num_frames, seq_len, hidden_size]
        pre_input_ids = torch.tensor([[1, 2]], dtype=torch.int64)
        post_input_ids = torch.tensor([[3, 4]], dtype=torch.int64)
        input_lengths = torch.tensor([8], dtype=torch.int32)

        # Mock ptuning_setup
        expected_ptuning_args = ["mock_prompt_table", "mock_tasks", "mock_task_vocab_size"]
        with patch.object(runner, "ptuning_setup", return_value=expected_ptuning_args) as mock_ptuning:
            input_ids, ptuning_args = runner.setup_fake_prompts(
                visual_features, pre_input_ids, post_input_ids, input_lengths
            )

            # Verify that visual_features was reshaped
            # The reshaped visual_features should have shape [1, 4*256, 4096] = [1, 1024, 4096]
            expected_reshaped_seq_len = 4 * 256  # num_frames * original_seq_len

            # Verify input_ids has correct length accounting for reshaped visual_features
            expected_length = pre_input_ids.shape[1] + expected_reshaped_seq_len + post_input_ids.shape[1]
            self.assertEqual(input_ids.shape[1], expected_length)

            # Verify fake_prompt_id generation accounts for reshaping
            fake_prompt_start = runner.model_config.vocab_size
            fake_prompt_end = fake_prompt_start + expected_reshaped_seq_len
            middle_section = input_ids[0, pre_input_ids.shape[1] : pre_input_ids.shape[1] + expected_reshaped_seq_len]
            expected_fake_ids = torch.arange(fake_prompt_start, fake_prompt_end, dtype=torch.int32)
            torch.testing.assert_close(middle_section, expected_fake_ids)

            # Verify ptuning_setup was called with reshaped visual_features
            mock_ptuning.assert_called_once()
            call_args = mock_ptuning.call_args[0]
            reshaped_visual_features = call_args[0]
            self.assertEqual(reshaped_visual_features.shape, (1, expected_reshaped_seq_len, 4096))

    @pytest.mark.run_only_on("GPU")
    def test_setup_fake_prompts_without_num_frames(self):
        """Test setup_fake_prompts without num_frames attribute (no reshaping)"""
        runner = self._create_runner_with_mocks("neva")

        # Ensure num_frames is not set (or doesn't match)
        if hasattr(runner, "num_frames"):
            delattr(runner, "num_frames")

        # Test data
        visual_features = torch.randn(1, 256, 4096)  # [batch_size, seq_len, hidden_size]
        pre_input_ids = torch.tensor([[1, 2, 3]], dtype=torch.int64)
        post_input_ids = torch.tensor([[4, 5]], dtype=torch.int64)
        input_lengths = torch.tensor([10], dtype=torch.int32)

        # Mock ptuning_setup
        expected_ptuning_args = ["mock_prompt_table", "mock_tasks", "mock_task_vocab_size"]
        with patch.object(runner, "ptuning_setup", return_value=expected_ptuning_args) as mock_ptuning:
            input_ids, ptuning_args = runner.setup_fake_prompts(
                visual_features, pre_input_ids, post_input_ids, input_lengths
            )

            # Verify that visual_features was NOT reshaped (same sequence length)
            expected_length = pre_input_ids.shape[1] + visual_features.shape[1] + post_input_ids.shape[1]
            self.assertEqual(input_ids.shape[1], expected_length)

            # Verify ptuning_setup was called with original visual_features
            mock_ptuning.assert_called_once()
            call_args = mock_ptuning.call_args[0]
            passed_visual_features = call_args[0]
            self.assertEqual(passed_visual_features.shape, visual_features.shape)

    @pytest.mark.run_only_on("GPU")
    def test_setup_fake_prompts_fake_prompt_id_generation(self):
        """Test setup_fake_prompts fake_prompt_id generation logic"""
        runner = self._create_runner_with_mocks("neva")

        # Test data with specific shapes to verify calculations
        batch_size = 2
        seq_len = 64
        hidden_size = 4096
        visual_features = torch.randn(batch_size, seq_len, hidden_size)
        pre_input_ids = torch.tensor([[1, 2], [3, 4]], dtype=torch.int64)
        post_input_ids = torch.tensor([[5], [6]], dtype=torch.int64)
        input_lengths = torch.tensor([10, 10], dtype=torch.int32)

        # Mock ptuning_setup
        expected_ptuning_args = ["mock_prompt_table", "mock_tasks", "mock_task_vocab_size"]
        with patch.object(runner, "ptuning_setup", return_value=expected_ptuning_args) as mock_ptuning:
            input_ids, ptuning_args = runner.setup_fake_prompts(
                visual_features, pre_input_ids, post_input_ids, input_lengths
            )

            # Verify fake_prompt_id generation
            vocab_size = runner.model_config.vocab_size
            expected_fake_start = vocab_size
            expected_fake_end = vocab_size + (batch_size * seq_len)

            # The fake_prompt_id should be reshaped to [batch_size, seq_len]
            # For batch_size=2, seq_len=64, it should be:
            # Row 0: [vocab_size, vocab_size+1, ..., vocab_size+63]
            # Row 1: [vocab_size+64, vocab_size+65, ..., vocab_size+127]

            # Extract the fake prompt section from input_ids
            fake_section = input_ids[:, pre_input_ids.shape[1] : pre_input_ids.shape[1] + seq_len]

            # Verify the fake IDs are correctly generated and placed
            expected_fake_ids = torch.arange(expected_fake_start, expected_fake_end, dtype=torch.int32)
            expected_fake_ids = expected_fake_ids.reshape(batch_size, seq_len)
            torch.testing.assert_close(fake_section, expected_fake_ids)

    @pytest.mark.run_only_on("GPU")
    def test_setup_fake_prompts_input_ids_contiguous_and_type(self):
        """Test setup_fake_prompts ensures input_ids are contiguous and correct type"""
        runner = self._create_runner_with_mocks("neva")

        # Test data
        visual_features = torch.randn(1, 128, 4096)
        pre_input_ids = torch.tensor([[1, 2, 3]], dtype=torch.int64)  # int64 input
        post_input_ids = torch.tensor([[4, 5]], dtype=torch.int64)  # int64 input
        input_lengths = torch.tensor([10], dtype=torch.int32)

        # Mock ptuning_setup
        expected_ptuning_args = ["mock_prompt_table", "mock_tasks", "mock_task_vocab_size"]
        with patch.object(runner, "ptuning_setup", return_value=expected_ptuning_args) as mock_ptuning:
            input_ids, ptuning_args = runner.setup_fake_prompts(
                visual_features, pre_input_ids, post_input_ids, input_lengths
            )

            # Verify input_ids type is torch.int32 (converted from int64)
            self.assertEqual(input_ids.dtype, torch.int32)

            # Verify input_ids is contiguous
            self.assertTrue(input_ids.is_contiguous())

            # Verify the tensor is properly formatted
            self.assertEqual(len(input_ids.shape), 2)  # Should be 2D tensor

            # Verify

            self.assertEqual(len(result), 3)  # Should return 3 elements
        self.assertIsInstance(result[0], torch.Tensor)  # prompt_table
        self.assertIsInstance(result[1], torch.Tensor)  # tasks
        self.assertIsInstance(result[2], torch.Tensor)  # task_vocab_size

    @pytest.mark.run_only_on("GPU")
    def test_setup_fake_prompts_with_post_input_ids(self):
        """Test setup_fake_prompts with post_input_ids provided"""
        runner = self._create_runner_with_mocks("neva")

        # Test data
        visual_features = torch.randn(1, 256, 4096)  # [batch_size, seq_len, hidden_size]
        pre_input_ids = torch.tensor([[1, 2, 3]], dtype=torch.int64)
        post_input_ids = torch.tensor([[4, 5, 6]], dtype=torch.int64)
        input_lengths = torch.tensor([10], dtype=torch.int32)

        # Mock ptuning_setup
        expected_ptuning_args = ["mock_prompt_table", "mock_tasks", "mock_task_vocab_size"]
        with patch.object(runner, "ptuning_setup", return_value=expected_ptuning_args) as mock_ptuning:
            input_ids, ptuning_args = runner.setup_fake_prompts(
                visual_features, pre_input_ids, post_input_ids, input_lengths
            )

            # Verify input_ids structure: [pre_input_ids, fake_prompt_id, post_input_ids]
            self.assertEqual(input_ids.dtype, torch.int32)
            self.assertEqual(input_ids.shape[0], 1)  # batch_size
            expected_length = pre_input_ids.shape[1] + visual_features.shape[1] + post_input_ids.shape[1]
            self.assertEqual(input_ids.shape[1], expected_length)

            # Verify fake_prompt_id generation (should be in the middle)
            fake_prompt_start = runner.model_config.vocab_size
            fake_prompt_end = fake_prompt_start + visual_features.shape[1]
            middle_section = input_ids[0, pre_input_ids.shape[1] : pre_input_ids.shape[1] + visual_features.shape[1]]
            expected_fake_ids = torch.arange(fake_prompt_start, fake_prompt_end, dtype=torch.int32)
            torch.testing.assert_close(middle_section, expected_fake_ids)

            # Verify pre and post prompts are correctly placed
            torch.testing.assert_close(input_ids[0, : pre_input_ids.shape[1]], pre_input_ids[0].to(torch.int32))
            torch.testing.assert_close(input_ids[0, -post_input_ids.shape[1] :], post_input_ids[0].to(torch.int32))

            # Verify ptuning_setup was called with correct parameters
            mock_ptuning.assert_called_once_with(visual_features, input_ids, input_lengths)

            # Verify returned ptuning_args
            self.assertEqual(ptuning_args, expected_ptuning_args)

    @pytest.mark.run_only_on("GPU")
    def test_setup_fake_prompts_without_post_input_ids(self):
        """Test setup_fake_prompts with post_input_ids=None"""
        runner = self._create_runner_with_mocks("neva")

        # Test data
        visual_features = torch.randn(1, 128, 4096)  # [batch_size, seq_len, hidden_size]
        pre_input_ids = torch.tensor([[1, 2, 3, 4]], dtype=torch.int64)
        post_input_ids = None
        input_lengths = torch.tensor([8], dtype=torch.int32)

        # Mock ptuning_setup
        expected_ptuning_args = ["mock_prompt_table", "mock_tasks", "mock_task_vocab_size"]
        with patch.object(runner, "ptuning_setup", return_value=expected_ptuning_args) as mock_ptuning:
            input_ids, ptuning_args = runner.setup_fake_prompts(
                visual_features, pre_input_ids, post_input_ids, input_lengths
            )

            # Verify input_ids structure: [fake_prompt_id, pre_input_ids]
            self.assertEqual(input_ids.dtype, torch.int32)
            self.assertEqual(input_ids.shape[0], 1)  # batch_size
            expected_length = visual_features.shape[1] + pre_input_ids.shape[1]
            self.assertEqual(input_ids.shape[1], expected_length)

            # Verify fake_prompt_id generation (should be at the beginning)
            fake_prompt_start = runner.model_config.vocab_size
            fake_prompt_end = fake_prompt_start + visual_features.shape[1]
            fake_section = input_ids[0, : visual_features.shape[1]]
            expected_fake_ids = torch.arange(fake_prompt_start, fake_prompt_end, dtype=torch.int32)
            torch.testing.assert_close(fake_section, expected_fake_ids)

            # Verify pre_input_ids are correctly placed at the end
            torch.testing.assert_close(input_ids[0, visual_features.shape[1] :], pre_input_ids[0].to(torch.int32))

            # Verify ptuning_setup was called with correct parameters
            mock_ptuning.assert_called_once_with(visual_features, input_ids, input_lengths)

            # Verify returned ptuning_args
            self.assertEqual(ptuning_args, expected_ptuning_args)

    @pytest.mark.run_only_on("GPU")
    def test_setup_fake_prompts_with_num_frames_reshaping(self):
        """Test setup_fake_prompts with num_frames attribute that triggers reshaping"""
        runner = self._create_runner_with_mocks("video-neva")

        # Set num_frames attribute to trigger reshaping
        runner.num_frames = 4

        # Test data - visual_features.shape[1] == num_frames to trigger reshaping
        visual_features = torch.randn(1, 4, 256, 4096)  # [batch_size, num_frames, seq_len, hidden_size]
        pre_input_ids = torch.tensor([[1, 2]], dtype=torch.int64)
        post_input_ids = torch.tensor([[3, 4]], dtype=torch.int64)
        input_lengths = torch.tensor([8], dtype=torch.int32)

        # Mock ptuning_setup
        expected_ptuning_args = ["mock_prompt_table", "mock_tasks", "mock_task_vocab_size"]
        with patch.object(runner, "ptuning_setup", return_value=expected_ptuning_args) as mock_ptuning:
            input_ids, ptuning_args = runner.setup_fake_prompts(
                visual_features, pre_input_ids, post_input_ids, input_lengths
            )

            # Verify that visual_features was reshaped
            # The reshaped visual_features should have shape [1, 4*256, 4096] = [1, 1024, 4096]
            expected_reshaped_seq_len = 4 * 256  # num_frames * original_seq_len

            # Verify input_ids has correct length accounting for reshaped visual_features
            expected_length = pre_input_ids.shape[1] + expected_reshaped_seq_len + post_input_ids.shape[1]
            self.assertEqual(input_ids.shape[1], expected_length)

            # Verify fake_prompt_id generation accounts for reshaping
            fake_prompt_start = runner.model_config.vocab_size
            fake_prompt_end = fake_prompt_start + expected_reshaped_seq_len
            middle_section = input_ids[0, pre_input_ids.shape[1] : pre_input_ids.shape[1] + expected_reshaped_seq_len]
            expected_fake_ids = torch.arange(fake_prompt_start, fake_prompt_end, dtype=torch.int32)
            torch.testing.assert_close(middle_section, expected_fake_ids)

            # Verify ptuning_setup was called with reshaped visual_features
            mock_ptuning.assert_called_once()
            call_args = mock_ptuning.call_args[0]
            reshaped_visual_features = call_args[0]
            self.assertEqual(reshaped_visual_features.shape, (1, expected_reshaped_seq_len, 4096))

    @pytest.mark.run_only_on("GPU")
    def test_setup_fake_prompts_without_num_frames(self):
        """Test setup_fake_prompts without num_frames attribute (no reshaping)"""
        runner = self._create_runner_with_mocks("neva")

        # Ensure num_frames is not set (or doesn't match)
        if hasattr(runner, "num_frames"):
            delattr(runner, "num_frames")

        # Test data
        visual_features = torch.randn(1, 256, 4096)  # [batch_size, seq_len, hidden_size]
        pre_input_ids = torch.tensor([[1, 2, 3]], dtype=torch.int64)
        post_input_ids = torch.tensor([[4, 5]], dtype=torch.int64)
        input_lengths = torch.tensor([10], dtype=torch.int32)

        # Mock ptuning_setup
        expected_ptuning_args = ["mock_prompt_table", "mock_tasks", "mock_task_vocab_size"]
        with patch.object(runner, "ptuning_setup", return_value=expected_ptuning_args) as mock_ptuning:
            input_ids, ptuning_args = runner.setup_fake_prompts(
                visual_features, pre_input_ids, post_input_ids, input_lengths
            )

            # Verify that visual_features was NOT reshaped (same sequence length)
            expected_length = pre_input_ids.shape[1] + visual_features.shape[1] + post_input_ids.shape[1]
            self.assertEqual(input_ids.shape[1], expected_length)

            # Verify ptuning_setup was called with original visual_features
            mock_ptuning.assert_called_once()
            call_args = mock_ptuning.call_args[0]
            passed_visual_features = call_args[0]
            self.assertEqual(passed_visual_features.shape, visual_features.shape)

    @pytest.mark.run_only_on("GPU")
    def test_setup_fake_prompts_fake_prompt_id_generation(self):
        """Test setup_fake_prompts fake_prompt_id generation logic"""
        runner = self._create_runner_with_mocks("neva")

        # Test data with specific shapes to verify calculations
        batch_size = 2
        seq_len = 64
        hidden_size = 4096
        visual_features = torch.randn(batch_size, seq_len, hidden_size)
        pre_input_ids = torch.tensor([[1, 2], [3, 4]], dtype=torch.int64)
        post_input_ids = torch.tensor([[5], [6]], dtype=torch.int64)
        input_lengths = torch.tensor([10, 10], dtype=torch.int32)

        # Mock ptuning_setup
        expected_ptuning_args = ["mock_prompt_table", "mock_tasks", "mock_task_vocab_size"]
        with patch.object(runner, "ptuning_setup", return_value=expected_ptuning_args) as mock_ptuning:
            input_ids, ptuning_args = runner.setup_fake_prompts(
                visual_features, pre_input_ids, post_input_ids, input_lengths
            )

            # Verify fake_prompt_id generation
            vocab_size = runner.model_config.vocab_size
            expected_fake_start = vocab_size
            expected_fake_end = vocab_size + (batch_size * seq_len)

            # The fake_prompt_id should be reshaped to [batch_size, seq_len]
            # For batch_size=2, seq_len=64, it should be:
            # Row 0: [vocab_size, vocab_size+1, ..., vocab_size+63]
            # Row 1: [vocab_size+64, vocab_size+65, ..., vocab_size+127]

            # Extract the fake prompt section from input_ids
            fake_section = input_ids[:, pre_input_ids.shape[1] : pre_input_ids.shape[1] + seq_len]

            # Verify the fake IDs are correctly generated and placed
            expected_fake_ids = torch.arange(expected_fake_start, expected_fake_end, dtype=torch.int32)
            expected_fake_ids = expected_fake_ids.reshape(batch_size, seq_len)
            torch.testing.assert_close(fake_section, expected_fake_ids)

    @pytest.mark.run_only_on("GPU")
    def test_setup_fake_prompts_input_ids_contiguous_and_type(self):
        """Test setup_fake_prompts ensures input_ids are contiguous and correct type"""
        runner = self._create_runner_with_mocks("neva")

        # Test data
        visual_features = torch.randn(1, 128, 4096)
        pre_input_ids = torch.tensor([[1, 2, 3]], dtype=torch.int64)  # int64 input
        post_input_ids = torch.tensor([[4, 5]], dtype=torch.int64)  # int64 input
        input_lengths = torch.tensor([10], dtype=torch.int32)

        # Mock ptuning_setup
        expected_ptuning_args = ["mock_prompt_table", "mock_tasks", "mock_task_vocab_size"]
        with patch.object(runner, "ptuning_setup", return_value=expected_ptuning_args) as mock_ptuning:
            input_ids, ptuning_args = runner.setup_fake_prompts(
                visual_features, pre_input_ids, post_input_ids, input_lengths
            )

            # Verify input_ids type is torch.int32 (converted from int64)
            self.assertEqual(input_ids.dtype, torch.int32)

            # Verify input_ids is contiguous
            self.assertTrue(input_ids.is_contiguous())

            # Verify the tensor is properly formatted
            self.assertEqual(len(input_ids.shape), 2)  # Should be 2D tensor

            # Verify ptuning_setup was called with the correctly formatted input_ids
            mock_ptuning.assert_called_once()
            call_args = mock_ptuning.call_args[0]
            passed_input_ids = call_args[1]
            self.assertEqual(passed_input_ids.dtype, torch.int32)
            self.assertTrue(passed_input_ids.is_contiguous())

    @pytest.mark.run_only_on("GPU")
    @patch("nemo_export.multimodal.run.torch.cuda.set_device")
    @patch("nemo_export.multimodal.run.torch.cuda.device_count", return_value=1)
    @patch("nemo_export.multimodal.run.tensorrt_llm.mpi_rank", return_value=0)
    @patch("nemo_export.multimodal.run.ModelRunner.from_dir")
    @patch("nemo_export.multimodal.run.Session.from_serialized_engine")
    @patch("transformers.AutoTokenizer.from_pretrained")
    def test_run_function_basic(
        self,
        mock_tokenizer,
        mock_session,
        mock_model_runner,
        mock_mpi_rank,
        mock_device_count,
        mock_set_device,
    ):
        """Test the run function with basic parameters"""
        runner = self._create_runner_with_mocks("vila")

        # Mock the methods that run() depends on
        mock_setup_inputs_return = (
            "test input",  # input_text
            ["test pre prompt"],  # pre_prompt
            ["test post prompt"],  # post_prompt
            torch.randn(1, 3, 224, 224),  # processed_image
            None,  # decoder_input_ids
            None,  # attention_mask
        )

        mock_generate_return = [["Generated response text"]]

        with (
            patch.object(runner, "setup_inputs", return_value=mock_setup_inputs_return) as mock_setup_inputs,
            patch.object(runner, "generate", return_value=mock_generate_return) as mock_generate,
            patch.object(runner, "print_result") as mock_print_result,
        ):
            # Test basic run
            result = runner.run(
                input_text="Hello, what's in this image?",
                input_image=Image.new("RGB", (224, 224), color="red"),
                max_new_tokens=50,
                batch_size=1,
                top_k=1,
                top_p=0.9,
                temperature=0.7,
                repetition_penalty=1.0,
                num_beams=1,
            )

            # Verify setup_inputs was called correctly
            mock_setup_inputs.assert_called_once()
            setup_call_args = mock_setup_inputs.call_args[0]
            self.assertEqual(setup_call_args[0], "Hello, what's in this image?")
            self.assertEqual(setup_call_args[2], 1)  # batch_size is third argument

            # Verify generate was called twice (warmup=True, then warmup=False)
            self.assertEqual(mock_generate.call_count, 2)

            # Check warmup call (first call)
            warmup_call = mock_generate.call_args_list[0]
            self.assertTrue(warmup_call[1]["warmup"])

            # Check actual generation call (second call)
            actual_call = mock_generate.call_args_list[1]
            self.assertFalse(actual_call[1]["warmup"])
            self.assertEqual(actual_call[1]["batch_size"], 1)
            # self.assertEqual(actual_call[1]['max_new_tokens'], 50)
            self.assertEqual(actual_call[1]["top_k"], 1)
            self.assertEqual(actual_call[1]["top_p"], 0.9)
            self.assertEqual(actual_call[1]["temperature"], 0.7)
            self.assertEqual(actual_call[1]["repetition_penalty"], 1.0)
            self.assertEqual(actual_call[1]["num_beams"], 1)

            # Verify print_result was called
            mock_print_result.assert_called_once()

            # Verify return value
            self.assertEqual(result, mock_generate_return)

    @pytest.mark.run_only_on("GPU")
    @patch("nemo_export.multimodal.run.torch.cuda.set_device")
    @patch("nemo_export.multimodal.run.torch.cuda.device_count", return_value=1)
    @patch("nemo_export.multimodal.run.tensorrt_llm.mpi_rank", return_value=0)
    @patch("nemo_export.multimodal.run.ModelRunner.from_dir")
    @patch("nemo_export.multimodal.run.Session.from_serialized_engine")
    @patch("transformers.AutoTokenizer.from_pretrained")
    def test_run_function_with_profiling(
        self,
        mock_tokenizer,
        mock_session,
        mock_model_runner,
        mock_mpi_rank,
        mock_device_count,
        mock_set_device,
    ):
        """Test the run function with profiling enabled"""
        runner = self._create_runner_with_mocks("vila")
        runner.profiling_iterations = 3  # Set to 3 for testing

        # Mock the methods that run() depends on
        mock_setup_inputs_return = (
            "test input",
            ["test pre prompt"],
            ["test post prompt"],
            torch.randn(1, 3, 224, 224),
            None,
            None,
        )

        mock_generate_return = [["Generated response text"]]

        with (
            patch.object(runner, "setup_inputs", return_value=mock_setup_inputs_return),
            patch.object(runner, "generate", return_value=mock_generate_return) as mock_generate,
            patch.object(runner, "print_result") as mock_print_result,
        ):
            # Test run with profiling
            result = runner.run(
                input_text="Hello, what's in this image?",
                input_image=Image.new("RGB", (224, 224), color="red"),
                max_new_tokens=50,
                batch_size=1,
                top_k=1,
                top_p=0.9,
                temperature=0.7,
                repetition_penalty=1.0,
                num_beams=1,
                run_profiling=True,
            )

            # Verify generate was called 1 (warmup) + 3 (profiling iterations) = 4 times
            self.assertEqual(mock_generate.call_count, 4)

            # Check that first call is warmup
            warmup_call = mock_generate.call_args_list[0]
            self.assertTrue(warmup_call[1]["warmup"])

            # Check that subsequent calls are not warmup
            for i in range(1, 4):
                actual_call = mock_generate.call_args_list[i]
                self.assertFalse(actual_call[1]["warmup"])

            # Verify print_result was called with profiling enabled
            mock_print_result.assert_called_once()
            print_result_args = mock_print_result.call_args[0]
            print_result_kwargs = mock_print_result.call_args[1]

            # Check that run_profiling was passed to print_result
            self.assertTrue(
                print_result_kwargs.get("run_profiling", False)
                or len(print_result_args) > 4
                and print_result_args[4] == True
            )

            # Verify return value
            self.assertEqual(result, mock_generate_return)

    @pytest.mark.run_only_on("GPU")
    @patch("nemo_export.multimodal.run.torch.cuda.set_device")
    @patch("nemo_export.multimodal.run.torch.cuda.device_count", return_value=1)
    @patch("nemo_export.multimodal.run.tensorrt_llm.mpi_rank", return_value=0)
    @patch("nemo_export.multimodal.run.ModelRunner.from_dir")
    @patch("nemo_export.multimodal.run.Session.from_serialized_engine")
    @patch("transformers.AutoTokenizer.from_pretrained")
    def test_run_function_with_lora_uids(
        self,
        mock_tokenizer,
        mock_session,
        mock_model_runner,
        mock_mpi_rank,
        mock_device_count,
        mock_set_device,
    ):
        """Test the run function with LoRA UIDs"""
        runner = self._create_runner_with_mocks("vila")

        # Mock the methods that run() depends on
        mock_setup_inputs_return = (
            "test input",
            ["test pre prompt"],
            ["test post prompt"],
            torch.randn(1, 3, 224, 224),
            None,
            None,
        )

        mock_generate_return = [["Generated response text"]]

        with (
            patch.object(runner, "setup_inputs", return_value=mock_setup_inputs_return),
            patch.object(runner, "generate", return_value=mock_generate_return) as mock_generate,
            patch.object(runner, "print_result"),
        ):
            # Test run with LoRA UIDs
            lora_uids = ["lora_1", "lora_2"]
            result = runner.run(
                input_text="Hello, what's in this image?",
                input_image=Image.new("RGB", (224, 224), color="red"),
                max_new_tokens=50,
                batch_size=1,
                top_k=1,
                top_p=0.9,
                temperature=0.7,
                repetition_penalty=1.0,
                num_beams=1,
                lora_uids=lora_uids,
            )

            # Verify generate was called with LoRA UIDs
            self.assertEqual(mock_generate.call_count, 2)

            # Check that both calls include lora_uids
            for call in mock_generate.call_args_list:
                self.assertEqual(call[1]["lora_uids"], lora_uids)

            # Verify return value
            self.assertEqual(result, mock_generate_return)

    @pytest.mark.run_only_on("GPU")
    @patch("nemo_export.multimodal.run.torch.cuda.set_device")
    @patch("nemo_export.multimodal.run.torch.cuda.device_count", return_value=1)
    @patch("nemo_export.multimodal.run.tensorrt_llm.mpi_rank", return_value=0)
    @patch("nemo_export.multimodal.run.ModelRunner.from_dir")
    @patch("nemo_export.multimodal.run.Session.from_serialized_engine")
    @patch("transformers.AutoTokenizer.from_pretrained")
    def test_run_function_with_check_accuracy(
        self,
        mock_tokenizer,
        mock_session,
        mock_model_runner,
        mock_mpi_rank,
        mock_device_count,
        mock_set_device,
    ):
        """Test the run function with accuracy checking enabled"""
        runner = self._create_runner_with_mocks("vila")

        # Mock the methods that run() depends on
        mock_setup_inputs_return = (
            "test input",
            ["test pre prompt"],
            ["test post prompt"],
            torch.randn(1, 3, 224, 224),
            None,
            None,
        )

        mock_generate_return = [["Generated response text"]]

        with (
            patch.object(runner, "setup_inputs", return_value=mock_setup_inputs_return),
            patch.object(runner, "generate", return_value=mock_generate_return) as mock_generate,
            patch.object(runner, "print_result") as mock_print_result,
        ):
            # Test run with accuracy checking
            result = runner.run(
                input_text="Hello, what's in this image?",
                input_image=Image.new("RGB", (224, 224), color="red"),
                max_new_tokens=50,
                batch_size=1,
                top_k=1,
                top_p=0.9,
                temperature=0.7,
                repetition_penalty=1.0,
                num_beams=1,
                check_accuracy=True,
            )

            # Verify generate was called
            self.assertEqual(mock_generate.call_count, 2)

            # Verify print_result was called with check_accuracy enabled
            mock_print_result.assert_called_once()
            print_result_args = mock_print_result.call_args[0]
            print_result_kwargs = mock_print_result.call_args[1]

            # Check that check_accuracy was passed to print_result
            self.assertTrue(
                print_result_kwargs.get("check_accuracy", False)
                or len(print_result_args) > 5
                and print_result_args[5] == True
            )

            # Verify return value
            self.assertEqual(result, mock_generate_return)

    @pytest.mark.run_only_on("GPU")
    @patch("nemo_export.multimodal.run.torch.cuda.set_device")
    @patch("nemo_export.multimodal.run.torch.cuda.device_count", return_value=1)
    @patch("nemo_export.multimodal.run.tensorrt_llm.mpi_rank", return_value=0)
    @patch("nemo_export.multimodal.run.ModelRunner.from_dir")
    @patch("nemo_export.multimodal.run.Session.from_serialized_engine")
    @patch("transformers.AutoTokenizer.from_pretrained")
    def test_run_function_batch_size_multiple(
        self,
        mock_tokenizer,
        mock_session,
        mock_model_runner,
        mock_mpi_rank,
        mock_device_count,
        mock_set_device,
    ):
        """Test the run function with batch size greater than 1"""
        runner = self._create_runner_with_mocks("vila")

        # Mock the methods that run() depends on
        mock_setup_inputs_return = (
            "test input",
            ["test pre prompt"] * 2,  # batch_size = 2
            ["test post prompt"] * 2,
            torch.randn(2, 3, 224, 224),
            None,
            None,
        )

        mock_generate_return = [["Generated response text 1"], ["Generated response text 2"]]

        with (
            patch.object(runner, "setup_inputs", return_value=mock_setup_inputs_return) as mock_setup_inputs,
            patch.object(runner, "generate", return_value=mock_generate_return) as mock_generate,
            patch.object(runner, "print_result") as mock_print_result,
        ):
            # Test run with batch size 2
            result = runner.run(
                input_text="Hello, what's in this image?",
                input_image=Image.new("RGB", (224, 224), color="red"),
                max_new_tokens=50,
                batch_size=2,
                top_k=1,
                top_p=0.9,
                temperature=0.7,
                repetition_penalty=1.0,
                num_beams=1,
            )

            # Verify setup_inputs was called with batch_size=2
            mock_setup_inputs.assert_called_once()
            setup_call_args = mock_setup_inputs.call_args[0]
            self.assertEqual(setup_call_args[2], 2)  # batch_size is third argument

            # Verify generate was called with batch_size=2
            for call in mock_generate.call_args_list:
                self.assertEqual(call[1]["batch_size"], 2)

            # Verify return value
            self.assertEqual(result, mock_generate_return)

    @pytest.mark.run_only_on("GPU")
    @patch("nemo_export.multimodal.run.torch.cuda.set_device")
    @patch("nemo_export.multimodal.run.torch.cuda.device_count", return_value=1)
    @patch("nemo_export.multimodal.run.tensorrt_llm.mpi_rank", return_value=0)
    @patch("nemo_export.multimodal.run.ModelRunner.from_dir")
    @patch("nemo_export.multimodal.run.Session.from_serialized_engine")
    @patch("transformers.AutoTokenizer.from_pretrained")
    def test_run_function_exception_handling(
        self,
        mock_tokenizer,
        mock_session,
        mock_model_runner,
        mock_mpi_rank,
        mock_device_count,
        mock_set_device,
    ):
        """Test the run function handles exceptions properly"""
        runner = self._create_runner_with_mocks("vila")

        # Mock setup_inputs to raise an exception
        with patch.object(runner, "setup_inputs", side_effect=RuntimeError("Setup failed")):
            # Test that exception is propagated
            with self.assertRaises(RuntimeError) as context:
                runner.run(
                    input_text="Hello, what's in this image?",
                    input_image=Image.new("RGB", (224, 224), color="red"),
                    max_new_tokens=50,
                    batch_size=1,
                    top_k=1,
                    top_p=0.9,
                    temperature=0.7,
                    repetition_penalty=1.0,
                    num_beams=1,
                )

            self.assertEqual(str(context.exception), "Setup failed")


if __name__ == "__main__":
    unittest.main()
