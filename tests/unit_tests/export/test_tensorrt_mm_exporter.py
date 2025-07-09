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


from unittest.mock import Mock, patch

import numpy as np
import pytest


@pytest.fixture
def model_dir(tmp_path):
    return str(tmp_path / "model_dir")


@pytest.fixture
def mock_runner():
    runner = Mock()
    runner.model_type = "neva"
    runner.load_test_media = Mock(return_value=np.zeros((1, 224, 224, 3)))
    runner.run = Mock(return_value="Test response")
    return runner


@pytest.fixture
def mock_trtllm_runner():
    runner = Mock()
    runner.model_type = "mllama"
    runner.args = Mock()
    runner.load_test_data = Mock(return_value=np.zeros((1, 224, 224, 3)))
    runner.run = Mock(return_value=["", "Test response"])
    return runner


try:
    import tensorrt_llm  # noqa: F401

    HAVE_TRTLLM = True
except ImportError:
    HAVE_TRTLLM = False


@pytest.mark.skipif(not HAVE_TRTLLM, reason="Skipping TensorRTMMExporter tests due to lack of trtllm")
class TestTensorRTMMExporter:
    @pytest.mark.run_only_on("GPU")
    def test_init(self, model_dir):
        # Test basic initialization
        from nemo_export.tensorrt_mm_exporter import TensorRTMMExporter

        exporter = TensorRTMMExporter(model_dir, load_model=False)
        assert exporter.model_dir == model_dir
        assert exporter.runner is None
        assert exporter.modality == "vision"

    @pytest.mark.run_only_on("GPU")
    def test_init_invalid_modality(self, model_dir):
        # Test initialization with invalid modality
        from nemo_export.tensorrt_mm_exporter import TensorRTMMExporter

        with pytest.raises(AssertionError):
            TensorRTMMExporter(model_dir, modality="invalid")

    @pytest.mark.run_only_on("GPU")
    @patch("nemo_export.tensorrt_mm_exporter.build_mllama_engine")
    def test_export_mllama(self, mock_build, model_dir):
        from nemo_export.tensorrt_mm_exporter import TensorRTMMExporter

        exporter = TensorRTMMExporter(model_dir, load_model=False)
        exporter.export(
            visual_checkpoint_path="dummy/path",
            model_type="mllama",
            tensor_parallel_size=1,
            load_model=False,
        )
        mock_build.assert_called_once()

    @pytest.mark.run_only_on("GPU")
    @patch("nemo_export.tensorrt_mm_exporter.build_trtllm_engine")
    @patch("nemo_export.tensorrt_mm_exporter.build_visual_engine")
    def test_export_neva(self, mock_visual, mock_trtllm, model_dir):
        from nemo_export.tensorrt_mm_exporter import TensorRTMMExporter

        exporter = TensorRTMMExporter(model_dir, load_model=False)
        exporter.export(
            visual_checkpoint_path="dummy/path",
            model_type="neva",
            tensor_parallel_size=1,
            load_model=False,
        )
        mock_trtllm.assert_called_once()
        mock_visual.assert_called_once()

    @pytest.mark.run_only_on("GPU")
    @patch("nemo_export.tensorrt_mm_exporter.build_trtllm_engine")
    @patch("nemo_export.tensorrt_mm_exporter.build_visual_engine")
    @patch("nemo_export.tensorrt_mm_exporter.extract_lora_ckpt")
    @patch("os.path.isdir")
    def test_export_with_lora(self, mock_isdir, mock_extract, mock_visual, mock_trtllm, model_dir):
        from nemo_export.tensorrt_mm_exporter import TensorRTMMExporter

        # Mock the LoRA path handling
        mock_isdir.return_value = True  # Treat as directory
        mock_extract.return_value = "dummy/lora/ckpt"

        exporter = TensorRTMMExporter(model_dir, load_model=False)
        exporter.export(
            visual_checkpoint_path="dummy/path",
            model_type="neva",
            tensor_parallel_size=1,
            load_model=False,
            lora_checkpoint_path="dummy/lora/path",
            use_lora_plugin="lora_plugin",
            lora_target_modules=["q_proj", "v_proj"],
            max_lora_rank=32,
        )
        mock_trtllm.assert_called_once()
        mock_visual.assert_called_once()
        mock_extract.assert_called_once()

    @pytest.mark.run_only_on("GPU")
    @patch("nemo_export.tensorrt_mm_exporter.build_trtllm_engine")
    @patch("nemo_export.tensorrt_mm_exporter.build_visual_engine")
    @patch("nemo_export.tensorrt_mm_exporter.extract_lora_ckpt")
    @patch("os.path.isdir")
    def test_export_with_lora_directory(self, mock_isdir, mock_extract, mock_visual, mock_trtllm, model_dir):
        from nemo_export.tensorrt_mm_exporter import TensorRTMMExporter

        # Mock the LoRA path handling - treat as directory
        mock_isdir.return_value = True  # Treat as directory
        mock_extract.return_value = "dummy/lora/ckpt"

        exporter = TensorRTMMExporter(model_dir, load_model=False)
        exporter.export(
            visual_checkpoint_path="dummy/path",
            model_type="neva",
            tensor_parallel_size=1,
            load_model=False,
            lora_checkpoint_path="dummy/lora/dir",
            use_lora_plugin="lora_plugin",
            lora_target_modules=["q_proj", "v_proj"],
            max_lora_rank=32,
        )
        mock_trtllm.assert_called_once()
        mock_visual.assert_called_once()
        mock_extract.assert_called_once()

    @pytest.mark.run_only_on("GPU")
    @patch("nemo_export.tensorrt_mm_exporter.build_trtllm_engine")
    @patch("nemo_export.tensorrt_mm_exporter.build_visual_engine")
    def test_export_vila(self, mock_visual, mock_trtllm, model_dir):
        from nemo_export.tensorrt_mm_exporter import TensorRTMMExporter

        exporter = TensorRTMMExporter(model_dir, load_model=False)
        exporter.export(
            visual_checkpoint_path="dummy/path",
            model_type="vila",
            tensor_parallel_size=1,
            load_model=False,
        )
        mock_trtllm.assert_called_once()
        mock_visual.assert_called_once()

    @pytest.mark.run_only_on("GPU")
    @patch("nemo_export.tensorrt_mm_exporter.build_trtllm_engine")
    @patch("nemo_export.tensorrt_mm_exporter.build_visual_engine")
    def test_export_video_neva(self, mock_visual, mock_trtllm, model_dir):
        from nemo_export.tensorrt_mm_exporter import TensorRTMMExporter

        exporter = TensorRTMMExporter(model_dir, load_model=False)
        exporter.export(
            visual_checkpoint_path="dummy/path",
            model_type="video-neva",
            tensor_parallel_size=1,
            load_model=False,
        )
        mock_trtllm.assert_called_once()
        mock_visual.assert_called_once()

    @pytest.mark.run_only_on("GPU")
    @patch("nemo_export.tensorrt_mm_exporter.build_trtllm_engine")
    @patch("nemo_export.tensorrt_mm_exporter.build_visual_engine")
    def test_export_lita(self, mock_visual, mock_trtllm, model_dir):
        from nemo_export.tensorrt_mm_exporter import TensorRTMMExporter

        exporter = TensorRTMMExporter(model_dir, load_model=False)
        exporter.export(
            visual_checkpoint_path="dummy/path",
            model_type="lita",
            tensor_parallel_size=1,
            load_model=False,
        )
        mock_trtllm.assert_called_once()
        mock_visual.assert_called_once()

    @pytest.mark.run_only_on("GPU")
    @patch("nemo_export.tensorrt_mm_exporter.build_trtllm_engine")
    @patch("nemo_export.tensorrt_mm_exporter.build_visual_engine")
    def test_export_vita(self, mock_visual, mock_trtllm, model_dir):
        from nemo_export.tensorrt_mm_exporter import TensorRTMMExporter

        exporter = TensorRTMMExporter(model_dir, load_model=False)
        exporter.export(
            visual_checkpoint_path="dummy/path",
            model_type="vita",
            tensor_parallel_size=1,
            load_model=False,
        )
        mock_trtllm.assert_called_once()
        mock_visual.assert_called_once()

    @pytest.mark.run_only_on("GPU")
    def test_forward_without_loading(self, model_dir):
        from nemo_export.tensorrt_mm_exporter import TensorRTMMExporter

        exporter = TensorRTMMExporter(model_dir, load_model=False)
        with pytest.raises(Exception) as exc_info:
            exporter.forward("test prompt", "test_image.jpg")
        assert "should be exported and" in str(exc_info.value)

    @pytest.mark.run_only_on("GPU")
    def test_forward(self, model_dir, mock_runner):
        from nemo_export.tensorrt_mm_exporter import TensorRTMMExporter

        exporter = TensorRTMMExporter(model_dir, load_model=False)
        exporter.runner = mock_runner

        result = exporter.forward(
            input_text="What's in this image?",
            input_media="test_image.jpg",
            batch_size=1,
            max_output_len=30,
        )

        assert result == "Test response"
        mock_runner.load_test_media.assert_called_once()
        mock_runner.run.assert_called_once()

    @pytest.mark.run_only_on("GPU")
    @patch("nemo_export.tensorrt_mm_exporter.isinstance")
    def test_forward_with_trtllm_runner(self, mock_isinstance, model_dir):
        from nemo_export.tensorrt_mm_exporter import TensorRTMMExporter

        # Create a mock runner
        mock_runner = Mock()
        mock_runner.model_type = "mllama"
        mock_runner.args = Mock()
        mock_runner.load_test_data = Mock(return_value=np.zeros((1, 224, 224, 3)))
        mock_runner.run = Mock(return_value=["", "Test response"])

        # Make isinstance return True for TRTLLMRunner check
        mock_isinstance.return_value = True

        exporter = TensorRTMMExporter(model_dir, load_model=False)
        exporter.runner = mock_runner

        result = exporter.forward(
            input_text="What's in this image?",
            input_media="test_image.jpg",
            batch_size=2,
            max_output_len=50,
            top_k=5,
            top_p=0.9,
            temperature=0.7,
            repetition_penalty=1.2,
            num_beams=4,
        )

        assert result == "Test response"
        assert mock_runner.args.image_path == "test_image.jpg"
        assert mock_runner.args.batch_size == 2
        assert mock_runner.args.top_k == 5
        assert mock_runner.args.top_p == 0.9
        assert mock_runner.args.temperature == 0.7
        assert mock_runner.args.repetition_penalty == 1.2
        assert mock_runner.args.num_beams == 4
        mock_runner.load_test_data.assert_called_once_with("test_image.jpg")
        mock_runner.run.assert_called_once()

    @pytest.mark.run_only_on("GPU")
    def test_get_triton_input(self, model_dir):
        from nemo_export.tensorrt_mm_exporter import TensorRTMMExporter

        exporter = TensorRTMMExporter(model_dir, load_model=False)
        inputs = exporter.get_triton_input

        # Verify we have the expected number of inputs
        assert len(inputs) == 10  # 1 text input + 1 media input + 8 optional parameters

        # Verify the first input is for text
        assert inputs[0].name == "input_text"
        assert inputs[0].dtype == bytes

    @pytest.mark.run_only_on("GPU")
    def test_get_triton_output(self, model_dir):
        from nemo_export.tensorrt_mm_exporter import TensorRTMMExporter

        exporter = TensorRTMMExporter(model_dir, load_model=False)
        outputs = exporter.get_triton_output

        assert len(outputs) == 1
        assert outputs[0].name == "outputs"
        assert outputs[0].dtype == bytes

    @pytest.mark.run_only_on("GPU")
    def test_forward_with_all_params(self, model_dir, mock_runner):
        from nemo_export.tensorrt_mm_exporter import TensorRTMMExporter

        exporter = TensorRTMMExporter(model_dir, load_model=False)
        exporter.runner = mock_runner

        result = exporter.forward(
            input_text="What's in this image?",
            input_media="test_image.jpg",
            batch_size=2,
            max_output_len=50,
            top_k=5,
            top_p=0.9,
            temperature=0.7,
            repetition_penalty=1.2,
            num_beams=4,
            lora_uids=["lora1", "lora2"],
        )

        assert result == "Test response"
        mock_runner.load_test_media.assert_called_once()
        mock_runner.run.assert_called_once_with(
            "What's in this image?",
            mock_runner.load_test_media.return_value,
            50,
            2,
            5,
            0.9,
            0.7,
            1.2,
            4,
            ["lora1", "lora2"],
        )

    @pytest.mark.run_only_on("GPU")
    def test_get_input_media_tensors_vision(self, model_dir):
        from nemo_export.tensorrt_mm_exporter import TensorRTMMExporter

        exporter = TensorRTMMExporter(model_dir, load_model=False, modality="vision")
        tensors = exporter.get_input_media_tensors()

        assert len(tensors) == 1
        assert tensors[0].name == "input_media"
        assert tensors[0].shape == (-1, -1, -1, 3)
        assert tensors[0].dtype == np.uint8

    @pytest.mark.run_only_on("GPU")
    def test_get_input_media_tensors_audio(self, model_dir):
        from nemo_export.tensorrt_mm_exporter import TensorRTMMExporter

        exporter = TensorRTMMExporter(model_dir, load_model=False, modality="audio")
        tensors = exporter.get_input_media_tensors()

        assert len(tensors) == 0

    @pytest.mark.run_only_on("GPU")
    def test_export_with_invalid_model_type(self, model_dir):
        from nemo_export.tensorrt_mm_exporter import TensorRTMMExporter

        exporter = TensorRTMMExporter(model_dir, load_model=False)
        with pytest.raises(Exception):
            exporter.export(
                visual_checkpoint_path="dummy/path",
                model_type="invalid_model_type",
                tensor_parallel_size=1,
                load_model=False,
            )

    @pytest.mark.run_only_on("GPU")
    def test_export_with_existing_files(self, model_dir):
        import os

        from nemo_export.tensorrt_mm_exporter import TensorRTMMExporter

        # Create some files in the model directory
        os.makedirs(model_dir, exist_ok=True)
        with open(os.path.join(model_dir, "test.txt"), "w") as f:
            f.write("test")

        exporter = TensorRTMMExporter(model_dir, load_model=False)
        with pytest.raises(Exception) as exc_info:
            exporter.export(
                visual_checkpoint_path="dummy/path",
                model_type="neva",
                tensor_parallel_size=1,
                load_model=False,
                delete_existing_files=False,
            )
        assert "There are files in this folder" in str(exc_info.value)

    @pytest.mark.run_only_on("GPU")
    @patch("os.path.exists")
    def test_load_no_llm_dir(self, mock_exists, model_dir):
        from nemo_export.tensorrt_mm_exporter import TensorRTMMExporter

        mock_exists.return_value = False
        exporter = TensorRTMMExporter(model_dir, load_model=False)
        exporter._load()
        assert exporter.runner is None

    @pytest.mark.run_only_on("GPU")
    @patch("os.path.exists")
    @patch("builtins.open", create=True)
    @patch("json.load")
    def test_load_mllama_model(self, mock_json_load, mock_open, mock_exists, model_dir):
        from nemo_export.tensorrt_mm_exporter import TensorRTMMExporter

        mock_exists.return_value = True
        mock_json_load.return_value = {"builder_config": {"model_type": "mllama"}}
        mock_open.return_value.__enter__ = lambda x: x
        mock_open.return_value.__exit__ = lambda x, y, z, w: None

        with patch("nemo_export.tensorrt_mm_exporter.TRTLLMRunner") as mock_trtllm_runner:
            exporter = TensorRTMMExporter(model_dir, load_model=False)
            exporter._load()
            mock_trtllm_runner.assert_called_once()

    @pytest.mark.run_only_on("GPU")
    @patch("os.path.exists")
    @patch("builtins.open", create=True)
    @patch("json.load")
    def test_load_other_model(self, mock_json_load, mock_open, mock_exists, model_dir):
        from nemo_export.tensorrt_mm_exporter import TensorRTMMExporter

        mock_exists.return_value = True
        mock_json_load.return_value = {"builder_config": {"model_type": "neva"}}
        mock_open.return_value.__enter__ = lambda x: x
        mock_open.return_value.__exit__ = lambda x, y, z, w: None

        with patch("nemo_export.tensorrt_mm_exporter.MultimodalModelRunner") as mock_multimodal_runner:
            exporter = TensorRTMMExporter(model_dir, load_model=False)
            exporter._load()
            mock_multimodal_runner.assert_called_once()
