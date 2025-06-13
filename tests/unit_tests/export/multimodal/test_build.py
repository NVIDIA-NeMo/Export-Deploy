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
import unittest
from unittest.mock import MagicMock, patch, mock_open

import pytest
import torch


@pytest.mark.run_only_on("GPU")
class TestBuild(unittest.TestCase):
    @pytest.mark.run_only_on("GPU")
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.mock_config = {
            "mm_cfg": {
                "vision_encoder": {
                    "from_pretrained": "test_model",
                    "hidden_size": 768,
                },
                "mm_mlp_adapter_type": "linear",
                "lita": {"sample_frames": 8},
            },
            "hidden_size": 4096,
            "data": {"num_frames": 4},
        }
        self.mock_weights = {
            "model.embedding.word_embeddings.adapter_layer.mm_projector_adapter.mm_projector.weight": torch.randn(
                4096, 768
            ),
            "model.embedding.word_embeddings.adapter_layer.mm_projector_adapter.mm_projector.bias": torch.randn(
                4096
            ),
        }

    @pytest.mark.run_only_on("GPU")
    def tearDown(self):
        # Clean up temporary directory
        if os.path.exists(self.temp_dir):
            for root, dirs, files in os.walk(self.temp_dir, topdown=False):
                for name in files:
                    os.remove(os.path.join(root, name))
                for name in dirs:
                    os.rmdir(os.path.join(root, name))
            os.rmdir(self.temp_dir)

    @pytest.mark.run_only_on("GPU")
    @patch("nemo_export.multimodal.build.TensorRTLLM")
    def test_build_trtllm_engine(self, mock_trtllm):
        # Test basic functionality
        mock_exporter = MagicMock()
        mock_trtllm.return_value = mock_exporter

        from nemo_export.multimodal.build import build_trtllm_engine

        build_trtllm_engine(
            model_dir=self.temp_dir,
            visual_checkpoint_path="test_path",
            model_type="neva",
            tensor_parallelism_size=1,
            max_input_len=256,
            max_output_len=256,
            max_batch_size=1,
            max_multimodal_len=1024,
            dtype="bfloat16",
        )

        mock_exporter.export.assert_called_once()

    @pytest.mark.run_only_on("GPU")
    @patch("nemo_export.multimodal.build.MLLaMAForCausalLM")
    @patch("nemo_export.multimodal.build.build_trtllm")
    def test_build_mllama_trtllm_engine(self, mock_build_trtllm, mock_mllama):
        # Test basic functionality
        mock_model = MagicMock()
        mock_mllama.from_hugging_face.return_value = mock_model
        mock_build_trtllm.return_value = MagicMock()

        from nemo_export.multimodal.build import build_mllama_trtllm_engine

        build_mllama_trtllm_engine(
            model_dir=self.temp_dir,
            hf_model_path="test_path",
            tensor_parallelism_size=1,
            max_input_len=256,
            max_output_len=256,
            max_batch_size=1,
            max_multimodal_len=1024,
            dtype="bfloat16",
        )

        mock_mllama.from_hugging_face.assert_called_once()
        mock_build_trtllm.assert_called_once()

    @pytest.mark.run_only_on("GPU")
    @patch("nemo_export.multimodal.build.torch.onnx.export")
    @patch("nemo_export.multimodal.build.os.makedirs")
    def test_export_visual_wrapper_onnx(self, mock_makedirs, mock_onnx_export):
        from nemo_export.multimodal.build import export_visual_wrapper_onnx

        mock_wrapper = MagicMock()
        mock_input = MagicMock()

        export_visual_wrapper_onnx(mock_wrapper, mock_input, self.temp_dir)

        mock_makedirs.assert_called_once_with(f"{self.temp_dir}/onnx", exist_ok=True)
        mock_onnx_export.assert_called_once()

    @pytest.mark.run_only_on("GPU")
    @patch("nemo_export.multimodal.build.yaml.dump")
    @patch("nemo_export.multimodal.build.Builder")
    @patch("nemo_export.multimodal.build.trt.Builder")
    @patch("nemo_export.multimodal.build.shutil.rmtree")
    @patch("builtins.open", new_callable=mock_open)
    def test_build_trt_engine(
        self, mock_file, mock_rmtree, mock_trt_builder, mock_builder, mock_yaml_dump
    ):
        from nemo_export.multimodal.build import build_trt_engine

        # Setup mocks
        mock_builder_instance = MagicMock()
        mock_trt_builder.return_value = mock_builder_instance
        mock_builder.return_value.create_builder_config.return_value.trt_builder_config = MagicMock()

        # Mock network and parser
        mock_network = MagicMock()
        mock_parser = MagicMock()
        mock_parser.parse.return_value = True
        mock_parser.num_errors = 0

        mock_builder_instance.create_network.return_value = mock_network
        mock_builder_instance.create_optimization_profile.return_value = MagicMock()
        mock_builder_instance.build_serialized_network.return_value = b"engine_data"

        # Mock input tensor
        mock_input = MagicMock()
        mock_input.name = "input"
        mock_network.get_input.return_value = mock_input
        mock_network.num_inputs = 1

        with patch(
            "nemo_export.multimodal.build.trt.OnnxParser", return_value=mock_parser
        ):
            build_trt_engine(
                model_type="neva",
                input_sizes=[3, 224, 224],
                output_dir=self.temp_dir,
                vision_max_batch_size=1,
                nemo_config=self.mock_config,
            )

        mock_rmtree.assert_called_once()

    @pytest.mark.run_only_on("GPU")
    @patch("nemo_export.multimodal.build.build_trt_engine")
    @patch("nemo_export.multimodal.build.export_visual_wrapper_onnx")
    @patch("nemo_export.multimodal.build.AutoModel.from_pretrained")
    @patch("nemo_export.multimodal.build.load_nemo_model")
    @patch("nemo_export.multimodal.build.torch.cuda.is_available", return_value=True)
    def test_build_neva_engine(
        self,
        mock_cuda,
        mock_load_nemo,
        mock_auto_model,
        mock_export_onnx,
        mock_build_trt,
    ):
        from nemo_export.multimodal.build import build_neva_engine

        # Setup mocks
        mock_load_nemo.return_value = (self.mock_weights, self.mock_config, None)

        mock_encoder = MagicMock()
        mock_encoder.vision_model = MagicMock()
        mock_encoder.config.vision_config.image_size = 224
        mock_encoder.config.torch_dtype = torch.bfloat16
        mock_auto_model.return_value = mock_encoder

        build_neva_engine(
            model_type="neva",
            model_dir=self.temp_dir,
            visual_checkpoint_path="test_checkpoint.nemo",
            vision_max_batch_size=1,
        )

        mock_load_nemo.assert_called_once()
        mock_auto_model.assert_called_once()
        mock_export_onnx.assert_called_once()
        mock_build_trt.assert_called_once()

    @pytest.mark.run_only_on("GPU")
    @patch("nemo_export.multimodal.build.build_trt_engine")
    @patch("nemo_export.multimodal.build.export_visual_wrapper_onnx")
    @patch("nemo_export.multimodal.build.AutoModel.from_pretrained")
    @patch("nemo_export.multimodal.build.tarfile.open")
    @patch("nemo_export.multimodal.build.torch.cuda.is_available", return_value=True)
    def test_build_video_neva_engine(
        self, mock_cuda, mock_tarfile, mock_auto_model, mock_export_onnx, mock_build_trt
    ):
        from nemo_export.multimodal.build import build_video_neva_engine

        # Setup mocks
        mock_tar = MagicMock()
        mock_tarfile.return_value.__enter__.return_value = mock_tar
        mock_tar.extractfile.side_effect = [
            mock_open(
                read_data="mm_cfg:\n  vision_encoder:\n    from_pretrained: test\n    hidden_size: 768\n  mm_mlp_adapter_type: linear\nhidden_size: 4096\ndata:\n  num_frames: 4"
            )().read(),
            self.mock_weights,
        ]

        mock_encoder = MagicMock()
        mock_encoder.vision_model = MagicMock()
        mock_encoder.config.vision_config.image_size = 224
        mock_encoder.config.torch_dtype = torch.bfloat16
        mock_auto_model.return_value = mock_encoder

        with patch(
            "nemo_export.multimodal.build.yaml.safe_load", return_value=self.mock_config
        ):
            with patch(
                "nemo_export.multimodal.build.torch.load",
                return_value=self.mock_weights,
            ):
                build_video_neva_engine(
                    model_dir=self.temp_dir,
                    visual_checkpoint_path="test_checkpoint.nemo",
                    vision_max_batch_size=1,
                )

        mock_auto_model.assert_called_once()
        mock_export_onnx.assert_called_once()
        mock_build_trt.assert_called_once()

    @pytest.mark.run_only_on("GPU")
    @patch("nemo_export.multimodal.build.VisionEngineBuilder")
    @patch("nemo_export.multimodal.build.AutoProcessor.from_pretrained")
    @patch("nemo_export.multimodal.build.shutil.copy2")
    @patch("nemo_export.multimodal.build.os.listdir")
    def test_build_mllama_visual_engine(
        self, mock_listdir, mock_copy, mock_processor, mock_vision_builder
    ):
        from nemo_export.multimodal.build import build_mllama_visual_engine

        # Setup mocks
        mock_processor_instance = MagicMock()
        mock_processor.return_value = mock_processor_instance
        mock_listdir.return_value = ["file1.json", "file2.txt"]

        mock_builder_instance = MagicMock()
        mock_vision_builder.return_value = mock_builder_instance

        build_mllama_visual_engine(
            model_dir=self.temp_dir,
            hf_model_path="test_path",
            vision_max_batch_size=1,
        )

        mock_processor.assert_called_once()
        mock_processor_instance.save_pretrained.assert_called_once()
        mock_builder_instance.build.assert_called_once()

    @pytest.mark.run_only_on("GPU")
    @patch("nemo_export.multimodal.build.build_neva_engine")
    @patch("nemo_export.multimodal.build.build_video_neva_engine")
    def test_build_visual_engine(self, mock_build_video_neva, mock_build_neva):
        from nemo_export.multimodal.build import build_visual_engine

        # Test neva model
        build_visual_engine(
            model_dir=self.temp_dir,
            visual_checkpoint_path="test_path",
            model_type="neva",
            vision_max_batch_size=1,
        )
        mock_build_neva.assert_called_once()

        # Test video-neva model
        build_visual_engine(
            model_dir=self.temp_dir,
            visual_checkpoint_path="test_path",
            model_type="video-neva",
            vision_max_batch_size=1,
        )
        mock_build_video_neva.assert_called_once()

        # Test invalid model type
        with self.assertRaises(RuntimeError):
            build_visual_engine(
                model_dir=self.temp_dir,
                visual_checkpoint_path="test_path",
                model_type="invalid",
                vision_max_batch_size=1,
            )

    @pytest.mark.run_only_on("GPU")
    @patch("nemo_export.multimodal.build.tarfile.open")
    @patch("nemo_export.multimodal.build.torch.save")
    @patch("nemo_export.multimodal.build.torch.load")
    @patch("nemo_export.multimodal.build.os.path.exists")
    def test_extract_lora_ckpt(
        self, mock_exists, mock_torch_load, mock_torch_save, mock_tarfile
    ):
        from nemo_export.multimodal.build import extract_lora_ckpt

        # Test with direct model_weights.ckpt
        def mock_exists_side_effect(path):
            return (
                "model_weights.ckpt" in path and "mp_rank_00" not in path
            ) or "model_config.yaml" in path

        mock_exists.side_effect = mock_exists_side_effect
        mock_torch_load.return_value = self.mock_weights

        result = extract_lora_ckpt("test_lora_path", self.temp_dir)

        self.assertTrue(result.endswith("llm_lora.nemo"))
        mock_torch_load.assert_called()
        mock_torch_save.assert_called()

    @pytest.mark.run_only_on("GPU")
    @patch("nemo_export.multimodal.build.build_mllama_trtllm_engine")
    @patch("nemo_export.multimodal.build.build_mllama_visual_engine")
    @patch("nemo_export.multimodal.build.llm.export_ckpt")
    def test_build_mllama_engine(
        self, mock_export_ckpt, mock_build_visual, mock_build_trtllm
    ):
        from nemo_export.multimodal.build import build_mllama_engine

        build_mllama_engine(
            model_dir=self.temp_dir,
            checkpoint_path="test_checkpoint",
            tensor_parallelism_size=1,
            max_input_len=256,
            max_output_len=256,
            max_batch_size=1,
            max_multimodal_len=1024,
            dtype="bfloat16",
        )

        mock_export_ckpt.assert_called_once()
        mock_build_visual.assert_called_once()
        mock_build_trtllm.assert_called_once()


if __name__ == "__main__":
    unittest.main()
