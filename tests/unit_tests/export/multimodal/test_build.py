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
from unittest.mock import MagicMock, mock_open, patch

import pytest


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
    def test_build_trt_engine(self, mock_file, mock_rmtree, mock_trt_builder, mock_builder, mock_yaml_dump):
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

        with patch("nemo_export.multimodal.build.trt.OnnxParser", return_value=mock_parser):
            build_trt_engine(
                model_type="neva",
                input_sizes=[3, 224, 224],
                output_dir=self.temp_dir,
                vision_max_batch_size=1,
                nemo_config=self.mock_config,
            )

        mock_rmtree.assert_called_once()

    @pytest.mark.run_only_on("GPU")
    @patch("nemo_export.multimodal.build.MultimodalEngineBuilder")
    @patch("nemo_export.multimodal.build.AutoProcessor.from_pretrained")
    @patch("nemo_export.multimodal.build.shutil.copy2")
    @patch("nemo_export.multimodal.build.os.listdir")
    def test_build_mllama_visual_engine(self, mock_listdir, mock_copy, mock_processor, mock_vision_builder):
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


if __name__ == "__main__":
    unittest.main()
