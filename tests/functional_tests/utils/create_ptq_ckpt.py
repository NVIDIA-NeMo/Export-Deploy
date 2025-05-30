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

import argparse

from nemo.collections import llm
from nemo.collections.llm.modelopt import ExportConfig, QuantizationConfig


def get_args():
    """Parse PTQ arguments."""
    parser = argparse.ArgumentParser(
        description="Create sample PTQ checkpoint"
    )
    parser.add_argument("--nemo_checkpoint", help="Source NeMo 2.0 checkpoint")
    parser.add_argument("--calibration_tp", type=int, default=1, help="TP size for calibration")
    parser.add_argument(
        "--inference_tp",
        type=int,
        default=1,
        help="TRT-LLM engine TP size. (Only used when `--export_format` is 'trtllm')",
    )
    parser.add_argument("--export_path", help="Path for the exported engine")
    parser.add_argument(
        "--export_format", default="trtllm", choices=["trtllm", "nemo", "hf"], help="Model format to export as"
    )
    parser.add_argument(
        "--algorithm",
        default="fp8",
        help="TensorRT-Model-Optimizer quantization algorithm",
    )
    parser.add_argument("--calibration_batch_size",  type=int, default=64, help="Calibration batch size")
    parser.add_argument(
        "--calibration_dataset_size", type=int, default=512, help="Size of calibration dataset"
    )
    parser.add_argument(
        "--calibration_dataset",
        default="cnn_dailymail",
        help='Calibration dataset to be used. Should be "wikitext", "cnn_dailymail" or path to a local .json file',
    )
    parser.add_argument(
        "--generate_sample", help="Generate sample model output after performing PTQ", action="store_true"
    )
    parser.add_argument("--legacy_ckpt", help="Load ckpt saved with TE < 1.14", action="store_true")
    args = parser.parse_args()

    return args


def main():
    """Example NeMo 2.0 Post Training Quantization workflow."""
    args = get_args()

    quantization_config = QuantizationConfig(
        algorithm=None if args.algorithm == "no_quant" else args.algorithm,
        calibration_dataset=args.calibration_dataset,
        calibration_dataset_size=args.calibration_dataset_size,
        calibration_batch_size=args.calibration_batch_size,
    )

    export_config = ExportConfig(
        export_format=args.export_format,
        path=args.export_path,
        inference_tp=args.inference_tp,
        generate_sample=args.generate_sample,
    )

    llm.ptq(
        model_path=args.nemo_checkpoint,
        export_config=export_config,
        calibration_tp=args.calibration_tp,
        devices=args.calibration_tp,
        quantization_config=quantization_config,
        legacy_ckpt=args.legacy_ckpt,
    )


if __name__ == "__main__":
    main()
