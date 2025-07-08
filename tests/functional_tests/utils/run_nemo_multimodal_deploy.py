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

from nemo.collections import llm, vlm

from nemo_deploy import DeployPyTriton
from nemo_deploy.multimodal import NemoQueryMultimodal
from nemo_export.tensorrt_mm_exporter import TensorRTMMExporter


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Deploy nemo multimodal models to Triton and test models",
    )
    parser.add_argument(
        "--hf_model",
        type=str,
        required=True,
    )
    parser.add_argument(
        "-mt",
        "--model_type",
        type=str,
        required=True,
        choices=["neva", "video-neva", "lita", "vila", "vita", "salm", "mllama"],
        help="Type of the model that is supported.",
    )
    return parser.parse_args()


def import_hf_model(args):
    if args.model_type == "mllama":
        model = vlm.MLlamaModel(vlm.MLlamaConfig11BInstruct())

    llm.import_ckpt(
        model=model,
        source=f"hf://{args.hf_model}",
        output_path="/tmp/nemo2_ckpt",
        overwrite=True,
    )


def run_inference_tests(args):
    model_dir = "/tmp/trt_llm_model_dir"
    exporter = TensorRTMMExporter(model_dir=model_dir, load_model=True)

    if args.model_type == "mllama":
        llm_model_type = "mllama"
    else:
        llm_model_type = "llama"

    exporter.export(
        visual_checkpoint_path="/tmp/nemo2_ckpt",
        processor_name=args.hf_model,
        model_type=args.model_type,
        llm_model_type=llm_model_type,
        max_multimodal_len=6404,
    )

    try:
        nm = DeployPyTriton(
            model=exporter,
            triton_model_name="mllama",
            max_batch_size=1,
        )

        nm.deploy()
        nm.run()

        nq = NemoQueryMultimodal(url="localhost:8000", model_name="mllama", model_type="mllama")

        output_deployed = nq.query(
            input_text="What is in this image?",
            input_media="tests/functional_tests/test_image.jpg",
            max_output_len=20,
        )

    finally:
        nm.stop()

    assert output_deployed is not None, "Output cannot be none."
    assert len(output_deployed) > 0, "Output should not be empty."
    assert len(output_deployed[0]) > 0, "Output should not be empty."
    assert isinstance(output_deployed[0][0], str), "Output should be a string."
    assert len(output_deployed[0][0]) > 0, "Output string should not be empty."


if __name__ == "__main__":
    args = get_args()
    import_hf_model(args)
    run_inference_tests(args)
