# Copyright (c) 2023-2024, NVIDIA CORPORATION.  All rights reserved.
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
import logging
import sys

from nemo_deploy import DeployPyTriton

# Configure the NeMo logger to look the same as vLLM
logging.basicConfig(
    format="%(levelname)s %(asctime)s %(filename)s:%(lineno)d] %(message)s",
    datefmt="%m-%d %H:%M:%S",
)
LOGGER = logging.getLogger("NeMo")

try:
    from nemo_export.vllm_exporter import vLLMExporter
except Exception as e:
    LOGGER.error(f"Cannot import the vLLM exporter. {type(e).__name__}: {e}")
    sys.exit(1)


def get_args(argv):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Export Megatron-Bridge or Hugging Face models to vLLM and deploy them on Triton",
    )
    parser.add_argument(
        "-mpi",
        "--model_path_id",
        required=True,
        type=str,
        help="Path of a Megatron-Bridge checkpoint or Hugging Face model ID or path.",
    )
    parser.add_argument(
        "-hfp",
        "--hf_model_id_path",
        type=str,
        help="Huggingface model path or id in case of Megatron-Bridge checkpoint does not contain the required metadata.",
    )
    parser.add_argument(
        "-mf",
        "--model_format",
        choices=["hf", "megatron_bridge"],
        default="hf",
        type=str,
        help="Format of the input checkpoint: 'hf' for Hugging Face, 'megatron_bridge' for Megatron-Bridge.",
    )
    parser.add_argument(
        "-t",
        "--tokenizer",
        type=str,
        default=None,
        help="Tokenizer file if it is not provided in the checkpoint.",
    )
    parser.add_argument(
        "-lc",
        "--lora_ckpt",
        default=[],
        type=str,
        nargs="+",
        help="List of LoRA checkpoints in HF format",
    )
    parser.add_argument(
        "-tps",
        "--tensor_parallelism_size",
        default=1,
        type=int,
        help="Tensor parallelism size",
    )
    parser.add_argument(
        "-dt",
        "--dtype",
        choices=["auto", "bfloat16", "float16", "float32"],
        default="auto",
        type=str,
        help="dtype of the model on vLLM",
    )
    parser.add_argument(
        "-q",
        "--quantization",
        choices=["awq", "gptq", "fp8"],
        default=None,
        help="Quantization method for vLLM.",
    )
    parser.add_argument(
        "-s",
        "--seed",
        default=0,
        type=int,
        help="Tensor parallelism size",
    )
    parser.add_argument(
        "-gmu",
        "--gpu_memory_utilization",
        default=0.9,
        type=float,
        help="GPU memory utilization percentage for vLLM.",
    )
    parser.add_argument(
        "-sp",
        "--swap_space",
        default=4,
        type=float,
        help="The size (GiB) of CPU memory per GPU to use as swap space.",
    )
    parser.add_argument(
        "-cog",
        "--cpu_offload_gb",
        default=0,
        type=float,
        help="The size (GiB) of CPU memory to use for offloading the model weights.",
    )
    parser.add_argument(
        "-ee",
        "--enforce_eager",
        default=False,
        action="store_true",
        help="Whether to enforce eager execution.",
    )
    parser.add_argument(
        "-mslc",
        "--max_seq_len_to_capture",
        default=8192,
        type=int,
        help="Maximum sequence len covered by CUDA graphs.",
    )
    parser.add_argument(
        "-tmn",
        "--triton_model_name",
        required=True,
        type=str,
        help="Name for the service",
    )
    parser.add_argument(
        "-tmv",
        "--triton_model_version",
        default=1,
        type=int,
        help="Version for the service",
    )
    parser.add_argument(
        "-trp",
        "--triton_port",
        default=8000,
        type=int,
        help="Port for the Triton server to listen for requests",
    )
    parser.add_argument(
        "-tha",
        "--triton_http_address",
        default="0.0.0.0",
        type=str,
        help="HTTP address for the Triton server",
    )
    parser.add_argument(
        "-mbs",
        "--max_batch_size",
        default=8,
        type=int,
        help="Max batch size of the model",
    )
    parser.add_argument(
        "-dm",
        "--debug_mode",
        default=False,
        action="store_true",
        help="Enable debug mode",
    )

    args = parser.parse_args(argv)
    return args


def nemo_deploy(argv):
    args = get_args(argv)

    if args.debug_mode:
        loglevel = logging.DEBUG
    else:
        loglevel = logging.INFO

    LOGGER.setLevel(loglevel)
    LOGGER.info("Logging level set to {}".format(loglevel))
    LOGGER.info(args)

    try:
        exporter = vLLMExporter()
        exporter.export(
            model_path_id=args.model_path_id,
            tokenizer=args.tokenizer,
            trust_remote_code=True,
            enable_lora=True if len(args.lora_ckpt) else False,
            tensor_parallel_size=args.tensor_parallelism_size,
            dtype=args.dtype,
            quantization=args.quantization,
            seed=args.seed,
            gpu_memory_utilization=args.gpu_memory_utilization,
            swap_space=args.swap_space,
            cpu_offload_gb=args.cpu_offload_gb,
            enforce_eager=args.enforce_eager,
            max_seq_len_to_capture=args.max_seq_len_to_capture,
            task="generate",
            model_format=args.model_format,
            hf_model_id=args.hf_model_id_path,
        )

        nm = DeployPyTriton(
            model=exporter,
            triton_model_name=args.triton_model_name,
            triton_model_version=args.triton_model_version,
            max_batch_size=args.max_batch_size,
            http_port=args.triton_port,
            address=args.triton_http_address,
        )

        LOGGER.info("Starting the Triton server...")
        nm.deploy()
        nm.serve()

        LOGGER.info("Stopping the Triton server...")
        nm.stop()

    except Exception as error:
        LOGGER.error("An error has occurred while setting up or serving the model. Error message: " + str(error))
        return


if __name__ == "__main__":
    nemo_deploy(sys.argv[1:])
