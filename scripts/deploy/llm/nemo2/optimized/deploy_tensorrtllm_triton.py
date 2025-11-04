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
import json
import logging
import os
import sys
from pathlib import Path
from typing import Optional

from nemo_deploy import DeployPyTriton

LOGGER = logging.getLogger("NeMo")


class UsageError(Exception):
    pass


trt_llm_supported = True
try:
    from nemo_export.tensorrt_llm import TensorRTLLM
except Exception as e:
    LOGGER.warning(f"Cannot import the TensorRTLLM exporter, it will not be available. {type(e).__name__}: {e}")
    trt_llm_supported = False


def get_args(argv):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Deploy nemo models to Triton",
    )
    parser.add_argument("-nc", "--nemo-checkpoint", type=str, help="Source NeMo 2.0 checkpoint folder")
    parser.add_argument("-mt", "--model-type", type=str, help="Type of the TensorRT-LLM model.")
    parser.add_argument(
        "-tmn",
        "--triton-model-name",
        required=True,
        type=str,
        help="Name for the service",
    )
    parser.add_argument(
        "-tmv",
        "--triton-model-version",
        default=1,
        type=int,
        help="Version for the service",
    )
    parser.add_argument(
        "-trp",
        "--triton-port",
        default=8000,
        type=int,
        help="Port for the Triton server to listen for requests",
    )
    parser.add_argument(
        "-tha",
        "--triton-http-address",
        default="0.0.0.0",
        type=str,
        help="HTTP address for the Triton server",
    )
    parser.add_argument(
        "-trt",
        "--triton-request-timeout",
        default=60,
        type=int,
        help="Timeout in seconds for Triton server",
    )
    parser.add_argument(
        "-tmr",
        "--triton-model-repository",
        default=None,
        type=str,
        help="Folder for the trt-llm conversion",
    )
    parser.add_argument(
        "-ng",
        "--num-gpus",
        default=None,
        type=int,
        help="Number of GPUs for the deployment",
    )
    parser.add_argument(
        "-tps",
        "--tensor-parallelism-size",
        default=1,
        type=int,
        help="Tensor parallelism size",
    )
    parser.add_argument(
        "-pps",
        "--pipeline-parallelism-size",
        default=1,
        type=int,
        help="Pipeline parallelism size",
    )
    parser.add_argument(
        "-dt",
        "--dtype",
        choices=["bfloat16", "float16", "fp8", "int8"],
        default="bfloat16",
        type=str,
        help="Data type of the model on TensorRT-LLM",
    )
    parser.add_argument(
        "-mil",
        "--max-input-len",
        default=256,
        type=int,
        help="Max input length of the model",
    )
    parser.add_argument(
        "-mol",
        "--max-output-len",
        default=256,
        type=int,
        help="Max output length of the model",
    )
    parser.add_argument(
        "-mbs",
        "--max-batch-size",
        default=8,
        type=int,
        help="Max batch size of the model",
    )
    parser.add_argument("-mnt", "--max-num-tokens", default=None, type=int, help="Max number of tokens")
    parser.add_argument(
        "-msl",
        "--max-seq-len",
        default=None,
        type=int,
        help="Maximum number of sequence length",
    )
    parser.add_argument(
        "-mp",
        "--multiple-profiles",
        default=False,
        action="store_true",
        help="Multiple profiles",
    )
    parser.add_argument(
        "-ont",
        "--opt-num-tokens",
        default=None,
        type=int,
        help="Optimum number of tokens",
    )
    parser.add_argument(
        "-gap",
        "--gpt-attention-plugin",
        default="auto",
        type=str,
        help="Data type of GPT attention plugin",
    )
    parser.add_argument("-gp", "--gemm-plugin", default="auto", type=str, help="Data type of GPT plugin")
    parser.add_argument(
        "-npkc",
        "--no-paged-kv-cache",
        default=False,
        action="store_true",
        help="Enable paged kv cache.",
    )
    parser.add_argument(
        "-drip",
        "--disable-remove-input-padding",
        default=False,
        action="store_true",
        help="Disables the remove input padding option.",
    )
    parser.add_argument(
        "-upe",
        "--use-parallel-embedding",
        default=False,
        action="store_true",
        help="Use parallel embedding feature of TensorRT-LLM.",
    )
    parser.add_argument(
        "-mbm",
        "--multi-block-mode",
        default=False,
        action="store_true",
        help="Split long kv sequence into multiple blocks (applied to generation MHA kernels). \
                        It is beneifical when batchxnum_heads cannot fully utilize GPU. \
                        Only available when using c++ runtime.",
    )
    parser.add_argument(
        "--use-lora-plugin",
        nargs="?",
        const=None,
        choices=["float16", "float32", "bfloat16"],
        help="Activates the lora plugin which enables embedding sharing.",
    )
    parser.add_argument(
        "--lora-target-modules",
        nargs="+",
        default=None,
        choices=[
            "attn_qkv",
            "attn_q",
            "attn_k",
            "attn_v",
            "attn_dense",
            "mlp_h_to_4h",
            "mlp_gate",
            "mlp_4h_to_h",
        ],
        help="Add lora in which modules. Only be activated when use_lora_plugin is enabled.",
    )
    parser.add_argument(
        "--max-lora-rank",
        type=int,
        default=64,
        help="maximum lora rank for different lora modules. It is used to compute the workspace size of lora plugin.",
    )
    parser.add_argument(
        "-lc",
        "--lora-ckpt",
        default=None,
        type=str,
        nargs="+",
        help="The checkpoint list of LoRA weights",
    )
    parser.add_argument(
        "-ucr",
        "--use-cpp-runtime",
        default=False,
        action="store_true",
        help="Use TensorRT LLM C++ runtime",
    )
    parser.add_argument(
        "-srs",
        "--start-rest-service",
        default=False,
        type=bool,
        help="Starts the REST service for OpenAI API support",
    )
    parser.add_argument(
        "-sha",
        "--service-http-address",
        default="0.0.0.0",
        type=str,
        help="HTTP address for the REST Service",
    )
    parser.add_argument(
        "-sp",
        "--service-port",
        default=8080,
        type=int,
        help="Port for the REST Service",
    )
    parser.add_argument(
        "-ofr",
        "--openai-format-response",
        default=False,
        type=bool,
        help="Return the response from PyTriton server in OpenAI compatible format",
    )
    parser.add_argument(
        "-dm",
        "--debug-mode",
        default=False,
        action="store_true",
        help="Enable debug mode",
    )
    parser.add_argument(
        "-fp8",
        "--export-fp8-quantized",
        default="auto",
        type=str,
        help="Enables exporting to a FP8-quantized TRT LLM checkpoint",
    )
    parser.add_argument(
        "-kv_fp8",
        "--use-fp8-kv-cache",
        default="auto",
        type=str,
        help="Enables exporting with FP8-quantizatized KV-cache",
    )
    args = parser.parse_args(argv)

    def str_to_bool(name: str, s: str, optional: bool = False) -> Optional[bool]:
        s = s.lower()
        true_strings = ["true", "1"]
        false_strings = ["false", "0"]
        if s in true_strings:
            return True
        if s in false_strings:
            return False
        if optional and s == "auto":
            return None
        raise UsageError(f"Invalid boolean value for argument --{name}: '{s}'")

    args.export_fp8_quantized = str_to_bool("export-fp8-quantized", args.export_fp8_quantized, optional=True)
    args.use_fp8_kv_cache = str_to_bool("use-fp8-kv-cache", args.use_fp8_kv_cache, optional=True)
    return args


def store_args_to_json(args):
    """Stores user defined arg values relevant for REST API in config.json.

    Gets called only when args.start_rest_service is True.
    """
    args_dict = {
        "triton_service_ip": args.triton_http_address,
        "triton_service_port": args.triton_port,
        "triton_request_timeout": args.triton_request_timeout,
        "openai_format_response": args.openai_format_response,
    }
    with open("nemo/deploy/service/config.json", "w") as f:
        json.dump(args_dict, f)


def get_trtllm_deployable(args):
    if args.triton_model_repository is None:
        trt_llm_path = "/tmp/trt_llm_model_dir/"
        LOGGER.info(
            "/tmp/trt_llm_model_dir/ path will be used as the TensorRT LLM folder. "
            "Please set the --triton_model_repository parameter if you'd like to use a path that already "
            "includes the TensorRT LLM model files."
        )
        Path(trt_llm_path).mkdir(parents=True, exist_ok=True)
    else:
        trt_llm_path = args.triton_model_repository

    checkpoint_missing = args.nemo_checkpoint is None
    if checkpoint_missing and args.triton_model_repository is None:
        raise ValueError(
            "The provided model repository is not a valid TensorRT-LLM model "
            "directory. Please provide a --nemo_checkpoint."
        )

    if checkpoint_missing and not os.path.isdir(args.triton_model_repository):
        raise ValueError(
            "The provided model repository is not a valid TensorRT-LLM model "
            "directory. Please provide a --nemo_checkpoint."
        )

    if not checkpoint_missing and args.model_type is None:
        raise ValueError("Model type is required to be defined if a nemo checkpoint is provided.")

    trt_llm_exporter = TensorRTLLM(
        model_dir=trt_llm_path,
        lora_ckpt_list=args.lora_ckpt,
        load_model=(args.nemo_checkpoint is None),
        use_python_runtime=(not args.use_cpp_runtime),
        multi_block_mode=args.multi_block_mode,
    )

    if args.nemo_checkpoint is not None:
        try:
            LOGGER.info("Export operation will be started to export the nemo checkpoint to TensorRT-LLM.")
            trt_llm_exporter.export(
                nemo_checkpoint_path=args.nemo_checkpoint,
                model_type=args.model_type,
                tensor_parallelism_size=args.tensor_parallelism_size,
                pipeline_parallelism_size=args.pipeline_parallelism_size,
                max_input_len=args.max_input_len,
                max_output_len=args.max_output_len,
                max_batch_size=args.max_batch_size,
                max_num_tokens=args.max_num_tokens,
                opt_num_tokens=args.opt_num_tokens,
                max_seq_len=args.max_seq_len,
                use_parallel_embedding=args.use_parallel_embedding,
                paged_kv_cache=(not args.no_paged_kv_cache),
                remove_input_padding=(not args.disable_remove_input_padding),
                dtype=args.dtype,
                use_lora_plugin=args.use_lora_plugin,
                lora_target_modules=args.lora_target_modules,
                max_lora_rank=args.max_lora_rank,
                multiple_profiles=args.multiple_profiles,
                gpt_attention_plugin=args.gpt_attention_plugin,
                gemm_plugin=args.gemm_plugin,
                fp8_quantized=args.export_fp8_quantized,
                fp8_kvcache=args.use_fp8_kv_cache,
            )
        except Exception as error:
            raise RuntimeError("An error has occurred during the model export. Error message: " + str(error))

    return trt_llm_exporter


def nemo_deploy(argv):
    args = get_args(argv)

    if args.debug_mode:
        loglevel = logging.DEBUG
    else:
        loglevel = logging.INFO

    LOGGER.setLevel(loglevel)
    LOGGER.info("Logging level set to {}".format(loglevel))
    LOGGER.info(args)

    if not trt_llm_supported:
        raise ValueError("TensorRT-LLM engine is not supported in this environment.")

    triton_deployable = get_trtllm_deployable(args)

    try:
        nm = DeployPyTriton(
            model=triton_deployable,
            triton_model_name=args.triton_model_name,
            triton_model_version=args.triton_model_version,
            max_batch_size=args.max_batch_size,
            http_port=args.triton_port,
            address=args.triton_http_address,
        )

        LOGGER.info("Triton deploy function will be called.")
        nm.deploy()
        nm.run()
    except Exception as error:
        LOGGER.error("Error message has occurred during deploy function. Error message: " + str(error))
        return

    try:
        LOGGER.info("Model serving on Triton is will be started.")
        nm.serve()
    except Exception as error:
        LOGGER.error("Error message has occurred during deploy function. Error message: " + str(error))
        return
    LOGGER.info("Model serving will be stopped.")
    nm.stop()


if __name__ == "__main__":
    nemo_deploy(sys.argv[1:])
