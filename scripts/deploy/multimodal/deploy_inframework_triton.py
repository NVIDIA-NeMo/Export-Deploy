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
import logging
import sys

import torch

from nemo_deploy import DeployPyTriton

LOGGER = logging.getLogger("NeMo")
# Add a stream handler if none exists
if not LOGGER.hasHandlers():
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    handler.setFormatter(formatter)
    LOGGER.addHandler(handler)

multimodal_supported = True
try:
    from nemo_deploy.multimodal.nemo_multimodal_deployable import NeMoMultimodalDeployable
except Exception as e:
    LOGGER.warning(f"Cannot import NeMoMultimodalDeployable, it will not be available. {type(e).__name__}: {e}")
    multimodal_supported = False


def get_args(argv):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Deploy nemo multimodal models to Triton",
    )
    parser.add_argument("-nc", "--nemo_checkpoint", type=str, help="Source .nemo file")
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
        "-ng",
        "--num_gpus",
        default=None,
        type=int,
        help="Number of GPUs for the deployment",
    )
    parser.add_argument(
        "-nn",
        "--num_nodes",
        default=None,
        type=int,
        help="Number of Nodes for the deployment",
    )
    parser.add_argument(
        "-tps",
        "--tensor_parallelism_size",
        default=1,
        type=int,
        help="Tensor parallelism size",
    )
    parser.add_argument(
        "-pps",
        "--pipeline_parallelism_size",
        default=1,
        type=int,
        help="Pipeline parallelism size",
    )

    parser.add_argument(
        "-mbs",
        "--max_batch_size",
        default=4,
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
    parser.add_argument(
        "-pd",
        "--params_dtype",
        default="bfloat16",
        type=str,
        choices=["float16", "bfloat16", "float32"],
        help="Data type for model parameters",
    )
    parser.add_argument(
        "-ibts",
        "--inference_batch_times_seqlen_threshold",
        default=1000,
        type=int,
        help="Inference batch times sequence length threshold",
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

    if not multimodal_supported:
        raise ValueError("NeMoMultimodalDeployable is not supported in this environment.")

    if args.nemo_checkpoint is None:
        raise ValueError("In-Framework deployment requires a checkpoint folder.")

    # Convert dtype string to torch dtype
    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    params_dtype = dtype_map[args.params_dtype]

    model = NeMoMultimodalDeployable(
        num_devices=args.num_gpus,
        num_nodes=args.num_nodes,
        nemo_checkpoint_filepath=args.nemo_checkpoint,
        tensor_model_parallel_size=args.tensor_parallelism_size,
        pipeline_model_parallel_size=args.pipeline_parallelism_size,
        params_dtype=params_dtype,
        inference_batch_times_seqlen_threshold=args.inference_batch_times_seqlen_threshold,
    )

    if torch.distributed.is_initialized():
        if torch.distributed.get_rank() == 0:
            try:
                nm = DeployPyTriton(
                    model=model,
                    triton_model_name=args.triton_model_name,
                    triton_model_version=args.triton_model_version,
                    max_batch_size=args.max_batch_size,
                    http_port=args.triton_port,
                    address=args.triton_http_address,
                )

                LOGGER.info("Triton deploy function will be called.")
                nm.deploy()
            except Exception as error:
                LOGGER.error("Error message has occurred during deploy function. Error message: " + str(error))
                return

            try:
                LOGGER.info("Model serving on Triton will be started.")
                nm.serve()
            except Exception as error:
                LOGGER.error("Error message has occurred during deploy function. Error message: " + str(error))
                return

            torch.distributed.broadcast(torch.tensor([1], dtype=torch.long, device="cuda"), src=0)

            LOGGER.info("Model serving will be stopped.")
            nm.stop()
        elif torch.distributed.get_rank() > 0:
            torch.distributed.broadcast(torch.tensor([1], dtype=torch.long, device="cuda"), src=0)

    else:
        LOGGER.info("Torch distributed wasn't initialized.")


if __name__ == "__main__":
    nemo_deploy(sys.argv[1:])
