# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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
import uvicorn

from nemo_deploy import DeployPyTriton

LOGGER = logging.getLogger("NeMo")
# Add a stream handler if none exists
if not LOGGER.hasHandlers():
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("%(asctime)s %(name)s %(levelname)s: %(message)s")
    handler.setFormatter(formatter)
    LOGGER.addHandler(handler)

megatron_llm_supported = True
try:
    from nemo_deploy.llm.megatronllm_deployable import MegatronLLMDeployableNemo2
except Exception as e:
    LOGGER.warning(f"Cannot import MegatronLLMDeployable, it will not be available. {type(e).__name__}: {e}")
    megatron_llm_supported = False


def get_args(argv):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Deploy nemo models to Triton",
    )
    parser.add_argument("-nc", "--nemo-checkpoint", type=str, help="Source NeMo 2.0 checkpoint folder")
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
        "-sp",
        "--server-port",
        default=8080,
        type=int,
        help="Port for the REST server to listen for requests",
    )
    parser.add_argument(
        "-sa",
        "--server-address",
        default="0.0.0.0",
        type=str,
        help="HTTP address for the REST server",
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
        "-ng",
        "--num-gpus",
        default=None,
        type=int,
        help="Number of GPUs for the deployment",
    )
    parser.add_argument(
        "-nn",
        "--num-nodes",
        default=None,
        type=int,
        help="Number of Nodes for the deployment",
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
        "-nlfps",
        "--num-layers-in-first-pipeline-stage",
        default=None,
        type=int,
        help="Number of layers in the first pipeline stage",
    )
    parser.add_argument(
        "-nllps",
        "--num-layers-in-last-pipeline-stage",
        default=None,
        type=int,
        help="Number of layers in the last pipeline stage",
    )
    parser.add_argument(
        "-cps",
        "--context-parallel-size",
        default=1,
        type=int,
        help="Context parallelism size",
    )
    parser.add_argument(
        "-emps",
        "--expert-model-parallel-size",
        default=1,
        type=int,
        help="Distributes MoE Experts across sub data parallel dimension.",
    )
    parser.add_argument(
        "-eps",
        "--account-for-embedding-in-pipeline-split",
        default=False,
        action="store_true",
        help="Account for embedding in the pipeline split",
    )
    parser.add_argument(
        "-lps",
        "--account-for-loss-in-pipeline-split",
        default=False,
        action="store_true",
        help="Account for loss in the pipeline split",
    )
    parser.add_argument(
        "-mbs",
        "--max-batch-size",
        default=8,
        type=int,
        help="Max batch size of the model",
    )
    parser.add_argument(
        "-dm",
        "--debug-mode",
        default=False,
        action="store_true",
        help="Enable debug mode",
    )
    parser.add_argument(
        "-fd",
        "--enable-flash-decode",
        default=False,
        action="store_true",
        help="Enable flash decoding",
    )
    parser.add_argument(
        "-cg",
        "--enable-cuda-graphs",
        default=False,
        action="store_true",
        help="Enable CUDA graphs",
    )
    parser.add_argument(
        "-lc",
        "--legacy-ckpt",
        action="store_true",
        help="Load checkpoint saved with TE < 1.14",
    )
    parser.add_argument(
        "-imsl",
        "--inference-max-seq-length",
        default=4096,
        type=int,
        help="Max sequence length for inference",
    )
    parser.add_argument(
        "-mb",
        "--micro-batch-size",
        type=int,
        default=None,
        help="Micro batch size for model execution",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=None,
        help="Random seed for reproducible inference",
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

    if not megatron_llm_supported:
        raise ValueError("MegatronLLMDeployable is not supported in this environment.")

    if args.nemo_checkpoint is None:
        raise ValueError("In-Framework deployment requires a checkpoint folder.")

    model_config_kwargs = {
        "account_for_embedding_in_pipeline_split": args.account_for_embedding_in_pipeline_split,
        "account_for_loss_in_pipeline_split": args.account_for_loss_in_pipeline_split,
    }

    if args.num_layers_in_first_pipeline_stage is not None:
        model_config_kwargs["num_layers_in_first_pipeline_stage"] = args.num_layers_in_first_pipeline_stage

    if args.num_layers_in_last_pipeline_stage is not None:
        model_config_kwargs["num_layers_in_last_pipeline_stage"] = args.num_layers_in_last_pipeline_stage

    model = MegatronLLMDeployableNemo2(
        num_devices=args.num_gpus,
        num_nodes=args.num_nodes,
        nemo_checkpoint_filepath=args.nemo_checkpoint,
        tensor_model_parallel_size=args.tensor_parallelism_size,
        pipeline_model_parallel_size=args.pipeline_parallelism_size,
        inference_max_seq_length=args.inference_max_seq_length,
        context_parallel_size=args.context_parallel_size,
        max_batch_size=args.max_batch_size,
        enable_flash_decode=args.enable_flash_decode,
        enable_cuda_graphs=args.enable_cuda_graphs,
        legacy_ckpt=args.legacy_ckpt,
        micro_batch_size=args.micro_batch_size,
        random_seed=args.random_seed,
        **model_config_kwargs,
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
                nm.run()
            except Exception as error:
                LOGGER.error("Error message has occurred during deploy function. Error message: " + str(error))
                return

            try:
                # start fastapi server which acts as a proxy to Pytriton server. Applies to PyTriton backend only.
                try:
                    LOGGER.info("REST service will be started.")
                    uvicorn.run(
                        "nemo_deploy.service.fastapi_interface_to_pytriton:app",
                        host=args.server_address,
                        port=args.server_port,
                        reload=True,
                    )
                except Exception as error:
                    LOGGER.error("Error message has occurred during REST service start. Error message: " + str(error))
                LOGGER.info("Model serving on Triton will be started.")
                nm.serve()
            except Exception as error:
                LOGGER.error("Error message has occurred during deploy function. Error message: " + str(error))

            torch.distributed.broadcast(torch.tensor([1], dtype=torch.long, device="cuda"), src=0)

            LOGGER.info("Model serving will be stopped.")
            nm.stop()
        elif torch.distributed.get_rank() > 0:
            model.generate_other_ranks()

    else:
        LOGGER.info("Torch distributed wasn't initialized.")


if __name__ == "__main__":
    nemo_deploy(sys.argv[1:])
