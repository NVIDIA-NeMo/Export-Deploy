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

import json
import logging
from typing import List, Optional

import numpy as np
import torch
from megatron.core.inference.common_inference_params import CommonInferenceParams
from nemo.collections.vlm.inference.base import generate, setup_model_and_tokenizer
from PIL.Image import Image

from nemo_deploy import ITritonDeployable
from nemo_deploy.utils import cast_output, ndarray2img, str_ndarray2list
from nemo_export_deploy_common.import_utils import MISSING_TRITON_MSG, UnavailableError, null_decorator

try:
    from pytriton.decorators import batch, first_value
    from pytriton.model_config import Tensor

    HAVE_TRITON = True
except (ImportError, ModuleNotFoundError):
    from unittest.mock import MagicMock

    HAVE_TRITON = False
    batch = MagicMock()
    first_value = MagicMock()
    Tensor = MagicMock()

    batch = null_decorator
    first_value = null_decorator

LOGGER = logging.getLogger("NeMo")


def dict_to_str(messages):
    """Serializes dict to str."""
    return json.dumps(messages)


class NeMoMultimodalDeployable(ITritonDeployable):
    """Triton inference server compatible deploy class for a NeMo multimodal model file.

    Args:
        nemo_checkpoint_filepath (str): path for the nemo checkpoint.
        num_devices (int): number of GPUs.
        num_nodes (int): number of nodes.
        tensor_model_parallel_size (int): tensor parallelism.
        pipeline_parallelism_size (int): pipeline parallelism.
        params_dtype (torch.dtype): data type for model parameters.
        inference_batch_times_seqlen_threshold (int): sequence threshold.
    """

    def __init__(
        self,
        num_devices: int = None,
        num_nodes: int = None,
        nemo_checkpoint_filepath: str = None,
        tensor_model_parallel_size: int = 1,
        pipeline_model_parallel_size: int = 1,
        params_dtype: torch.dtype = torch.bfloat16,
        inference_batch_times_seqlen_threshold: int = 1000,
    ):
        if not HAVE_TRITON:
            raise UnavailableError(MISSING_TRITON_MSG)

        self.nemo_checkpoint_filepath = nemo_checkpoint_filepath
        self.num_devices = num_devices
        self.num_nodes = num_nodes
        self.tensor_model_parallel_size = tensor_model_parallel_size
        self.pipeline_model_parallel_size = pipeline_model_parallel_size
        self.params_dtype = params_dtype
        self.inference_batch_times_seqlen_threshold = inference_batch_times_seqlen_threshold

        self.inference_wrapped_model, self.processor = setup_model_and_tokenizer(
            path=nemo_checkpoint_filepath,
            tp_size=tensor_model_parallel_size,
            pp_size=pipeline_model_parallel_size,
            params_dtype=params_dtype,
            inference_batch_times_seqlen_threshold=inference_batch_times_seqlen_threshold,
        )

    def generate(
        self,
        prompts: List[str],
        images: List[Image],
        inference_params: Optional[CommonInferenceParams] = None,
        max_batch_size: int = 4,
        random_seed: Optional[int] = None,
    ) -> dict:
        """Generates text based on the provided input prompts and images.

        Args:
            prompts (List[str]): A list of input strings.
            images (List[Union[Image, List[Image]]]): A list of input images.
            inference_params (Optional[CommonInferenceParams]): Parameters for controlling the inference process.
            max_batch_size (int): max batch size for inference. Defaults to 4.
            random_seed (Optional[int]): random seed for inference. Defaults to None.

        Returns:
            dict: A dictionary containing the generated results.
        """

        results = generate(
            wrapped_model=self.inference_wrapped_model,
            tokenizer=self.processor.tokenizer,
            image_processor=self.processor.image_processor,
            prompts=prompts,
            images=images,
            processor=self.processor,
            max_batch_size=max_batch_size,
            random_seed=random_seed,
            inference_params=inference_params,
        )

        return results

    @property
    def get_triton_input(self):
        inputs = (
            Tensor(name="prompts", shape=(-1,), dtype=bytes),
            Tensor(name="images", shape=(-1, -1, -1, 3), dtype=np.uint8),
            Tensor(name="max_length", shape=(-1,), dtype=np.int_, optional=True),
            Tensor(name="max_batch_size", shape=(-1,), dtype=np.int_, optional=True),
            Tensor(name="top_k", shape=(-1,), dtype=np.int_, optional=True),
            Tensor(name="top_p", shape=(-1,), dtype=np.single, optional=True),
            Tensor(name="temperature", shape=(-1,), dtype=np.single, optional=True),
            Tensor(name="random_seed", shape=(-1,), dtype=np.int_, optional=True),
        )
        return inputs

    @property
    def get_triton_output(self):
        return (Tensor(name="sentences", shape=(-1,), dtype=bytes),)

    @batch
    @first_value(
        "max_length",
        "max_batch_size",
        "top_k",
        "top_p",
        "temperature",
        "random_seed",
    )
    def triton_infer_fn(self, **inputs: np.ndarray):
        prompts = str_ndarray2list(inputs.pop("prompts"))
        images = ndarray2img(inputs.pop("images"))
        temperature = inputs.pop("temperature", 1.0)
        top_k = inputs.pop("top_k", 1)
        top_p = inputs.pop("top_p", 0.0)
        num_tokens_to_generate = inputs.pop("max_length", 50)
        random_seed = inputs.pop("random_seed", None)
        max_batch_size = inputs.pop("max_batch_size", 4)

        output_infer = self._infer_fn(
            prompts=prompts,
            images=images,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            num_tokens_to_generate=num_tokens_to_generate,
            random_seed=random_seed,
            max_batch_size=max_batch_size,
        )

        # Format output for triton
        output_infer["sentences"] = cast_output(output_infer["sentences"], np.bytes_)
        return output_infer

    def _infer_fn(
        self,
        prompts,
        images,
        temperature=1.0,
        top_k=1,
        top_p=0.0,
        num_tokens_to_generate=256,
        random_seed=None,
        max_batch_size=4,
    ):
        """Private helper function that handles the core inference logic shared between triton and ray inference.

        Args:
            prompts (List[str]): List of input prompts
            images (List[Union[Image, List[Image]]]): List of input images
            temperature (float): Sampling temperature
            top_k (int): Top-k sampling parameter
            top_p (float): Top-p sampling parameter
            num_tokens_to_generate (int): Maximum number of tokens to generate
            random_seed (Optional[int]): Random seed for inference
            max_batch_size (int): Maximum batch size for inference

        Returns:
            dict: sentences.
        """
        inference_params = CommonInferenceParams(
            temperature=float(temperature),
            top_k=int(top_k),
            top_p=float(top_p),
            num_tokens_to_generate=num_tokens_to_generate,
        )

        results = self.generate(
            prompts, images, inference_params, max_batch_size=max_batch_size, random_seed=random_seed
        )

        output_infer = {"sentences": [r.generated_text for r in results]}

        return output_infer
