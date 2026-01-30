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
from PIL import Image

from nemo_deploy import ITritonDeployable
from nemo_deploy.utils import cast_output, str_ndarray2list
from nemo_export_deploy_common.import_utils import (
    MISSING_MBRIDGE_MSG,
    MISSING_TRITON_MSG,
    UnavailableError,
    null_decorator,
)

try:
    from megatron.bridge.inference.vlm.base import generate, setup_model_and_tokenizer
    from megatron.bridge.inference.vlm.qwenvl_inference_wrapper import QwenVLInferenceWrapper

    HAVE_MBRIDGE = True
except (ImportError, ModuleNotFoundError):
    HAVE_MBRIDGE = False
    from typing import Any

    generate = Any
    setup_model_and_tokenizer = Any
    QwenVLInferenceWrapper = Any

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


class MegatronMultimodalDeployable(ITritonDeployable):
    """Triton inference server compatible deploy class for a Megatron multimodal model file.

    Args:
        megatron_checkpoint_filepath (str): path for the megatron checkpoint.
        tensor_model_parallel_size (int): tensor parallelism.
        pipeline_model_parallel_size (int): pipeline parallelism.
        params_dtype (torch.dtype): data type for model parameters.
        inference_batch_times_seqlen_threshold (int): sequence threshold.
        inference_max_seq_length (int): maximum sequence length for inference.
    """

    def __init__(
        self,
        megatron_checkpoint_filepath: str,
        tensor_model_parallel_size: int = 1,
        pipeline_model_parallel_size: int = 1,
        params_dtype: torch.dtype = torch.bfloat16,
        inference_batch_times_seqlen_threshold: int = 1000,
        inference_max_seq_length: int = 8192,
    ):
        if not HAVE_TRITON:
            raise UnavailableError(MISSING_TRITON_MSG)
        if not HAVE_MBRIDGE:
            raise UnavailableError(MISSING_MBRIDGE_MSG)

        self.megatron_checkpoint_filepath = megatron_checkpoint_filepath
        self.tensor_model_parallel_size = tensor_model_parallel_size
        self.pipeline_model_parallel_size = pipeline_model_parallel_size
        self.params_dtype = params_dtype
        self.inference_batch_times_seqlen_threshold = inference_batch_times_seqlen_threshold
        self.inference_max_seq_length = inference_max_seq_length

        self.inference_wrapped_model, self.processor = setup_model_and_tokenizer(
            megatron_model_path=megatron_checkpoint_filepath,
            tp=tensor_model_parallel_size,
            pp=pipeline_model_parallel_size,
            params_dtype=params_dtype,
            inference_batch_times_seqlen_threshold=inference_batch_times_seqlen_threshold,
            inference_max_seq_length=inference_max_seq_length,
        )

    def generate(
        self,
        prompts: List[str],
        images: List[Image],
        inference_params: Optional[CommonInferenceParams] = None,
        max_batch_size: int = 4,
        random_seed: Optional[int] = None,
        apply_chat_template: bool = False,
    ) -> dict:
        """Generates text based on the provided input prompts and images.

        Args:
            prompts (List[str]): A list of input strings.
            images (List[Union[Image, List[Image]]]): A list of input images.
            inference_params (Optional[CommonInferenceParams]): Parameters for controlling the inference process.
            max_batch_size (int): max batch size for inference. Defaults to 4.
            random_seed (Optional[int]): random seed for inference. Defaults to None.
            apply_chat_template (bool): Whether to apply chat template. Defaults to False.

        Returns:
            dict: A dictionary containing the generated results.
        """
        if apply_chat_template:
            prompts = [self.apply_chat_template(prompt) for prompt in prompts]

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

    def apply_chat_template(self, messages, add_generation_prompt=True):
        """Apply the chat template using the processor.

        Works when model's processor has chat template (typically chat models).
        """
        if isinstance(messages, str):
            messages = json.loads(messages)

        text = self.processor.apply_chat_template(
            messages, tokenizer=False, add_generation_prompt=add_generation_prompt
        )
        return text

    def process_image_input(self, image_source):
        """Process image input from base64-encoded string or HTTP URL.

        Args:
            image_source (str): Image source - either base64-encoded image string with data URI prefix
                               (e.g., "data:image;base64,...") or HTTP/HTTPS URL (e.g., "http://example.com/image.jpg")

        Returns:
            Processed image content suitable for model inference.
        """
        if isinstance(self.inference_wrapped_model, QwenVLInferenceWrapper):
            from qwen_vl_utils import process_vision_info

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_source},
                    ],
                }
            ]
            image_content, _ = process_vision_info(messages)
            return image_content
        else:
            raise ValueError(f"Model {self.inference_wrapped_model} not supported")

    @property
    def get_triton_input(self):
        inputs = (
            Tensor(name="prompts", shape=(-1,), dtype=bytes),
            Tensor(name="images", shape=(-1,), dtype=bytes),
            Tensor(name="max_length", shape=(-1,), dtype=np.int_, optional=True),
            Tensor(name="max_batch_size", shape=(-1,), dtype=np.int_, optional=True),
            Tensor(name="top_k", shape=(-1,), dtype=np.int_, optional=True),
            Tensor(name="top_p", shape=(-1,), dtype=np.single, optional=True),
            Tensor(name="temperature", shape=(-1,), dtype=np.single, optional=True),
            Tensor(name="random_seed", shape=(-1,), dtype=np.int_, optional=True),
            Tensor(name="apply_chat_template", shape=(-1,), dtype=np.bool_, optional=True),
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
        "apply_chat_template",
    )
    def triton_infer_fn(self, **inputs: np.ndarray):
        prompts = str_ndarray2list(inputs.pop("prompts"))
        images = str_ndarray2list(inputs.pop("images"))
        temperature = inputs.pop("temperature", 1.0)
        top_k = inputs.pop("top_k", 1)
        top_p = inputs.pop("top_p", 0.0)
        num_tokens_to_generate = inputs.pop("max_length", 50)
        random_seed = inputs.pop("random_seed", None)
        max_batch_size = inputs.pop("max_batch_size", 4)
        apply_chat_template = inputs.pop("apply_chat_template", False)

        output_infer = self._infer_fn(
            prompts=prompts,
            images=images,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            num_tokens_to_generate=num_tokens_to_generate,
            random_seed=random_seed,
            max_batch_size=max_batch_size,
            apply_chat_template=apply_chat_template,
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
        apply_chat_template=False,
    ):
        """Private helper function that handles the core inference logic shared between triton and ray inference.

        Args:
            prompts (List[str]): List of input prompts
            images (List[str]): List of input base64 encoded images
            temperature (float): Sampling temperature
            top_k (int): Top-k sampling parameter
            top_p (float): Top-p sampling parameter
            num_tokens_to_generate (int): Maximum number of tokens to generate
            random_seed (Optional[int]): Random seed for inference
            max_batch_size (int): Maximum batch size for inference
            apply_chat_template (bool): Whether to apply chat template

        Returns:
            dict: sentences.
        """
        # Handle temperature=0.0 for greedy decoding
        if temperature == 0.0:
            LOGGER.warning("temperature=0.0 detected. Setting top_k=1 for greedy sampling.")
            top_k = 1
            top_p = 0.0

        inference_params = CommonInferenceParams(
            temperature=float(temperature),
            top_k=int(top_k),
            top_p=float(top_p),
            num_tokens_to_generate=num_tokens_to_generate,
        )

        images = [self.process_image_input(image_source) for image_source in images]

        results = self.generate(
            prompts,
            images,
            inference_params,
            max_batch_size=max_batch_size,
            random_seed=random_seed,
            apply_chat_template=apply_chat_template,
        )

        output_infer = {"sentences": [r.generated_text for r in results]}

        return output_infer

    def ray_infer_fn(self, inputs: dict):
        """Ray-compatible inference function that takes a dictionary of inputs and returns a dictionary of outputs.

        Args:
            inputs (dict): Dictionary containing the following optional keys:
                - prompts (List[str]): List of input prompts
                - images (List[str]): List of input base64 encoded images
                - temperature (float): Sampling temperature (default: 1.0)
                - top_k (int): Top-k sampling parameter (default: 1)
                - top_p (float): Top-p sampling parameter (default: 0.0)
                - max_length (int): Maximum number of tokens to generate (default: 50)
                - random_seed (Optional[int]): Random seed for reproducibility (default: None)
                - max_batch_size (int): Maximum batch size for inference (default: 4)
                - apply_chat_template (bool): Whether to apply chat template (default: False)

        Returns:
            dict: Dictionary containing:
                - sentences (List[str]): List of generated texts
        """
        prompts = inputs.get("prompts", [])
        images = inputs.get("images", [])
        temperature = inputs.get("temperature", 1.0)
        top_k = inputs.get("top_k", 1)
        top_p = inputs.get("top_p", 0.0)
        num_tokens_to_generate = inputs.get("max_length", 50)
        random_seed = inputs.get("random_seed", None)
        max_batch_size = inputs.get("max_batch_size", 4)
        apply_chat_template = inputs.get("apply_chat_template", False)

        return self._infer_fn(
            prompts=prompts,
            images=images,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            num_tokens_to_generate=num_tokens_to_generate,
            random_seed=random_seed,
            max_batch_size=max_batch_size,
            apply_chat_template=apply_chat_template,
        )
