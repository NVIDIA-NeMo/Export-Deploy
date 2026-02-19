# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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

import time
from io import BytesIO
from typing import List, Optional

import numpy as np
import requests

from nemo_deploy.utils import str_list2numpy
from nemo_export_deploy_common.import_utils import (
    MISSING_DECORD_MSG,
    MISSING_PIL_MSG,
    MISSING_TRITON_MSG,
    UnavailableError,
)

try:
    from PIL import Image

    HAVE_PIL = True
except (ImportError, ModuleNotFoundError):
    HAVE_PIL = False

try:
    from decord import VideoReader

    HAVE_DECORD = True
except (ImportError, ModuleNotFoundError):
    HAVE_DECORD = False

try:
    from pytriton.client import ModelClient

    HAVE_TRITON = True
except (ImportError, ModuleNotFoundError):
    from unittest.mock import MagicMock

    ModelClient = MagicMock()
    HAVE_TRITON = False


class NemoQueryMultimodal:
    """Sends a query to Triton for Multimodal inference.

    Example:
        from nemo_deploy.multimodal import NemoQueryMultimodal

        nq = NemoQueryMultimodal(url="localhost", model_name="neva", model_type="neva")

        input_text = "Hi! What is in this image?"
        output = nq.query(
            input_text=input_text,
            input_media="/path/to/image.jpg",
            max_output_len=30,
            top_k=1,
            top_p=0.0,
            temperature=1.0,
        )
        print("prompts: ", prompts)
    """

    def __init__(self, url, model_name, model_type):
        self.url = url
        self.model_name = model_name
        self.model_type = model_type

    def setup_media(self, input_media):
        """Setup input media."""
        if self.model_type == "video-neva":
            if not HAVE_DECORD:
                raise UnavailableError(MISSING_DECORD_MSG)

            vr = VideoReader(input_media)
            frames = [f.asnumpy() for f in vr]
            return np.array(frames)
        elif self.model_type == "lita" or self.model_type == "vita":
            if not HAVE_DECORD:
                raise UnavailableError(MISSING_DECORD_MSG)

            vr = VideoReader(input_media)
            frames = [f.asnumpy() for f in vr]
            subsample_len = self.frame_len(frames)
            sub_frames = self.get_subsampled_frames(frames, subsample_len)
            return np.array(sub_frames)
        elif self.model_type in ["neva", "vila", "mllama"]:
            if not HAVE_PIL:
                raise UnavailableError(MISSING_PIL_MSG)

            if input_media.startswith("http") or input_media.startswith("https"):
                response = requests.get(input_media, timeout=5)
                media = Image.open(BytesIO(response.content)).convert("RGB")
            else:
                media = Image.open(input_media).convert("RGB")
            return np.expand_dims(np.array(media), axis=0)
        else:
            raise RuntimeError(f"Invalid model type {self.model_type}")

    def frame_len(self, frames):
        """Get frame len."""
        max_frames = 256
        if len(frames) <= max_frames:
            return len(frames)
        else:
            subsample = int(np.ceil(float(len(frames)) / max_frames))
            return int(np.round(float(len(frames)) / subsample))

    def get_subsampled_frames(self, frames, subsample_len):
        """Get subsampled frames."""
        idx = np.round(np.linspace(0, len(frames) - 1, subsample_len)).astype(int)
        sub_frames = [frames[i] for i in idx]
        return sub_frames

    def query(
        self,
        input_text,
        input_media,
        batch_size=1,
        max_output_len=30,
        top_k=1,
        top_p=0.0,
        temperature=1.0,
        repetition_penalty=1.0,
        num_beams=1,
        init_timeout=60.0,
        lora_uids=None,
    ):
        """Run query."""
        if not HAVE_TRITON:
            raise UnavailableError(MISSING_TRITON_MSG)

        prompts = str_list2numpy([input_text])
        inputs = {"input_text": prompts}

        media = self.setup_media(input_media)
        if isinstance(media, dict):
            inputs.update(media)
        else:
            inputs["input_media"] = np.repeat(media[np.newaxis, :, :, :, :], prompts.shape[0], axis=0)

        if batch_size is not None:
            inputs["batch_size"] = np.full(prompts.shape, batch_size, dtype=np.int_)

        if max_output_len is not None:
            inputs["max_output_len"] = np.full(prompts.shape, max_output_len, dtype=np.int_)

        if top_k is not None:
            inputs["top_k"] = np.full(prompts.shape, top_k, dtype=np.int_)

        if top_p is not None:
            inputs["top_p"] = np.full(prompts.shape, top_p, dtype=np.single)

        if temperature is not None:
            inputs["temperature"] = np.full(prompts.shape, temperature, dtype=np.single)

        if repetition_penalty is not None:
            inputs["repetition_penalty"] = np.full(prompts.shape, repetition_penalty, dtype=np.single)

        if num_beams is not None:
            inputs["num_beams"] = np.full(prompts.shape, num_beams, dtype=np.int_)

        if lora_uids is not None:
            lora_uids = np.char.encode(lora_uids, "utf-8")
            inputs["lora_uids"] = np.full((prompts.shape[0], len(lora_uids)), lora_uids)

        with ModelClient(self.url, self.model_name, init_timeout_s=init_timeout) as client:
            result_dict = client.infer_batch(**inputs)
            output_type = client.model_config.outputs[0].dtype

            if output_type == np.bytes_:
                sentences = np.char.decode(result_dict["outputs"].astype("bytes"), "utf-8")
                return sentences
            else:
                return result_dict["outputs"]


class NemoQueryMultimodalPytorch:
    """Sends a query to Triton for Multimodal inference using PyTorch deployment.

    Example:
        from nemo_deploy.multimodal import NemoQueryMultimodalPytorch
        import base64

        nq = NemoQueryMultimodalPytorch(url="localhost", model_name="qwen")

        # Option 1: Use HTTP URL directly
        output = nq.query_multimodal(
            prompts=["Describe this image"],
            images=["http://example.com/image.jpg"],
            max_length=100,
        )

        # Option 2: Encode image to base64 with data URI prefix
        with open("image.jpg", "rb") as f:
            image_base64 = "data:image;base64," + base64.b64encode(f.read()).decode('utf-8')

        output = nq.query_multimodal(
            prompts=["Describe this image"],
            images=[image_base64],
            max_length=100,
            top_k=1,
            top_p=0.0,
            temperature=1.0,
        )
        print("output: ", output)
    """

    def __init__(self, url, model_name):
        self.url = url
        self.model_name = model_name

    def query_multimodal(
        self,
        prompts: List[str],
        images: List[str],
        max_length: Optional[int] = None,
        max_batch_size: Optional[int] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        temperature: Optional[float] = None,
        random_seed: Optional[int] = None,
        apply_chat_template: Optional[bool] = None,
        init_timeout: float = 60.0,
    ):
        """Query the Triton server synchronously for multimodal inference.

        Args:
            prompts (List[str]): List of input text prompts.
            images (List[str]): List of image strings - either base64-encoded with data URI prefix
                               (e.g., "data:image;base64,...") or HTTP/HTTPS URLs (e.g., "http://example.com/image.jpg").
            max_length (Optional[int]): Maximum number of tokens to generate.
            max_batch_size (Optional[int]): Maximum batch size for inference.
            top_k (Optional[int]): Limits to the top K tokens to consider at each step.
            top_p (Optional[float]): Limits to the top tokens within cumulative probability p.
            temperature (Optional[float]): Sampling temperature.
            random_seed (Optional[int]): Random seed for generation.
            apply_chat_template (Optional[bool]): Whether to apply chat template.
            init_timeout (float): Timeout for the connection.

        Returns:
            dict: Dictionary containing generated sentences.
        """
        if not HAVE_TRITON:
            raise UnavailableError(MISSING_TRITON_MSG)

        # Convert prompts to numpy arrays
        prompts_np = str_list2numpy(prompts)

        # Convert base64 images to numpy arrays
        images_np = str_list2numpy(images)

        inputs = {
            "prompts": prompts_np,
            "images": images_np,
        }

        # Add optional parameters if provided
        if max_length is not None:
            inputs["max_length"] = np.full(prompts_np.shape, max_length, dtype=np.int_)

        if max_batch_size is not None:
            inputs["max_batch_size"] = np.full(prompts_np.shape, max_batch_size, dtype=np.int_)

        if top_k is not None:
            inputs["top_k"] = np.full(prompts_np.shape, top_k, dtype=np.int_)

        if top_p is not None:
            inputs["top_p"] = np.full(prompts_np.shape, top_p, dtype=np.single)

        if temperature is not None:
            inputs["temperature"] = np.full(prompts_np.shape, temperature, dtype=np.single)

        if random_seed is not None:
            inputs["random_seed"] = np.full(prompts_np.shape, random_seed, dtype=np.int_)

        if apply_chat_template is not None:
            inputs["apply_chat_template"] = np.full(prompts_np.shape, apply_chat_template, dtype=np.bool_)

        with ModelClient(self.url, self.model_name, init_timeout_s=init_timeout) as client:
            result_dict = client.infer_batch(**inputs)

            # Handle output based on the model configuration
            output_type = client.model_config.outputs[0].dtype

            if output_type == np.bytes_:
                # Decode bytes output
                if "sentences" in result_dict:
                    sentences = np.char.decode(result_dict["sentences"].astype("bytes"), "utf-8")
                else:
                    return "Unknown output keyword: sentences not found"

                # Prepare OpenAI-formatted response
                openai_response = {
                    "id": f"cmpl-{int(time.time())}",
                    "object": "text_completion",
                    "created": int(time.time()),
                    "model": self.model_name,
                    "choices": [{"text": sentences}],
                }

                return openai_response
            else:
                # Return raw output if not bytes
                return result_dict
