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
import os
from typing import List, Optional

import numpy as np
import requests
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from pydantic_settings import BaseSettings

from nemo_deploy.multimodal.query_multimodal import NemoQueryMultimodalPytorch

logger = logging.getLogger(__name__)


class TritonSettings(BaseSettings):
    """TritonSettings class that gets the values of TRITON_HTTP_ADDRESS and TRITON_PORT."""

    _triton_service_port: int
    _triton_service_ip: str

    def __init__(self):
        super(TritonSettings, self).__init__()
        try:
            self._triton_service_port = int(os.environ.get("TRITON_PORT", 8000))
            self._triton_service_ip = os.environ.get("TRITON_HTTP_ADDRESS", "0.0.0.0")
        except Exception as error:
            logger.error(
                "An exception occurred trying to retrieve set args in TritonSettings class. Error:",
                error,
            )
            return

    @property
    def triton_service_port(self):
        """Returns the port number for the Triton service."""
        return self._triton_service_port

    @property
    def triton_service_ip(self):
        """Returns the IP address for the Triton service."""
        return self._triton_service_ip


app = FastAPI()
triton_settings = TritonSettings()


class BaseMultimodalRequest(BaseModel):
    """Common parameters for multimodal completions and chat requests.

    Attributes:
        model (str): The name of the model to use for completion.
        max_tokens (int): The maximum number of tokens to generate in the response.
        temperature (float): Sampling temperature for randomness in generation.
        top_p (float): Cumulative probability for nucleus sampling.
        top_k (int): Number of highest-probability tokens to consider for sampling.
        random_seed (Optional[int]): Random seed for generation.
        max_batch_size (int): Maximum batch size for inference.
    """

    model: str
    max_tokens: int = 50
    temperature: float = 1.0
    top_p: float = 0.0
    top_k: int = 0
    random_seed: Optional[int] = None
    max_batch_size: int = 4


class MultimodalCompletionRequest(BaseMultimodalRequest):
    """Represents a request for multimodal text completion.

    Attributes:
        prompt (str): The input text to generate a response from.
        image (Optional[str]): Base64-encoded image or image URL.
        apply_chat_template (bool): Whether to apply chat template.
    """

    prompt: str
    image: Optional[str] = None
    apply_chat_template: bool = False


class ImageContent(BaseModel):
    """Image content in chat messages."""

    type: str = "image_url"
    image_url: dict


class TextContent(BaseModel):
    """Text content in chat messages."""

    type: str = "text"
    text: str


class MultimodalChatCompletionRequest(BaseMultimodalRequest):
    """Represents a request for multimodal chat completion.

    Attributes:
        messages (List[dict]): A list of message dictionaries for chat completion.
    """

    messages: List[dict]


@app.get("/v1/health")
def health_check():
    """Health check endpoint to verify that the API is running.

    Returns:
        dict: A dictionary indicating the status of the application.
    """
    return {"status": "ok"}


@app.get("/v1/triton_health")
async def check_triton_health():
    """This method exposes endpoint "/triton_health".

    This can be used to verify if Triton server is accessible while running the REST or FastAPI application.
    Verify by running: curl http://service_http_address:service_port/v1/triton_health and the returned status should
    inform if the server is accessible.
    """
    triton_url = (
        f"http://{triton_settings.triton_service_ip}:{str(triton_settings.triton_service_port)}/v2/health/ready"
    )
    logger.info(f"Attempting to connect to Triton server at: {triton_url}")
    try:
        response = requests.get(triton_url, timeout=5)
        if response.status_code == 200:
            return {"status": "Triton server is reachable and ready"}
        else:
            raise HTTPException(status_code=503, detail="Triton server is not ready")
    except requests.RequestException as e:
        raise HTTPException(status_code=503, detail=f"Cannot reach Triton server: {str(e)}")


def convert_numpy(obj):
    """Convert NumPy arrays in output to lists."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(i) for i in obj]
    else:
        return obj


def _helper_fun(
    url,
    model,
    prompts,
    images,
    temperature,
    top_k,
    top_p,
    max_length,
    random_seed,
    max_batch_size,
    apply_chat_template,
):
    """run_in_executor doesn't allow to pass kwargs, so we have this helper function to pass args as a list."""
    nq = NemoQueryMultimodalPytorch(url=url, model_name=model)

    output = nq.query_multimodal(
        prompts=prompts,
        images=images,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        max_length=max_length,
        random_seed=random_seed,
        max_batch_size=max_batch_size,
        apply_chat_template=apply_chat_template,
        init_timeout=300,
    )
    return output


async def query_multimodal_async(
    *,
    url,
    model,
    prompts,
    images,
    temperature,
    top_k,
    top_p,
    max_length,
    random_seed,
    max_batch_size,
    apply_chat_template,
):
    """Sends requests to `NemoQueryMultimodalPytorch.query_multimodal` in a non-blocking way.

    This allows the server to process concurrent requests. This way enables batching of requests
    in the underlying Triton server.
    """
    import asyncio
    import concurrent

    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor() as pool:
        result = await loop.run_in_executor(
            pool,
            _helper_fun,
            url,
            model,
            prompts,
            images,
            temperature,
            top_k,
            top_p,
            max_length,
            random_seed,
            max_batch_size,
            apply_chat_template,
        )
    return result


@app.post("/v1/completions/")
async def completions_v1(request: MultimodalCompletionRequest):
    """Defines the multimodal completions endpoint and queries the model deployed on PyTriton server."""
    url = f"http://{triton_settings.triton_service_ip}:{triton_settings.triton_service_port}"

    prompts = request.prompt
    if not isinstance(request.prompt, list):
        prompts = [request.prompt]

    images = []
    if request.image:
        images = [request.image]

    output = await query_multimodal_async(
        url=url,
        model=request.model,
        prompts=prompts,
        images=images,
        temperature=request.temperature,
        top_k=request.top_k,
        top_p=request.top_p,
        max_length=request.max_tokens,
        random_seed=request.random_seed,
        max_batch_size=request.max_batch_size,
        apply_chat_template=False,
    )

    output_serializable = convert_numpy(output)
    output_serializable["choices"][0]["text"] = output_serializable["choices"][0]["text"][0][0]
    logger.info(f"Output: {output_serializable}")
    return output_serializable


def dict_to_str(messages):
    """Serializes dict to str."""
    return json.dumps(messages)


@app.post("/v1/chat/completions/")
async def chat_completions_v1(request: MultimodalChatCompletionRequest):
    """Defines the multimodal chat completions endpoint and queries the model deployed on PyTriton server.

    Supports two image content formats (normalized internally to format 1):
    1. {"type": "image", "image": "url_or_base64"}
    2. {"type": "image_url", "image_url": {"url": "url_or_base64"}} (OpenAI-style, converted to format 1)
    """
    url = f"http://{triton_settings.triton_service_ip}:{triton_settings.triton_service_port}"

    prompts = request.messages
    if not isinstance(request.messages, list):
        prompts = [request.messages]

    # Normalize image_url format to image format for consistent processing
    for message in prompts:
        for content in message["content"]:
            if content["type"] == "image_url":
                # Convert OpenAI-style image_url to standard image format
                if isinstance(content.get("image_url"), dict):
                    image_data = content["image_url"]["url"]
                else:
                    image_data = content["image_url"]
                # Transform to image format
                content["type"] = "image"
                content["image"] = image_data
                # Remove image_url field
                content.pop("image_url", None)

    # Serialize the dictionary to a JSON string represnetation to be able to convert to numpy array
    # (str_list2numpy) and back to list (str_ndarray2list) as required by PyTriton. Using the dictionaries directly
    # with these methods is not possible as they expect string type.
    json_prompts = [dict_to_str(prompts)]

    # Extract images from messages
    images = []
    for message in prompts:
        for content in message["content"]:
            if content["type"] == "image":
                images.append(content["image"])

    output = await query_multimodal_async(
        url=url,
        model=request.model,
        prompts=json_prompts,
        images=images,
        temperature=request.temperature,
        top_k=request.top_k,
        top_p=request.top_p,
        max_length=request.max_tokens,
        random_seed=request.random_seed,
        max_batch_size=request.max_batch_size,
        apply_chat_template=True,
    )

    # Add 'role' as 'assistant' key to the output dict
    output["choices"][0]["message"] = {
        "role": "assistant",
        "content": output["choices"][0]["text"],
    }
    output["object"] = "chat.completion"

    del output["choices"][0]["text"]

    output_serializable = convert_numpy(output)
    output_serializable["choices"][0]["message"]["content"] = output_serializable["choices"][0]["message"]["content"][
        0
    ][0]

    logger.info(f"Output: {output_serializable}")
    return output_serializable
