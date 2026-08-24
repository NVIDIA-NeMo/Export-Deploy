# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
import base64
import json
import logging
from pathlib import Path

import requests

LOGGER = logging.getLogger("NeMo")


def parse_args():
    parser = argparse.ArgumentParser(description="Query a deployed Megatron multimodal model using Ray")
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host address of the Ray Serve server",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=1024,
        help="Port number of the Ray Serve server",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="megatron-multimodal-model",
        help="Identifier for the model in the API responses",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Describe this image in detail.",
        help="Text prompt to use for testing",
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default=None,
        help="Path to an image file to use for testing. If not provided, a sample URL will be used.",
    )
    parser.add_argument(
        "--image_url",
        type=str,
        default="https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg",
        help="URL of an image to use for testing (used if --image_path is not provided)",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=100,
        help="Maximum number of tokens to generate",
    )
    return parser.parse_args()


def encode_image_to_base64(image_path: str) -> str:
    """Encode a local image file to base64 string with data URI prefix."""
    with open(image_path, "rb") as image_file:
        image_data = base64.b64encode(image_file.read()).decode("utf-8")

    # Determine image format from file extension
    ext = Path(image_path).suffix.lower()
    mime_type = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".gif": "image/gif",
        ".webp": "image/webp",
    }.get(ext, "image/jpeg")

    return f"data:{mime_type};base64,{image_data}"


def test_chat_completions_endpoint(
    base_url: str,
    model_id: str,
    prompt: str,
    image_source: str,
    max_tokens: int = 100,
) -> None:
    """Test the multimodal chat completions endpoint."""
    url = f"{base_url}/v1/chat/completions/"

    payload = {
        "model": model_id,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_source}},
                    {"type": "text", "text": prompt},
                ],
            }
        ],
        "max_tokens": max_tokens,
    }

    LOGGER.info(f"Testing multimodal chat completions endpoint at {url}")
    LOGGER.info(f"Prompt: {prompt}")
    LOGGER.info(
        f"Image source: {image_source[:100]}..." if len(image_source) > 100 else f"Image source: {image_source}"
    )

    try:
        response = requests.post(url, json=payload, timeout=120)
        LOGGER.info(f"Response status code: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            LOGGER.info(f"Response: {json.dumps(result, indent=2)}")
            if "choices" in result and len(result["choices"]) > 0:
                generated_text = result["choices"][0]["message"]["content"]
                LOGGER.info(f"\nGenerated text: {generated_text}")
        else:
            LOGGER.error(f"Error: {response.text}")
    except requests.exceptions.RequestException as e:
        LOGGER.error(f"Request failed: {str(e)}")


def test_completions_endpoint(
    base_url: str,
    model_id: str,
    prompt: str,
    image_source: str,
    max_tokens: int = 100,
) -> None:
    """Test the multimodal completions endpoint."""
    url = f"{base_url}/v1/completions/"

    payload = {
        "model": model_id,
        "prompt": prompt,
        "image": image_source,
        "max_tokens": max_tokens,
    }

    LOGGER.info(f"Testing multimodal completions endpoint at {url}")
    LOGGER.info(f"Prompt: {prompt}")
    LOGGER.info(
        f"Image source: {image_source[:100]}..." if len(image_source) > 100 else f"Image source: {image_source}"
    )

    try:
        response = requests.post(url, json=payload, timeout=120)
        LOGGER.info(f"Response status code: {response.status_code}")
        if response.status_code == 200:
            result = response.json()
            LOGGER.info(f"Response: {json.dumps(result, indent=2)}")
            if "choices" in result and len(result["choices"]) > 0:
                generated_text = result["choices"][0]["text"]
                LOGGER.info(f"\nGenerated text: {generated_text}")
        else:
            LOGGER.error(f"Error: {response.text}")
    except requests.exceptions.RequestException as e:
        LOGGER.error(f"Request failed: {str(e)}")


def test_models_endpoint(base_url: str) -> None:
    """Test the models endpoint."""
    url = f"{base_url}/v1/models"

    LOGGER.info(f"Testing models endpoint at {url}")
    try:
        response = requests.get(url, timeout=30)
        LOGGER.info(f"Response status code: {response.status_code}")
        if response.status_code == 200:
            LOGGER.info(f"Response: {json.dumps(response.json(), indent=2)}")
        else:
            LOGGER.error(f"Error: {response.text}")
    except requests.exceptions.RequestException as e:
        LOGGER.error(f"Request failed: {str(e)}")


def test_health_endpoint(base_url: str) -> None:
    """Test the health endpoint."""
    url = f"{base_url}/v1/health"

    LOGGER.info(f"Testing health endpoint at {url}")
    try:
        response = requests.get(url, timeout=30)
        LOGGER.info(f"Response status code: {response.status_code}")
        if response.status_code == 200:
            LOGGER.info(f"Response: {json.dumps(response.json(), indent=2)}")
        else:
            LOGGER.error(f"Error: {response.text}")
    except requests.exceptions.RequestException as e:
        LOGGER.error(f"Request failed: {str(e)}")


def main():
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    args = parse_args()
    base_url = f"http://{args.host}:{args.port}"

    LOGGER.info(f"Testing multimodal endpoints for model {args.model_id} at {base_url}")

    # Determine image source
    if args.image_path:
        LOGGER.info(f"Using local image: {args.image_path}")
        image_source = encode_image_to_base64(args.image_path)
    else:
        LOGGER.info(f"Using image URL: {args.image_url}")
        image_source = args.image_url

    # Test all endpoints
    LOGGER.info("\n" + "=" * 80)
    LOGGER.info("Testing Health Endpoint")
    LOGGER.info("=" * 80)
    test_health_endpoint(base_url)

    LOGGER.info("\n" + "=" * 80)
    LOGGER.info("Testing Models Endpoint")
    LOGGER.info("=" * 80)
    test_models_endpoint(base_url)

    LOGGER.info("\n" + "=" * 80)
    LOGGER.info("Testing Chat Completions Endpoint")
    LOGGER.info("=" * 80)
    test_chat_completions_endpoint(
        base_url,
        args.model_id,
        args.prompt,
        image_source,
        args.max_tokens,
    )

    LOGGER.info("\n" + "=" * 80)
    LOGGER.info("Testing Completions Endpoint")
    LOGGER.info("=" * 80)
    test_completions_endpoint(
        base_url,
        args.model_id,
        args.prompt,
        image_source,
        args.max_tokens,
    )


if __name__ == "__main__":
    main()
