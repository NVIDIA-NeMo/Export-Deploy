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
import base64
import json
import logging

import requests

LOGGER = logging.getLogger("NeMo")


def parse_args():
    parser = argparse.ArgumentParser(description="Query a deployed multimodal model using Ray")
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host address of the Ray Serve server",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port number of the Ray Serve server",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="nemo-multimodal-model",
        help="Identifier for the model in the API responses",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Custom prompt to use for testing. If not provided, default prompt will be used.",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path or URL to input image file",
    )
    return parser.parse_args()


def load_image_from_path(image_path: str) -> str:
    """Load an image from a file path or URL.

    Args:
        image_path: Path to local image file or URL

    Returns:
        Base64-encoded image string
    """
    if image_path.startswith(("http://", "https://")):
        LOGGER.info(f"Loading image from URL: {image_path}")
        response = requests.get(image_path, timeout=30)
        response.raise_for_status()
        image_content = response.content
    else:
        LOGGER.info(f"Loading image from local path: {image_path}")
        with open(image_path, "rb") as f:
            image_content = f.read()

    return base64.b64encode(image_content).decode("utf-8")


def test_completions_endpoint(base_url: str, model_id: str, prompt: str = None, image_source: str = None) -> None:
    """Test the completions endpoint for multimodal models."""
    url = f"{base_url}/v1/completions/"

    # Use provided prompt or default
    default_prompt = "Describe this image in detail."
    default_image = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
    prompt_text = prompt if prompt is not None else default_prompt
    image_source = image_source if image_source is not None else default_image

    # Prepare payload
    payload = {
        "model": model_id,
        "max_tokens": 100,
        "temperature": 1.0,
        "top_p": 0.0,
        "top_k": 1,
    }

    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-3B-Instruct")
    text = processor.apply_chat_template(
        [
            {
                "role": "user",
                "content": [{"type": "image", "image": image_source}, {"type": "text", "text": prompt_text}],
            }
        ],
        tokenizer=False,
        add_generation_prompt=True,
    )
    payload["prompt"] = text

    try:
        image_base64 = load_image_from_path(image_source)
        payload["image"] = image_base64
    except Exception as e:
        LOGGER.error(f"Failed to load image: {e}")
        return

    LOGGER.info(f"Testing completions endpoint at {url}")
    response = requests.post(url, json=payload)
    LOGGER.info(f"Response status code: {response.status_code}")
    if response.status_code == 200:
        LOGGER.info(f"Response: {json.dumps(response.json(), indent=2)}")
    else:
        LOGGER.error(f"Error: {response.text}")


def test_chat_completions_endpoint(base_url: str, model_id: str, prompt: str = None, image_source: str = None) -> None:
    """Test the chat completions endpoint for multimodal models."""
    url = f"{base_url}/v1/chat/completions/"

    # Use provided prompt or default
    default_message = "What do you see in this image?"
    default_image = "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg"
    message_content = prompt if prompt is not None else default_message
    image_source = image_source if image_source is not None else default_image

    content = []
    try:
        image_base64 = load_image_from_path(image_source)
        content.append({"type": "image", "image": image_base64})
    except Exception as e:
        LOGGER.error(f"Failed to load image: {e}")
        return

    content.append({"type": "text", "text": message_content})

    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 100,
        "temperature": 1.0,
        "top_p": 0.0,
        "top_k": 1,
    }

    LOGGER.info(f"Testing chat completions endpoint at {url}")
    response = requests.post(url, json=payload)
    LOGGER.info(f"Response status code: {response.status_code}")
    if response.status_code == 200:
        LOGGER.info(f"Response: {json.dumps(response.json(), indent=2)}")
    else:
        LOGGER.error(f"Error: {response.text}")


def test_models_endpoint(base_url: str) -> None:
    """Test the models endpoint."""
    url = f"{base_url}/v1/models"

    LOGGER.info(f"Testing models endpoint at {url}")
    response = requests.get(url)
    LOGGER.info(f"Response status code: {response.status_code}")
    if response.status_code == 200:
        LOGGER.info(f"Response: {json.dumps(response.json(), indent=2)}")
    else:
        LOGGER.error(f"Error: {response.text}")


def test_health_endpoint(base_url: str) -> None:
    """Test the health endpoint."""
    url = f"{base_url}/v1/health"

    LOGGER.info(f"Testing health endpoint at {url}")
    response = requests.get(url)
    LOGGER.info(f"Response status code: {response.status_code}")
    if response.status_code == 200:
        LOGGER.info(f"Response: {json.dumps(response.json(), indent=2)}")
    else:
        LOGGER.error(f"Error: {response.text}")


def main():
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    args = parse_args()
    base_url = f"http://{args.host}:{args.port}"

    LOGGER.info(f"Testing endpoints for multimodal model {args.model_id} at {base_url}")
    if args.prompt:
        LOGGER.info(f"Using custom prompt: {args.prompt[:100]}{'...' if len(args.prompt) > 100 else ''}")
    else:
        LOGGER.info("Using default prompts")

    if args.image:
        LOGGER.info(f"Using image: {args.image}")
    else:
        LOGGER.warning("No image provided. Multimodal endpoints may not work properly.")

    # Test all endpoints
    test_completions_endpoint(base_url, args.model_id, args.prompt, args.image)
    test_chat_completions_endpoint(base_url, args.model_id, args.prompt, args.image)
    test_health_endpoint(base_url)
    test_models_endpoint(base_url)


if __name__ == "__main__":
    main()
