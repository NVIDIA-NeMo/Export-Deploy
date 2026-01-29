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
import logging
import time

from transformers import AutoProcessor

from nemo_deploy.multimodal.query_multimodal import NemoQueryMultimodalPytorch

LOGGER = logging.getLogger("NeMo")


def load_image_from_path(image_path: str) -> str:
    """Load an image from a file path or URL.

    Args:
        image_path: Path to local image file or URL

    Returns:
        Image string - HTTP URL directly or base64-encoded string for local files
    """
    if image_path.startswith(("http://", "https://")):
        LOGGER.info(f"Using image URL directly: {image_path}")
        return image_path
    else:
        LOGGER.info(f"Loading and encoding image from local path: {image_path}")
        with open(image_path, "rb") as f:
            image_content = f.read()
        return "data:image;base64," + base64.b64encode(image_content).decode("utf-8")


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Queries Triton server running an in-framework Nemo multimodal model",
    )
    parser.add_argument("-u", "--url", default="0.0.0.0", type=str, help="url for the triton server")
    parser.add_argument("-mn", "--model_name", required=True, type=str, help="Name of the triton model")
    parser.add_argument(
        "-pn",
        "--processor_name",
        default="Qwen/Qwen2.5-VL-7B-Instruct",
        type=str,
        help="Processor name for qwen-vl models (default: Qwen/Qwen2.5-VL-7B-Instruct)",
    )

    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument("-p", "--prompt", required=False, type=str, help="Prompt")
    prompt_group.add_argument(
        "-pf",
        "--prompt_file",
        required=False,
        type=str,
        help="File to read the prompt from",
    )
    parser.add_argument(
        "-i",
        "--image",
        required=True,
        type=str,
        help="Path or URL to input image file",
    )
    parser.add_argument(
        "-mol",
        "--max_output_len",
        default=50,
        type=int,
        help="Max output token length",
    )
    parser.add_argument(
        "-mbs",
        "--max_batch_size",
        default=4,
        type=int,
        help="Max batch size for inference",
    )
    parser.add_argument("-tk", "--top_k", default=1, type=int, help="top_k")
    parser.add_argument("-tpp", "--top_p", default=0.0, type=float, help="top_p")
    parser.add_argument("-t", "--temperature", default=1.0, type=float, help="temperature")
    parser.add_argument(
        "-rs",
        "--random_seed",
        default=None,
        type=int,
        help="Random seed for generation",
    )
    parser.add_argument(
        "-it",
        "--init_timeout",
        default=60.0,
        type=float,
        help="init timeout for the triton server",
    )

    args = parser.parse_args()
    return args


def query():
    args = get_args()

    if args.prompt_file is not None:
        with open(args.prompt_file, "r") as f:
            args.prompt = f.read()

    image_source = load_image_from_path(args.image)

    if "Qwen" in args.processor_name:
        processor = AutoProcessor.from_pretrained(args.processor_name)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": args.image,  # Use image path directly
                    },
                    {"type": "text", "text": args.prompt},
                ],
            }
        ]
        text = processor.apply_chat_template(messages, tokenizer=False, add_generation_prompt=True)
        args.prompt = text
    else:
        raise ValueError(f"Model {args.processor_name} not supported")

    start_time = time.time()
    nemo_query = NemoQueryMultimodalPytorch(args.url, args.model_name)
    outputs = nemo_query.query_multimodal(
        prompts=[args.prompt],
        images=[image_source],
        max_length=args.max_output_len,
        max_batch_size=args.max_batch_size,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
        random_seed=args.random_seed,
        init_timeout=args.init_timeout,
    )
    end_time = time.time()
    LOGGER.info(f"Query execution time: {end_time - start_time:.2f} seconds")
    print(outputs)


if __name__ == "__main__":
    query()
