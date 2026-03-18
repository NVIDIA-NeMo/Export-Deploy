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

import logging
import subprocess
import time

import requests

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Base64-encoded 1x1 JPEG
BASE64_IMAGE = "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQABAAD/2wBDAAgGBgcGBQgHBwcJCQgKDBQNDAsLDBkSEw8UHRofHh0aHBwgJC4nICIsIxwcKDcpLDAxNDQ0Hyc5PTgyPC4zNDL/2wBDAQkJCQwLDBgNDRgyIRwhMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjIyMjL/wAARCAAQABADASIAAhEBAxEB/8QAHwAAAQUBAQEBAQEAAAAAAAAAAAECAwQFBgcICQoL/8QAtRAAAgEDAwIEAwUFBAQAAAF9AQIDAAQRBRIhMUEGE1FhByJxFDKBkaEII0KxwRVS0fAkM2JyggkKFhcYGRolJicoKSo0NTY3ODk6Q0RFRkdISUpTVFVWV1hZWmNkZWZnaGlqc3R1dnd4eXqDhIWGh4iJipKTlJWWl5iZmqKjpKWmp6ipqrKztLW2t7i5usLDxMXGx8jJytLT1NXW19jZ2uHi4+Tl5ufo6erx8vP09fb3+Pn6/8QAHwEAAwEBAQEBAQEBAQAAAAAAAAECAwQFBgcICQoL/8QAtREAAgECBAQDBAcFBAQAAQJ3AAECAxEEBSExBhJBUQdhcRMiMoEIFEKRobHBCSMzUvAVYnLRChYkNOEl8RcYGRomJygpKjU2Nzg5OkNERUZHSElKU1RVVldYWVpjZGVmZ2hpanN0dXZ3eHl6goOEhYaHiImKkpOUlZaXmJmaoqOkpaanqKmqsrO0tba3uLm6wsPExcbHyMnK0tPU1dbX2Nna4uPk5ebn6Onq8vP09fb3+Pn6/9oADAMBAAIRAxEAPwD3+iiigAooooA//9k="


def query_ray_chat_with_base64_image(host: str, port: int, model_id: str, messages: list, max_tokens: int = 16) -> str:
    """Query /v1/chat/completions/ with a messages payload (e.g. containing base64 image)."""
    url = f"http://{host}:{port}/v1/chat/completions/"
    payload = {
        "model": model_id,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "top_p": 1.0,
    }
    response = requests.post(url, json=payload, timeout=60)
    if response.status_code == 200:
        result = response.json()
        return result.get("choices", [{}])[0].get("message", {}).get("content", "")
    logger.error(f"Chat completions error: {response.status_code} - {response.text}")
    return ""


from tests.functional_tests.utils.ray_test_utils import (
    query_ray_deployment,
    terminate_deployment_process,
    wait_for_deployment_ready,
)


class TestDeployRayVLM:
    def setup_method(self):
        """Setup for each test method."""
        self.deploy_proc = None

    def teardown_method(self):
        """Cleanup after each test method."""
        if self.deploy_proc is not None:
            terminate_deployment_process(self.deploy_proc)
            # Avoid double termination in case test used finally to clean up
            self.deploy_proc = None

    def test_deploy_ray(self):
        vlm_checkpoint_path = "/home/TestData/megatron_bridge/checkpoints/qwen25-vl-3b"

        try:
            # Run Ray deployment for Megatron multimodal (VLM) model
            self.deploy_proc = subprocess.Popen(
                [
                    "coverage",
                    "run",
                    "--data-file=/workspace/.coverage",
                    "--source=/workspace/",
                    "--parallel-mode",
                    "scripts/deploy/multimodal/deploy_ray_inframework.py",
                    "--megatron_checkpoint",
                    vlm_checkpoint_path,
                    "--model_id",
                    "megatron-multimodal",
                    "--num_gpus",
                    str(1),
                    "--host",
                    "0.0.0.0",
                    "--port",
                    str(8000),
                    "--cuda_visible_devices",
                    "0",
                ]
            )
            logging.info("Deployment started. Waiting for it to be ready...")

            # Wait for deployment to be ready
            if not wait_for_deployment_ready(host="0.0.0.0", port=8000, max_wait_time=180):
                assert False, "Deployment failed to become ready within timeout"

            time.sleep(120)

            # Text-only completions (no images)
            output = query_ray_deployment(
                host="0.0.0.0",
                port=8000,
                model_id="megatron-multimodal",
                prompt="What is the color of a banana?",
                max_tokens=5,
            )

            print(output)

            # Check if deployment was successful
            assert output != "", "First prediction is empty"

            # Text-only chat completions (no images)
            chat_messages = [
                {
                    "role": "user",
                    "content": [{"type": "text", "text": "Hello, how are you?"}],
                }
            ]
            output_chat = query_ray_deployment(
                host="0.0.0.0",
                port=8000,
                model_id="megatron-multimodal",
                prompt=chat_messages,
                max_tokens=5,
                use_chat=True,
            )
            print(output_chat)
            # Check if deployment was successful
            assert output_chat != "", "Second prediction (chat) is empty"

            # Chat with base64 image (base64)
            messages_with_image = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": BASE64_IMAGE},
                        },
                        {"type": "text", "text": "Describe the image:"},
                    ],
                }
            ]
            output_image = query_ray_chat_with_base64_image(
                host="0.0.0.0",
                port=8000,
                model_id="megatron-multimodal",
                messages=messages_with_image,
                max_tokens=5,
            )
            print(output_image)
            assert output_image != "", "Chat with base64 image returned empty"
        finally:
            # Ensure the deployment is terminated as soon as queries complete or on failure
            if self.deploy_proc is not None:
                terminate_deployment_process(self.deploy_proc)
                self.deploy_proc = None
