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

import logging
import signal
import subprocess
import time
from typing import Dict, List, Union

import requests

logger = logging.getLogger(__name__)


def terminate_deployment_process(process: subprocess.Popen | None) -> None:
    if process is None:
        return
    logger.info("Terminating Ray deployment process...")
    try:
        process.send_signal(signal.SIGTERM)
        try:
            process.wait(timeout=10)
            logger.info("Ray deployment terminated gracefully")
        except subprocess.TimeoutExpired:
            logger.warning("Graceful shutdown timed out, forcing termination...")
            process.kill()
            process.wait()
            logger.info("Ray deployment force terminated")
    except Exception as e:
        logger.error(f"Error terminating deployment: {e}")
        try:
            process.kill()
        except Exception:
            pass


def query_ray_deployment(
    host: str = "0.0.0.0",
    port: int = 8000,
    model_id: str = "model",
    prompt: Union[str, List[Dict[str, str]]] = "What is the color of a banana?",
    max_tokens: int = 20,
    temperature: float = 0.7,
    use_chat: bool = False,
) -> str:
    """Query the Ray deployment with a single prompt.

    When use_chat is False, sends to /v1/completions/ with a plain prompt.
    When use_chat is True, sends to /v1/chat/completions/ with a minimal messages payload.
    """
    base_url = f"http://{host}:{port}"

    try:
        if not use_chat:
            completions_url = f"{base_url}/v1/completions/"
            payload = {
                "model": model_id,
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            response = requests.post(completions_url, json=payload, timeout=30)
            if response.status_code == 200:
                result = response.json()
                text = result.get("choices", [{}])[0].get("text", "")
                logger.info(f"Completions endpoint response: {text}")
                return text
            else:
                logger.error(f"Completions endpoint error: {response.status_code} - {response.text}")
                return ""
        else:
            chat_url = f"{base_url}/v1/chat/completions/"
            payload = {
                "model": model_id,
                "messages": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "apply_chat_template": False,
            }
            response = requests.post(chat_url, json=payload, timeout=30)
            if response.status_code == 200:
                result = response.json()
                text = result.get("choices", [{}])[0].get("message", {}).get("content", "")
                logger.info(f"Chat completions endpoint response: {text}")
                return text
            else:
                logger.error(f"Chat completions endpoint error: {response.status_code} - {response.text}")
                return ""
    except Exception as e:
        logger.error(f"Exception during query: {e}")
        return ""


def wait_for_deployment_ready(
    host: str = "0.0.0.0",
    port: int = 8000,
    max_wait_time: int = 180,
    check_interval: int = 5,
) -> bool:
    """Wait for Ray deployment to be ready by checking health endpoint."""
    base_url = f"http://{host}:{port}"
    health_url = f"{base_url}/v1/health"

    start_time = time.time()
    while time.time() - start_time < max_wait_time:
        try:
            response = requests.get(health_url, timeout=10)
            if response.status_code == 200:
                logger.info("Deployment is ready!")
                return True
        except Exception as e:
            logger.info(f"Waiting for deployment... ({e})")

        time.sleep(check_interval)

    logger.error(f"Deployment not ready after {max_wait_time} seconds")
    return False
