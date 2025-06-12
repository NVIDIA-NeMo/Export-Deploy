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


from ray import serve
import logging
import numpy as np
import time
from typing import Any, Dict

import torch
from fastapi import FastAPI, HTTPException

from nemo_deploy.ray_utils import find_available_port
from nemo_deploy.nlp.hf_deployable import HuggingFaceLLMDeploy

LOGGER = logging.getLogger("NeMo")

app = FastAPI()


@serve.deployment(
    num_replicas=1,  # One replica per GPU
    ray_actor_options={
        "num_gpus": 1,  # Each replica gets 1 GPU
        "num_cpus": 8,
    },
    max_ongoing_requests=10,
)
@serve.ingress(app)
class HFRayDeployable:
    """A Ray Serve compatible wrapper for deploying HuggingFace models.

    This class provides a standardized interface for deploying HuggingFace models
    in Ray Serve. It supports various NLP tasks and handles model loading,
    inference, and deployment configurations.

    Args:
        hf_model_id_path (str): Path to the HuggingFace model or model identifier.
            Can be a local path or a model ID from HuggingFace Hub.
        task (str): HuggingFace task type (e.g., "text-generation"). Defaults to "text-generation".
        trust_remote_code (bool): Whether to trust remote code when loading the model. Defaults to True.
        device_map (str): Device mapping strategy for model placement. Defaults to "auto".
        tp_plan (str): Tensor parallelism plan for distributed inference. Defaults to None.
        model_id (str): Identifier for the model in the API responses. Defaults to "nemo-model".
    """

    def __init__(
        self,
        hf_model_id_path: str,
        task: str = "text-generation",
        trust_remote_code: bool = True,
        model_id: str = "nemo-model",
        device_map: str = "auto",
        max_memory: str = None
    ):
        """Initialize the HuggingFace model deployment.

        Args:
            hf_model_id_path (str): Path to the HuggingFace model or model identifier.
            task (str): HuggingFace task type. Defaults to "text-generation".
            trust_remote_code (bool): Whether to trust remote code. Defaults to True.
            device_map (str): Device mapping strategy. Defaults to "auto".
            model_id (str): Model identifier. Defaults to "nemo-model".
            max_memory (str): Maximum memory allocation when using balanced device map.

        Raises:
            ImportError: If Ray is not installed.
            Exception: If model initialization fails.
        """
        try:
            max_memory_dict = None
            self._setup_unique_distributed_parameters(device_map)
            if device_map == "balanced":
                if not max_memory:
                    raise ValueError(
                        "max_memory must be provided when device_map is 'balanced'"
                    )
                num_gpus = torch.cuda.device_count()
                if num_gpus > 1:
                    print(
                        f"Using tensor parallel across {num_gpus} GPUs for large model"
                    )
                    max_memory_dict = {i: "75GiB" for i in range(num_gpus)}
            self.model = HuggingFaceLLMDeploy(
                hf_model_id_path=hf_model_id_path,
                task=task,
                trust_remote_code=trust_remote_code,
                device_map=device_map,
                max_memory=max_memory_dict,
            )
            self.model_id = model_id

        except Exception as e:
            LOGGER.error(f"Error initializing HuggingFaceLLMServe replica: {str(e)}")
            raise

    def _setup_unique_distributed_parameters(self, device_map):
        """Configure unique distributed communication parameters for each model replica.

        This function sets up unique MASTER_PORT environment variables for each Ray Serve
        replica to ensure they can initialize their own torch.distributed process groups
        without port conflicts. Only runs for 'balanced' or 'auto' device maps.

        Args:
            device_map (str): The device mapping strategy ('auto', 'balanced', etc.)
        """
        if device_map == "balanced" or device_map == "auto":
            import os

            import torch.distributed as dist

            # Check if torch.distributed is already initialized
            if not dist.is_initialized():
                # Get a unique port based on current process ID to avoid conflicts

                unique_port = find_available_port(29500, "127.0.0.1")
                # Set environment variables for torch.distributed
                os.environ["MASTER_ADDR"] = "127.0.0.1"
                os.environ["MASTER_PORT"] = str(unique_port)

    @app.post("/v1/completions/")
    async def completions(self, request: Dict[Any, Any]):
        """Handle text completion requests.

        This endpoint processes text completion requests in OpenAI API format and returns
        generated completions with token usage information.

        Args:
            request (Dict[Any, Any]): Request dictionary containing:
                - prompts: List of input prompts
                - max_tokens: Maximum tokens to generate (optional)
                - temperature: Sampling temperature (optional)
                - top_k: Top-k sampling parameter (optional)
                - top_p: Top-p sampling parameter (optional)
                - model: Model identifier (optional)

        Returns:
            Dict containing:
                - id: Unique completion ID
                - object: Response type ("text_completion")
                - created: Timestamp
                - model: Model identifier
                - choices: List of completion choices
                - usage: Token usage statistics

        Raises:
            HTTPException: If inference fails.
        """
        try:
            if "prompt" in request:
                request["prompts"] = [request["prompt"]]
            temperature = request.get("temperature", 0.0)
            top_p = request.get("top_p", 0.0)
            if temperature == 0.0 and top_p == 0.0:
                LOGGER.warning(
                    "Both temperature and top_p are 0. Setting top_k to 1 to ensure greedy sampling."
                )
                request["top_k"] = 1.0
            
            inference_inputs = {
                "prompts": request.get("prompts", []),
                "max_length": request.get("max_tokens", 256),
                "temperature": temperature,
                "top_k": request.get("top_k", 0),
                "top_p": request.get("top_p", 0.0),
                "output_logits": request.get("output_logits", False),
                "output_scores": request.get("output_scores", False),
            }

            # Run tokenization and model inference in the thread pool
            results = self.model.ray_infer_fn(inference_inputs)
            # Extract generated texts from results
            generated_texts = results.get("sentences", [])

            # Calculate token counts asynchronously
            prompt_tokens = sum(len(p.split()) for p in request.get("prompts", []))
            completion_tokens = sum(len(r.split()) for r in generated_texts)
            total_tokens = prompt_tokens + completion_tokens

            # Convert numpy arrays to Python lists for JSON serialization
            scores = results.get("scores", None)
            if scores is not None and isinstance(scores, np.ndarray):
                scores = scores.tolist()
            
            logits = results.get("logits", None)
            if logits is not None and isinstance(logits, np.ndarray):
                logits = logits.tolist()

            output = {
                "id": f"cmpl-{int(time.time())}",
                "object": "text_completion",
                "created": int(time.time()),
                "model": self.model_id,
                "choices": [
                    {
                        "text": " ".join(generated_texts),
                        "index": 0,
                        "logprobs": (
                            {
                                "scores": scores,
                            }
                            if scores is not None
                            else None,
                            {
                                "logits": logits,
                            }
                            if logits is not None
                            else None,
                        ),
                        "finish_reason": (
                            "length"
                            if generated_texts
                            and len(generated_texts[0])
                            >= request.get("max_tokens", 256)
                            else "stop"
                        ),
                    }
                ],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                },
            }
            return output
        except Exception as e:
            LOGGER.error(f"Error during inference: {str(e)}")
            raise HTTPException(
                status_code=500, detail=f"Error during inference: {str(e)}"
            )

    @app.post("/v1/chat/completions/")
    async def chat_completions(self, request: Dict[Any, Any]):
        """Handle chat completion requests.

        This endpoint processes chat completion requests in OpenAI API format and returns
        generated responses with token usage information.

        Args:
            request (Dict[Any, Any]): Request dictionary containing:
                - messages: List of chat messages
                - max_tokens: Maximum tokens to generate (optional)
                - temperature: Sampling temperature (optional)
                - top_k: Top-k sampling parameter (optional)
                - top_p: Top-p sampling parameter (optional)
                - model: Model identifier (optional)

        Returns:
            Dict containing:
                - id: Unique chat completion ID
                - object: Response type ("chat.completion")
                - created: Timestamp
                - model: Model identifier
                - choices: List of chat completion choices
                - usage: Token usage statistics

        Raises:
            HTTPException: If inference fails.
        """
        try:
            # Extract parameters from the request dictionary
            messages = request.get("messages", [])

            # Convert messages to a single prompt
            prompt = "\n".join(
                [
                    f"{msg.get('role', 'user')}: {msg.get('content', '')}"
                    for msg in messages
                ]
            )
            prompt += "\nassistant:"

            # Create a modified request with the prompt
            chat_request = request.copy()
            chat_request["prompt"] = prompt

            # Extract parameters from the request dictionary
            messages = request.get("messages", [])

            # Prepare inference parameters
            inference_inputs = {
                "prompts": [
                    messages
                ],  # Wrap messages in a list so apply_chat_template gets the full conversation
                "max_length": request.get("max_tokens", 256),
                "temperature": request.get("temperature", 1.0),
                "top_k": request.get("top_k", 0),
                "top_p": request.get("top_p", 0.0),
                "output_logits": request.get("output_logits", False),
                "output_scores": request.get("output_scores", False)
            }

            # Run model inference in the thread pool
            results = self.model.ray_infer_fn(inference_inputs)
            # Extract generated texts from results
            generated_texts = results["sentences"]

            # Calculate token counts
            prompt_tokens = sum(len(str(msg).split()) for msg in messages)
            completion_tokens = sum(len(r.split()) for r in generated_texts)
            total_tokens = prompt_tokens + completion_tokens

            # Convert numpy arrays to Python lists for JSON serialization
            scores = results.get("scores", None)
            if scores is not None and isinstance(scores, np.ndarray):
                scores = scores.tolist()
            
            logits = results.get("logits", None)
            if logits is not None and isinstance(logits, np.ndarray):
                logits = logits.tolist()


            output = {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": self.model_id,
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": generated_texts[0] if generated_texts else "",
                        },
                        "index": 0,
                        "logprobs": (
                             {
                                "scores": scores,
                            }
                            if scores is not None
                            else None,
                            {
                                "logits": logits,
                            }
                            if logits is not None
                            else None
                        ),
                        "finish_reason": (
                            "length"
                            if generated_texts
                            and len(generated_texts[0])
                            >= inference_inputs["max_length"]
                            else "stop"
                        ),
                    }
                ],
                "usage": {
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                },
            }
            return output
        except Exception as e:
            LOGGER.error(f"Error during chat completion: {str(e)}")
            raise HTTPException(
                status_code=500, detail=f"Error during chat completion: {str(e)}"
            )

    @app.get("/v1/models")
    async def list_models(self):
        """List available models.

        This endpoint returns information about the deployed model in OpenAI API format.

        Returns:
            Dict containing:
                - object: Response type ("list")
                - data: List of model information
        """
        return {
            "object": "list",
            "data": [
                {"id": self.model_id, "object": "model", "created": int(time.time())}
            ],
        }

    @app.get("/v1/health")
    async def health_check(self):
        """Check the health status of the service.

        This endpoint is used to verify that the service is running and healthy.

        Returns:
            Dict containing:
                - status: Health status ("healthy")
        """
        return {"status": "healthy"}
