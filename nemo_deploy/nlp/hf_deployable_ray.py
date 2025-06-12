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
import asyncio
import logging
import time
from typing import Any, Dict, List

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
        max_memory: str = None,
        max_batch_size: int = 8,
        batch_wait_timeout_s: float = 0.3,
    ):
        """Initialize the HuggingFace model deployment.

        Args:
            hf_model_id_path (str): Path to the HuggingFace model or model identifier.
            task (str): HuggingFace task type. Defaults to "text-generation".
            trust_remote_code (bool): Whether to trust remote code. Defaults to True.
            device_map (str): Device mapping strategy. Defaults to "auto".
            model_id (str): Model identifier. Defaults to "nemo-model".
            max_memory (str): Maximum memory allocation when using balanced device map.
            max_batch_size (int): Maximum number of requests to batch together. Defaults to 8.
            batch_wait_timeout_s (float): Maximum time to wait for batching requests. Defaults to 0.3.

        Raises:
            ImportError: If Ray is not installed.
            Exception: If model initialization fails.
        """
        try:
            max_memory_dict = None
            self._setup_unique_distributed_parameters(device_map)
            if device_map == "balanced":
                if not max_memory:
                    raise ValueError("max_memory must be provided when device_map is 'balanced'")
                num_gpus = torch.cuda.device_count()
                if num_gpus > 1:
                    print(f"Using tensor parallel across {num_gpus} GPUs for large model")
                    max_memory_dict = {i: "75GiB" for i in range(num_gpus)}
            self.model = HuggingFaceLLMDeploy(
                hf_model_id_path=hf_model_id_path,
                task=task,
                trust_remote_code=trust_remote_code,
                device_map=device_map,
                max_memory=max_memory_dict,
            )
            self.model_id = model_id
            self.max_batch_size = max_batch_size
            self.batch_wait_timeout_s = batch_wait_timeout_s

            # Dynamically apply the serve.batch decorator with the user-configured parameters
            # This allows the batch size and timeout to be configured when instantiating the class
            self.batched_inference = serve.batch(
                max_batch_size=self.max_batch_size, batch_wait_timeout_s=self.batch_wait_timeout_s
            )(self._batched_inference)

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
            # Call the batched method
            result = await self.batched_inference(request, "completion")

            if "error" in result:
                raise HTTPException(status_code=500, detail=result["error"])

            return result
        except Exception as e:
            LOGGER.error(f"Error during inference: {str(e)}")
            LOGGER.error(f"Request: {request}")
            raise HTTPException(status_code=500, detail=f"Error during inference: {str(e)}")

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
            prompt = "\n".join([f"{msg.get('role', 'user')}: {msg.get('content', '')}" for msg in messages])
            prompt += "\nassistant:"

            # Create a modified request with the prompt
            chat_request = request.copy()
            chat_request["prompt"] = prompt

            # Call the batched method with a single request
            results = await self.batched_inference(chat_request, "chat")

            if not results or len(results) == 0:
                raise HTTPException(status_code=500, detail="No results returned from model")

            return results

        except Exception as e:
            LOGGER.error(f"Error during chat completion: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error during chat completion: {str(e)}")

    async def _batched_inference(self, requests: List[Dict[Any, Any]], request_type: str = "completion"):
        """Internal method for processing batched inference requests.

        This method is decorated with serve.batch in the constructor with the configured
        max_batch_size and batch_wait_timeout_s parameters. It's called by the completions
        and chat_completions endpoints as self.batched_inference, which is the decorated version.

        Args:
            requests (List[Dict[Any, Any]]): List of request dictionaries.
            request_type (str): Type of request, either "completion" or "chat".

        Returns:
            List[Dict]: List of results for each request.
        """
        LOGGER.error(f"Received {len(requests)} {request_type} requests")

        if not requests:
            return []

        try:
            # Extract parameters from the first request
            first_request = requests[0]
            model_name = first_request.get("model", "nemo-model")
            max_length = first_request.get("max_tokens", 256)
            temperature = first_request.get("temperature", 1.0)
            top_k = first_request.get("top_k", 1)

            # Collect all prompts from all requests
            all_prompts = [request.get("prompt", "") for request in requests]

            LOGGER.error(f"Combined {len(all_prompts)} prompts")

            # Prepare a single inference input with all prompts
            inference_inputs = {
                "prompts": all_prompts,
                "max_length": max_length,
                "temperature": temperature,
                "top_k": top_k,
                "top_p": 0.0,
                "compute_logprob": request_type == "completion",  # Only compute logprobs for text completions
                "apply_chat_template": False,
            }

            # Run model inference once with all prompts
            loop = asyncio.get_event_loop()
            combined_result = await loop.run_in_executor(None, self.model.ray_infer_fn, inference_inputs)

            # Distribute results back to individual responses
            results = []

            for i, request in enumerate(requests):
                try:
                    # Get this request's result
                    request_sentence = combined_result["sentences"][i]

                    # Calculate token counts
                    prompt_tokens = len(all_prompts[i].split())
                    completion_tokens = len(request_sentence.split())
                    total_tokens = prompt_tokens + completion_tokens

                    # Get log probs if available
                    log_probs = None
                    if "log_probs" in combined_result and i < len(combined_result["log_probs"]):
                        log_probs = combined_result["log_probs"][i]

                    # Format response based on request type
                    if request_type == "chat":
                        output = {
                            "id": f"chatcmpl-{int(time.time())}",
                            "object": "chat.completion",
                            "created": int(time.time()),
                            "model": model_name,
                            "choices": [
                                {
                                    "message": {"role": "assistant", "content": request_sentence},
                                    "index": 0,
                                    "finish_reason": ("length" if len(request_sentence) >= max_length else "stop"),
                                }
                            ],
                            "usage": {
                                "prompt_tokens": prompt_tokens,
                                "completion_tokens": completion_tokens,
                                "total_tokens": total_tokens,
                            },
                        }
                    else:  # completion
                        output = {
                            "id": f"cmpl-{int(time.time())}",
                            "object": "text_completion",
                            "created": int(time.time()),
                            "model": model_name,
                            "choices": [
                                {
                                    "text": request_sentence,
                                    "index": 0,
                                    "logprobs": log_probs,
                                    "finish_reason": ("length" if len(request_sentence) >= max_length else "stop"),
                                }
                            ],
                            "usage": {
                                "prompt_tokens": prompt_tokens,
                                "completion_tokens": completion_tokens,
                                "total_tokens": total_tokens,
                            },
                        }

                    results.append(output)

                except Exception as e:
                    LOGGER.error(f"Error processing {request_type} result for request {i}: {str(e)}")
                    results.append({"error": str(e)})

        except Exception as e:
            LOGGER.error(f"Error in batched {request_type} processing: {str(e)}")
            # If the batched approach fails, return error for all requests
            results = [{"error": str(e)} for _ in requests]

        return results

    @app.get("/v1/models")
    async def list_models(self):
        """List available models.

        This endpoint returns information about the deployed model in OpenAI API format.

        Returns:
            Dict containing:
                - object: Response type ("list")
                - data: List of model information
        """
        return {"object": "list", "data": [{"id": self.model_id, "object": "model", "created": int(time.time())}]}

    @app.get("/v1/health")
    async def health_check(self):
        """Check the health status of the service.

        This endpoint is used to verify that the service is running and healthy.

        Returns:
            Dict containing:
                - status: Health status ("healthy")
        """
        return {"status": "healthy"}
