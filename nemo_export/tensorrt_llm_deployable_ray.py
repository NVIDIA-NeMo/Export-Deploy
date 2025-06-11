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
import time
from typing import Any, Dict, List
import numpy as np

from fastapi import FastAPI, HTTPException

from nemo_deploy.ray_utils import find_available_port
from nemo_export.tensorrt_llm import TensorRTLLM

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
class TensorRTLLMRayDeployable:
    """A Ray Serve compatible wrapper for deploying TensorRT-LLM models.

    This class provides a standardized interface for deploying TensorRT-LLM models
    in Ray Serve. It supports various NLP tasks and handles model loading,
    inference, and deployment configurations.

    Args:
        model_dir (str): Path to the TensorRT-LLM model directory.
        model_id (str): Identifier for the model in the API responses. Defaults to "tensorrt-llm-model".
        max_batch_size (int): Maximum number of requests to batch together. Defaults to 8.
        batch_wait_timeout_s (float): Maximum time to wait for batching requests. Defaults to 0.3.
        load_model (bool): Whether to load the model during initialization. Defaults to True.
        use_python_runtime (bool): Whether to use Python runtime. Defaults to True.
        enable_chunked_context (bool): Whether to enable chunked context. Defaults to None.
        max_tokens_in_paged_kv_cache (int): Maximum tokens in paged KV cache. Defaults to None.
        multi_block_mode (bool): Whether to enable multi-block mode. Defaults to False.
    """

    def __init__(
        self,
        trt_llm_path: str,
        model_id: str = "tensorrt-llm-model",
        use_python_runtime: bool = True,
        enable_chunked_context: bool = None,
        max_tokens_in_paged_kv_cache: int = None,
        multi_block_mode: bool = False,
        lora_ckpt_list: List[str] = None,
    ):
        """Initialize the TensorRT-LLM model deployment.

        Args:
            model_dir (str): Path to the TensorRT-LLM model directory.
            model_id (str): Model identifier. Defaults to "tensorrt-llm-model".
            max_batch_size (int): Maximum number of requests to batch together. Defaults to 8.
            pipeline_parallelism_size (int): Number of pipeline parallelism. Defaults to 1.
            tensor_parallelism_size (int): Number of tensor parallelism. Defaults to 1.
            use_python_runtime (bool): Whether to use Python runtime. Defaults to True.
            enable_chunked_context (bool): Whether to enable chunked context. Defaults to None.
            max_tokens_in_paged_kv_cache (int): Maximum tokens in paged KV cache. Defaults to None.
            multi_block_mode (bool): Whether to enable multi-block mode. Defaults to False.
            lora_ckpt_list (List[str]): List of LoRA checkpoint paths. Defaults to None.

        Raises:
            ImportError: If Ray is not installed.
            Exception: If model initialization fails.
        """
        try:
            self.model = TensorRTLLM(
                model_dir=trt_llm_path,
                lora_ckpt_list=lora_ckpt_list,
                load_model=True,
                use_python_runtime=use_python_runtime,
                enable_chunked_context=enable_chunked_context,
                max_tokens_in_paged_kv_cache=max_tokens_in_paged_kv_cache,
                multi_block_mode=multi_block_mode,
            )
            self.model_id = model_id

        except Exception as e:
            LOGGER.error(f"Error initializing TensorRTLLMRayDeployable replica: {str(e)}")
            raise

    @app.post("/v1/completions/")
    async def completions(self, request: Dict[Any, Any]):
        """Handle text completion requests."""
        try:
            if "prompt" in request:
                request["prompts"] = [request["prompt"]]
            temperature = request.get("temperature", 0.0)
            top_p = request.get("top_p", 0.0)
            if temperature == 0.0 and top_p == 0.0:
                LOGGER.warning("Both temperature and top_p are 0. Setting top_k to 1 to ensure greedy sampling.")
                request["top_k"] = 1.0
            
            # Prepare inference inputs with proper parameter mapping
            inference_inputs = {
                "prompts": request.get("prompts", []),
                "max_length": request.get("max_tokens", 256),
                "temperature": request.get("temperature", 1.0),
                "top_k": request.get("top_k", 0),
                "top_p": request.get("top_p", 0.0),
                "compute_logprob": True if request.get("logprobs") == 1 else False,
                "apply_chat_template": False,
            }
            
            results = self.model.ray_infer_fn(inference_inputs)
            # Extract generated texts from results
            generated_texts_raw = results.get("sentences", [])
            
            # Flatten the nested list structure - sentences is a list of lists
            generated_texts = []
            for batch in generated_texts_raw:
                if isinstance(batch, list):
                    generated_texts.extend(batch)
                else:
                    generated_texts.append(batch)
            
            # Calculate token counts asynchronously
            prompt_tokens = sum(len(p.split()) for p in request.get("prompts", []))
            completion_tokens = sum(len(str(r).split()) for r in generated_texts)
            total_tokens = prompt_tokens + completion_tokens
            
            # Convert numpy arrays to Python lists for JSON serialization
            log_probs_data = results.get("log_probs", None)
            if log_probs_data is not None and isinstance(log_probs_data, np.ndarray):
                log_probs_data = log_probs_data.tolist()
            
            output = {
                "id": f"cmpl-{int(time.time())}",
                "object": "text_completion",
                "created": int(time.time()),
                "model": self.model_id,
                "choices": [
                    {
                        "text": " ".join(str(t) for t in generated_texts),
                        "index": 0,
                        "logprobs": (
                            {
                                "token_logprobs": log_probs_data,
                                "top_logprobs": log_probs_data,
                            }
                            if log_probs_data is not None
                            else None
                        ),
                        "finish_reason": (
                            "length" if generated_texts and len(str(generated_texts[0])) >= request.get('max_tokens', 256) else "stop"
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
            raise HTTPException(status_code=500, detail=f"Error during inference: {str(e)}")

    @app.post("/v1/chat/completions/")
    async def chat_completions(self, request: Dict[Any, Any]):
        """Handle chat completion requests."""
        try:
            # Extract parameters from the request dictionary
            messages = request.get('messages', [])

            inference_inputs = {
                "prompts": [messages],  # Wrap messages in a list so apply_chat_template gets the full conversation
                "max_length": request.get("max_tokens", 256),
                "temperature": request.get("temperature", 1.0),
                "top_k": request.get("top_k", 0),
                "top_p": request.get("top_p", 0.0),
                "compute_logprob": True if request.get("logprobs") == 1 else False,
                "apply_chat_template": request.get("apply_chat_template", True),
            }

            # Run model inference in the thread pool
            results = self.model.ray_infer_fn(inference_inputs)

            # Extract generated texts from results
            generated_texts_raw = results["sentences"]
            
            # Flatten the nested list structure - sentences is a list of lists
            generated_texts = []
            for batch in generated_texts_raw:
                if isinstance(batch, list):
                    generated_texts.extend(batch)
                else:
                    generated_texts.append(batch)

            # Calculate token counts
            prompt_tokens = sum(len(str(msg).split()) for msg in messages)
            completion_tokens = sum(len(str(r).split()) for r in generated_texts)
            total_tokens = prompt_tokens + completion_tokens

            # Convert numpy arrays to Python lists for JSON serialization
            log_probs_data = results.get("log_probs", None)
            if log_probs_data is not None and isinstance(log_probs_data, np.ndarray):
                log_probs_data = log_probs_data.tolist()

            output = {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": self.model_id,
                "choices": [
                    {
                        "message": {"role": "assistant", "content": str(generated_texts[0]) if generated_texts else ""},
                        "index": 0,
                        "logprobs": (
                            {
                                "token_logprobs": log_probs_data,
                                "top_logprobs": log_probs_data,
                            }
                            if log_probs_data is not None
                            else None
                        ),
                        "finish_reason": (
                            "length" if generated_texts and len(str(generated_texts[0])) >= inference_inputs["max_length"] else "stop"
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
            raise HTTPException(status_code=500, detail=f"Error during chat completion: {str(e)}")

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
