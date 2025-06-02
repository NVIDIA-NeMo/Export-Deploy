#!/usr/bin/env python3
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
import os
import time
from typing import Any, Dict

import numpy as np
import ray
import torch
from fastapi import FastAPI, HTTPException
from ray import serve

from .megatronllm_deployable import MegatronLLMDeployableNemo2
from ..ray_utils import find_available_port

LOGGER = logging.getLogger("NeMo")

app = FastAPI()

@ray.remote(num_gpus=1)
class ModelWorker:
    """Ray actor that loads and runs inference on a shard of the model.
    
    Each ModelWorker is responsible for a specific rank in the model parallel setup.
    """  
    def __init__(
        self,
        nemo_checkpoint_filepath: str,
        rank: int,
        world_size: int,
        tensor_model_parallel_size: int,
        pipeline_model_parallel_size: int,
        context_parallel_size: int,
        expert_model_parallel_size: int,
        master_port: str,
        replica_id: int = 0,
        enable_cuda_graphs: bool = False,
        enable_flash_decode: bool = False,
    ):
        # Use replica-specific environment variables to avoid conflicts
        os.environ["MASTER_PORT"] = master_port
        os.environ["MASTER_ADDR"] = ray._private.services.get_node_ip_address()
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(rank % torch.cuda.device_count())
        
        # Set a unique process group name for each replica to avoid conflicts
        os.environ["TORCH_DISTRIBUTED_GROUP_NAME"] = f"replica_{replica_id}"
        
        # Use INFO level logging only for important initialization steps
        if rank == 0:  # Only log from rank 0 to reduce noise
            LOGGER.info(f"Replica {replica_id} - Initializing workers for world_size={world_size}")
            LOGGER.info(f"Replica {replica_id} - MASTER_PORT: {os.environ['MASTER_PORT']}")
            LOGGER.info(f"Replica {replica_id} - MASTER_ADDR: {os.environ['MASTER_ADDR']}")
        
        try:
            self.model = MegatronLLMDeployableNemo2(
                nemo_checkpoint_filepath=nemo_checkpoint_filepath,
                num_devices=world_size,
                num_nodes=world_size // torch.cuda.device_count(),
                tensor_model_parallel_size=tensor_model_parallel_size,
                pipeline_model_parallel_size=pipeline_model_parallel_size,
                expert_model_parallel_size=expert_model_parallel_size,
                context_parallel_size=context_parallel_size,
                enable_cuda_graphs=enable_cuda_graphs,
                enable_flash_decode=enable_flash_decode,
            )
            if rank != 0:
                self.model.generate_other_ranks()
        except Exception as e:
            LOGGER.error(f"Replica {replica_id} - Failed to initialize model for rank {rank}: {str(e)}")
            raise
        
    def infer(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Run inference on the model shard."""
        return self.model.ray_infer_fn(inputs)


@serve.deployment(
    num_replicas=1,
    ray_actor_options={
        "num_cpus": 8  
    },
    max_ongoing_requests=32,  
)
@serve.ingress(app)
class MegatronRayDeployable:
    """A Ray Serve deployment for distributed Megatron LLM models.
    
    This class coordinates model parallelism across multiple GPUs and nodes,
    with each shard handled by a separate Ray actor.
    """
    
    def __init__(
        self,
        nemo_checkpoint_filepath: str,
        num_gpus: int = 1,
        num_nodes: int = 1,
        tensor_model_parallel_size: int = 1,
        pipeline_model_parallel_size: int = 1,
        context_parallel_size: int = 1,
        expert_model_parallel_size: int = 1,
        model_id: str = "nemo-model",
        enable_cuda_graphs: bool = False,
        enable_flash_decode: bool = False
    ):
        """Initialize the distributed Megatron LLM model deployment.
        
        Args:
            nemo_checkpoint_filepath (str): Path to the .nemo checkpoint file.
            num_gpus (int): Number of GPUs to use per replica.
            num_nodes (int): Number of nodes to use for deployment.
            tensor_model_parallel_size (int): Size of tensor model parallelism.
            pipeline_model_parallel_size (int): Size of pipeline model parallelism.
            context_parallel_size (int): Size of context parallelism.
            model_id (str): Identifier for the model in API responses.
            enable_cuda_graphs (bool): Whether to enable CUDA graphs for faster inference.
            enable_flash_decode (bool): Whether to enable Flash Attention decode.
            max_batch_size (int): Maximum batch size for request batching.
            batch_wait_timeout_s (float): Maximum time to wait for batching requests.
        """
        try:
            self.model_id = model_id
            world_size = num_gpus * num_nodes
            
            # Validate parallelism configuration
            total_parallel_size = tensor_model_parallel_size * pipeline_model_parallel_size * context_parallel_size
            if total_parallel_size != world_size:
                raise ValueError(
                    f"Total parallelism size ({total_parallel_size}) must equal total GPUs per replica ({world_size})"
                )
            
            # Generate a unique replica ID based on the actor handle
            replica_id = abs(hash(str(self))) % 10000
            
            # Pre-allocate master port to avoid race conditions between workers
            # Use replica-specific port to avoid conflicts between replicas
            base_port = 29500 + (replica_id % 100) * 100
            master_port = str(find_available_port(base_port, ray._private.services.get_node_ip_address()))
            LOGGER.info(f"Replica {replica_id} - Pre-allocated master port: {master_port}")
            
            # Create workers with proper synchronization for distributed initialization
            # Rank 0 must be created first as it acts as the master in PyTorch distributed
            worker_futures = []
            
            # Create rank 0 worker first
            rank_0_worker = ModelWorker.remote(
                nemo_checkpoint_filepath=nemo_checkpoint_filepath,
                rank=0,
                world_size=world_size,
                tensor_model_parallel_size=tensor_model_parallel_size,
                pipeline_model_parallel_size=pipeline_model_parallel_size,
                context_parallel_size=context_parallel_size,
                expert_model_parallel_size=expert_model_parallel_size,
                master_port=master_port,
                replica_id=replica_id,
                enable_cuda_graphs=enable_cuda_graphs,
                enable_flash_decode=enable_flash_decode,
            )
            worker_futures.append(rank_0_worker)
            
            # Wait for rank 0 to start before creating other workers
            # This ensures the master node is ready for distributed initialization
            LOGGER.info(f"Replica {replica_id} - Waiting for rank 0 to initialize...")
            time.sleep(1)  # Give rank 0 time to start the distributed backend
            
            # Create remaining workers in parallel
            for rank in range(1, world_size):
                worker = ModelWorker.remote(
                    nemo_checkpoint_filepath=nemo_checkpoint_filepath,
                    rank=rank,
                    world_size=world_size,
                    tensor_model_parallel_size=tensor_model_parallel_size,
                    pipeline_model_parallel_size=pipeline_model_parallel_size,
                    context_parallel_size=context_parallel_size,
                    expert_model_parallel_size=expert_model_parallel_size,
                    master_port=master_port,
                    replica_id=replica_id,
                    enable_cuda_graphs=enable_cuda_graphs,
                    enable_flash_decode=enable_flash_decode,
                )
                worker_futures.append(worker)
            
            # Wait for all workers to be created and store them
            self.workers = worker_futures
            LOGGER.info(f"Replica {replica_id} - All {world_size} workers created successfully")
            
            # Primary worker for coordinating inference
            self.primary_worker = self.workers[0]
            
            LOGGER.info(f"Replica {replica_id} - Initialized {world_size} model workers across {num_nodes} nodes")
            
        except Exception as e:
            LOGGER.error(f"Error initializing distributed model deployment: {str(e)}")
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
            
            # Run tokenization and model inference in the thread pool
            results = ray.get(self.primary_worker.infer.remote(inference_inputs))
            LOGGER.error(f"Results: {results}")
            # Extract generated texts from results
            generated_texts = results.get("sentences", [])

            # Calculate token counts asynchronously
            prompt_tokens = sum(len(p.split()) for p in request.get("prompts", []))
            completion_tokens = sum(len(r.split()) for r in generated_texts)
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
                        "text": " ".join(generated_texts),
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
                            "length" if generated_texts and len(generated_texts[0]) >= request.get('max_tokens', 256) else "stop"
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

            # Prepare inference parameters
            # For chat templates, we need to pass the entire messages list as a single prompt
            # so that apply_chat_template receives the full conversation context
            inference_inputs = {
                "prompts": [messages],  # Wrap messages in a list so apply_chat_template gets the full conversation
                "max_length": request.get("max_tokens", 256),
                "temperature": request.get("temperature", 1.0),
                "top_k": request.get("top_k", 0),
                "top_p": request.get("top_p", 0.0),
                "compute_logprob": True if request.get("logprobs") == 1 else False,
                "apply_chat_template": True,
            }

            # Run model inference in the thread pool
            results = ray.get(self.primary_worker.infer.remote(inference_inputs))

            # Extract generated texts from results
            generated_texts = results["sentences"]

            # Calculate token counts
            prompt_tokens = sum(len(str(msg).split()) for msg in messages)
            completion_tokens = sum(len(r.split()) for r in generated_texts)
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
                        "message": {"role": "assistant", "content": generated_texts[0] if generated_texts else ""},
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
                            "length" if generated_texts and len(generated_texts[0]) >= inference_inputs["max_length"] else "stop"
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
        """List available models."""
        return {
            "data": [
                {
                    "id": self.model_id,
                    "object": "model",
                    "created": int(time.time()),
                    "owned_by": "nvidia",
                }
            ],
            "object": "list",
        }

    @app.get("/v1/health")
    async def health_check(self):
        """Health check endpoint."""
        return {"status": "healthy"}