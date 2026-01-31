#!/usr/bin/env python3
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

import json
import logging
import os
import random
import time
from typing import Any, Dict, Optional

import ray
import torch
from fastapi import FastAPI, HTTPException
from ray import serve
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from nemo_deploy.ray_utils import find_available_port

from .megatron_multimodal_deployable import MegatronMultimodalDeployable

LOGGER = logging.getLogger("NeMo")

app = FastAPI()


@ray.remote(num_gpus=1)
class ModelWorker:
    """Ray actor that loads and runs inference on a shard of the multimodal model.

    Each ModelWorker is responsible for a specific rank in the model parallel setup.
    """

    def __init__(
        self,
        megatron_checkpoint_filepath: str,
        rank: int,
        world_size: int,
        tensor_model_parallel_size: int,
        pipeline_model_parallel_size: int,
        master_port: str,
        master_addr: Optional[str] = None,
        replica_id: int = 0,
        **model_config_kwargs,
    ):
        # Use replica-specific environment variables to avoid conflicts
        os.environ["MASTER_PORT"] = master_port
        # All ranks must use the SAME MASTER_ADDR (rank 0 node IP)
        os.environ["MASTER_ADDR"] = master_addr if master_addr else ray._private.services.get_node_ip_address()
        os.environ["RANK"] = str(rank)
        os.environ["WORLD_SIZE"] = str(world_size)
        os.environ["LOCAL_RANK"] = str(rank % torch.cuda.device_count())

        # Set a unique process group name for each replica to avoid conflicts
        os.environ["TORCH_DISTRIBUTED_GROUP_NAME"] = f"replica_{replica_id}"

        # Use INFO level logging only for important initialization steps
        if rank == 0:  # Only log from rank 0 to reduce noise
            LOGGER.info(f"Replica {replica_id} - Initializing multimodal workers for world_size={world_size}")
            LOGGER.info(f"Replica {replica_id} - MASTER_PORT: {os.environ['MASTER_PORT']}")
            LOGGER.info(f"Replica {replica_id} - MASTER_ADDR: {os.environ['MASTER_ADDR']}")

        try:
            self.model = MegatronMultimodalDeployable(
                megatron_checkpoint_filepath=megatron_checkpoint_filepath,
                tensor_model_parallel_size=tensor_model_parallel_size,
                pipeline_model_parallel_size=pipeline_model_parallel_size,
                **model_config_kwargs,
            )
            self.rank = rank
        except Exception as e:
            LOGGER.error(f"Replica {replica_id} - Failed to initialize multimodal model for rank {rank}: {str(e)}")
            raise

    def infer(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Run inference on the model shard."""
        return self.model.ray_infer_fn(inputs)


@serve.deployment(
    num_replicas=1,
    ray_actor_options={"num_cpus": 8},
    max_ongoing_requests=32,
)
@serve.ingress(app)
class MegatronMultimodalRayDeployable:
    """A Ray Serve deployment for distributed Megatron multimodal models.

    This class coordinates model parallelism across multiple GPUs and nodes,
    with each shard handled by a separate Ray actor.
    """

    def __init__(
        self,
        megatron_checkpoint_filepath: str,
        num_gpus: int = 1,
        tensor_model_parallel_size: int = 1,
        pipeline_model_parallel_size: int = 1,
        model_id: str = "megatron-model",
        **model_config_kwargs,
    ):
        """Initialize the distributed Megatron multimodal model deployment.

        Args:
            megatron_checkpoint_filepath (str): Path to the Megatron checkpoint directory.
            num_gpus (int): Number of GPUs to use for the deployment
            tensor_model_parallel_size (int): Size of tensor model parallelism.
            pipeline_model_parallel_size (int): Size of pipeline model parallelism.
            model_id (str): Identifier for the model in API responses.
            **model_config_kwargs: Additional model configuration arguments.
        """
        try:
            self.model_id = model_id

            # Generate a unique replica ID based on the actor handle
            replica_id = abs(hash(str(self))) % 10000

            # Pre-allocate master port to avoid race conditions between workers
            # Use replica-specific port to avoid conflicts between replicas
            base_port = random.randint(29500, 29999) + (replica_id % 100) * 100
            deploy_node_ip = ray._private.services.get_node_ip_address()
            master_port = str(find_available_port(base_port, deploy_node_ip))
            LOGGER.info(f"Replica {replica_id} - Pre-allocated master port: {master_port}")

            # Create workers with proper synchronization for distributed initialization
            # Rank 0 must be created first as it acts as the master in PyTorch distributed
            worker_futures = []

            # Create rank 0 worker first
            # Force rank 0 to run on the same node as this deployment so MASTER_ADDR is routable
            # Resolve the node_id for this deployment's node
            deployment_node_id = None
            for node in ray.nodes():
                if node.get("Alive") and node.get("NodeManagerAddress") == deploy_node_ip:
                    deployment_node_id = node.get("NodeID")
                    break

            # Common arguments for rank 0 worker
            rank_0_kwargs = dict(
                megatron_checkpoint_filepath=megatron_checkpoint_filepath,
                rank=0,
                world_size=num_gpus,
                tensor_model_parallel_size=tensor_model_parallel_size,
                pipeline_model_parallel_size=pipeline_model_parallel_size,
                master_port=master_port,
                master_addr=deploy_node_ip,
                replica_id=replica_id,
                **model_config_kwargs,
            )

            # Use node affinity if we found a matching node, otherwise use default scheduling
            if deployment_node_id is not None:
                rank_0_worker = ModelWorker.options(
                    scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=deployment_node_id, soft=True)
                ).remote(**rank_0_kwargs)
            else:
                rank_0_worker = ModelWorker.remote(**rank_0_kwargs)
            worker_futures.append(rank_0_worker)

            # Wait for rank 0 to start before creating other workers
            # This ensures the master node is ready for distributed initialization
            LOGGER.info(f"Replica {replica_id} - Waiting for rank 0 to initialize...")
            time.sleep(1)  # Give rank 0 time to start the distributed backend

            # Create remaining workers in parallel
            for rank in range(1, num_gpus):
                worker = ModelWorker.remote(
                    megatron_checkpoint_filepath=megatron_checkpoint_filepath,
                    rank=rank,
                    world_size=num_gpus,
                    tensor_model_parallel_size=tensor_model_parallel_size,
                    pipeline_model_parallel_size=pipeline_model_parallel_size,
                    master_port=master_port,
                    master_addr=deploy_node_ip,
                    replica_id=replica_id,
                    **model_config_kwargs,
                )
                worker_futures.append(worker)

            # Wait for all workers to be created and store them
            self.workers = worker_futures
            LOGGER.info(f"Replica {replica_id} - All {num_gpus} workers created successfully")

            # Primary worker for coordinating inference
            self.primary_worker = self.workers[0]

            LOGGER.info(f"Replica {replica_id} - Initialized {num_gpus} multimodal model workers")

        except Exception as e:
            LOGGER.error(f"Error initializing distributed multimodal model deployment: {str(e)}")
            raise

    @app.post("/v1/chat/completions/")
    async def chat_completions(self, request: Dict[Any, Any]):
        """Handle multimodal chat completion requests.

        Supports two image content formats (normalized internally to format 1):
        1. {"type": "image", "image": "url_or_base64"}
        2. {"type": "image_url", "image_url": {"url": "url_or_base64"}} (OpenAI-style, converted to format 1)
        """
        try:
            # Extract parameters from the request dictionary
            messages = request.get("messages", [])

            prompts = messages
            if not isinstance(messages, list):
                prompts = [messages]

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

            # Serialize the dictionary to a JSON string representation
            json_prompts = [json.dumps(prompts)]

            # Extract images from messages
            images = []
            for message in prompts:
                for content in message["content"]:
                    if content["type"] == "image":
                        images.append(content["image"])

            # Prepare inference parameters
            inference_inputs = {
                "prompts": json_prompts,
                "images": images,
                "max_length": request.get("max_tokens", 50),
                "temperature": request.get("temperature", 1.0),
                "top_k": request.get("top_k", 0),
                "top_p": request.get("top_p", 0.0),
                "apply_chat_template": True,
                "random_seed": request.get("random_seed", None),
                "max_batch_size": request.get("max_batch_size", 4),
            }

            # Run model inference on all workers (distributed inference)
            # All workers participate in the forward pass for model parallelism
            inference_futures = [worker.infer.remote(inference_inputs) for worker in self.workers]
            # Get results from primary worker (rank 0)
            results = ray.get(inference_futures[0])

            # Extract generated texts from results
            generated_texts = results["sentences"]

            output = {
                "id": f"chatcmpl-{int(time.time())}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": request.get("model", self.model_id),
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": generated_texts[0] if generated_texts else "",
                        },
                        "index": 0,
                    }
                ],
            }

            LOGGER.info(f"Output: {output}")
            return output
        except Exception as e:
            LOGGER.error(f"Error during multimodal chat completion: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error during multimodal chat completion: {str(e)}")

    @app.post("/v1/completions/")
    async def completions(self, request: Dict[Any, Any]):
        """Handle multimodal completion requests."""
        try:
            # Handle both "prompt" and "prompts" fields
            prompts = request.get("prompts", [])
            if "prompt" in request and not prompts:
                prompts = [request["prompt"]]

            images = []
            if "image" in request and request["image"]:
                images = [request["image"]]
            elif "images" in request:
                images = request["images"]

            # Prepare inference inputs
            inference_inputs = {
                "prompts": prompts,
                "images": images,
                "max_length": request.get("max_tokens", 50),
                "temperature": request.get("temperature", 1.0),
                "top_k": request.get("top_k", 0),
                "top_p": request.get("top_p", 0.0),
                "apply_chat_template": False,
                "random_seed": request.get("random_seed", None),
                "max_batch_size": request.get("max_batch_size", 4),
            }

            # Run model inference on all workers (distributed inference)
            # All workers participate in the forward pass for model parallelism
            inference_futures = [worker.infer.remote(inference_inputs) for worker in self.workers]
            # Get results from primary worker (rank 0)
            results = ray.get(inference_futures[0])

            # Extract generated texts from results
            generated_texts = results.get("sentences", [])

            output = {
                "id": f"cmpl-{int(time.time())}",
                "object": "text_completion",
                "created": int(time.time()),
                "model": request.get("model", self.model_id),
                "choices": [
                    {
                        "text": generated_texts[0] if generated_texts else "",
                        "index": 0,
                    }
                ],
            }

            LOGGER.info(f"Output: {output}")
            return output
        except Exception as e:
            LOGGER.error(f"Error during multimodal completion: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Error during multimodal completion: {str(e)}")

    @app.get("/v1/models")
    async def list_models(self):
        """List available models."""
        return {
            "data": [{"id": self.model_id, "object": "model", "created": int(time.time())}],
            "object": "list",
        }

    @app.get("/v1/health")
    async def health_check(self):
        """Health check endpoint."""
        return {"status": "healthy"}
