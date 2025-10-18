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
import multiprocessing
import signal
import sys
from typing import Optional

from nemo_deploy.ray_utils import find_available_port
from nemo_export_deploy_common.import_utils import MISSING_RAY_MSG, UnavailableError

try:
    import ray
    from ray import serve

    from nemo_deploy.nlp.hf_deployable_ray import HFRayDeployable
    from nemo_deploy.nlp.megatronllm_deployable_ray import MegatronRayDeployable
    from nemo_export.tensorrt_llm_deployable_ray import TensorRTLLMRayDeployable

    HAVE_RAY = True
except (ImportError, ModuleNotFoundError):
    from unittest.mock import MagicMock

    ray = MagicMock()
    serve = MagicMock()
    Application = MagicMock()
    MegatronRayDeployable = MagicMock()
    HFRayDeployable = MagicMock()
    TensorRTLLMRayDeployable = MagicMock()
    HAVE_RAY = False

LOGGER = logging.getLogger("NeMo")


def get_available_cpus():
    """Get the total number of available CPUs in the system."""
    return multiprocessing.cpu_count()


class DeployRay:
    """A class for managing Ray deployment and serving of models.

    This class provides functionality to initialize Ray, start Ray Serve,
    deploy models, and manage the lifecycle of the Ray cluster. It supports
    both NeMo inframework models, Hugging Face models, and TensorRT-LLM models.

    Attributes:
        address (str): The address of the Ray cluster to connect to.
        num_cpus (int): Number of CPUs to allocate for the Ray cluster.
        num_gpus (int): Number of GPUs to allocate for the Ray cluster.
        include_dashboard (bool): Whether to include the Ray dashboard.
        ignore_reinit_error (bool): Whether to ignore errors when reinitializing Ray.
        runtime_env (dict): Runtime environment configuration for Ray.
        host (str): Host address to bind the server to.
        port (int): Port number for the server.

    Methods:
        deploy_inframework_model: Deploy a NeMo inframework model using Ray Serve.
        deploy_huggingface_model: Deploy a Hugging Face model using Ray Serve.
        deploy_tensorrt_llm_model: Deploy a TensorRT-LLM model using Ray Serve.
    """

    def __init__(
        self,
        address: str = "auto",
        num_cpus: Optional[int] = None,
        num_gpus: int = 1,
        include_dashboard: bool = False,
        ignore_reinit_error: bool = True,
        runtime_env: dict = None,
        host: str = "0.0.0.0",
        port: Optional[int] = None,
    ):
        """Initialize the DeployRay instance and set up the Ray cluster.

        Args:
            address (str, optional): Address of the Ray cluster. Defaults to "auto".
            num_cpus (int, optional): Number of CPUs to allocate. If None, uses all available. Defaults to None.
            num_gpus (int, optional): Number of GPUs to allocate. Defaults to 1.
            include_dashboard (bool, optional): Whether to include the dashboard. Defaults to False.
            ignore_reinit_error (bool, optional): Whether to ignore reinit errors. Defaults to True.
            runtime_env (dict, optional): Runtime environment configuration. Defaults to None.
            host (str, optional): Host address to bind the server to. Defaults to "0.0.0.0".
            port (int, optional): Port number for the server. If None, an available port will be found. Defaults to None.

        Raises:
            Exception: If Ray is not installed.
        """
        if not HAVE_RAY:
            raise UnavailableError(MISSING_RAY_MSG)

        # Initialize Ray with proper configuration
        if num_cpus is None:
            self.num_cpus = get_available_cpus()
        else:
            self.num_cpus = num_cpus
        self.num_gpus = num_gpus
        self.host = host
        self.port = port

        try:
            # Try to connect to existing Ray cluster
            ray.init(
                address=address,
                ignore_reinit_error=ignore_reinit_error,
                runtime_env=runtime_env,
            )
        except ConnectionError:
            # If no cluster exists, start a local one
            LOGGER.info("No existing Ray cluster found. Starting a local Ray cluster...")
            ray.init(
                num_cpus=self.num_cpus,
                num_gpus=self.num_gpus,
                include_dashboard=include_dashboard,
                ignore_reinit_error=ignore_reinit_error,
                runtime_env=runtime_env,
            )

    def _signal_handler(self, signum, frame):
        """Handle signal interrupts and gracefully shutdown the deployer."""
        LOGGER.info("Received interrupt signal. Shutting down gracefully...")
        self._stop()
        sys.exit(0)

    def _start(self):
        """Start Ray Serve with the configured host and port.

        Uses the host and port specified during DeployRay initialization.
        If port is None, an available port will be found automatically.
        """
        port = self.port
        if not port:
            port = find_available_port(8000, self.host)
        serve.start(
            http_options={
                "host": self.host,
                "port": port,
            }
        )

    def _stop(self):
        """Stop the Ray Serve deployment and shutdown the Ray cluster.

        This method attempts to gracefully shutdown both Ray Serve and the Ray cluster.
        If any errors occur during shutdown, they are logged as warnings.
        """
        try:
            # First try to gracefully shutdown Ray Serve
            LOGGER.info("Shutting down Ray Serve...")
            serve.shutdown()
        except Exception as e:
            LOGGER.warning(f"Error during serve.shutdown(): {str(e)}")
        try:
            # Then try to gracefully shutdown Ray
            LOGGER.info("Shutting down Ray...")
            ray.shutdown()
        except Exception as e:
            LOGGER.warning(f"Error during ray.shutdown(): {str(e)}")

    def deploy_inframework_model(
        self,
        nemo_checkpoint: str,
        num_gpus: int = 1,
        tensor_model_parallel_size: int = 1,
        pipeline_model_parallel_size: int = 1,
        expert_model_parallel_size: int = 1,
        context_parallel_size: int = 1,
        model_id: str = "nemo-model",
        num_cpus_per_replica: float = 8,
        num_replicas: int = 1,
        enable_cuda_graphs: bool = False,
        enable_flash_decode: bool = False,
        legacy_ckpt: bool = False,
        max_batch_size: int = 32,
        random_seed: Optional[int] = None,
        test_mode: bool = False,
        megatron_checkpoint_filepath: str = None,
        model_type: str = "gpt",
        model_format: str = "nemo",
        micro_batch_size: Optional[int] = None,
        **model_config_kwargs,
    ):
        """Deploy an inframework NeMo/Megatron model using Ray Serve.

        This method handles the complete deployment lifecycle including:
        - Starting Ray Serve
        - Creating and deploying the MegatronRayDeployable
        - Setting up signal handlers for graceful shutdown
        - Keeping the deployment running until interrupted

        Args:
            nemo_checkpoint (str): Path to the .nemo checkpoint file.
            num_gpus (int, optional): Number of GPUs per node. Defaults to 1.
            tensor_model_parallel_size (int, optional): Tensor model parallel size. Defaults to 1.
            pipeline_model_parallel_size (int, optional): Pipeline model parallel size. Defaults to 1.
            expert_model_parallel_size (int, optional): Expert model parallel size. Defaults to 1.
            context_parallel_size (int, optional): Context parallel size. Defaults to 1.
            model_id (str, optional): Model identifier for API responses. Defaults to "nemo-model".
            num_cpus_per_replica (float, optional): CPUs per model replica. Defaults to 8.
            num_replicas (int, optional): Number of replicas for deployment. Defaults to 1.
            enable_cuda_graphs (bool, optional): Enable CUDA graphs. Defaults to False.
            enable_flash_decode (bool, optional): Enable Flash Attention decode. Defaults to False.
            legacy_ckpt (bool, optional): Use legacy checkpoint format. Defaults to False.
            test_mode (bool, optional): Enable test mode. Defaults to False.
            megatron_checkpoint_filepath (str, optional): Path to the Megatron checkpoint file. Defaults to None.
            model_type (str, optional): Type of model to load. Defaults to "gpt".
            model_format (str, optional): Format of model to load. Defaults to "nemo".
            micro_batch_size (Optional[int], optional): Micro batch size for model execution. Defaults to None.

        Raises:
            SystemExit: If parallelism configuration is invalid.
            Exception: If deployment fails.
        """
        if not HAVE_RAY:
            raise UnavailableError(MISSING_RAY_MSG)

        # Calculate total GPUs and GPUs per replica
        gpus_per_replica = num_gpus // num_replicas

        # Validate parallelism configuration
        parallelism_per_replica = (
            tensor_model_parallel_size
            * pipeline_model_parallel_size
            * context_parallel_size
            * expert_model_parallel_size
        )

        if parallelism_per_replica != gpus_per_replica:
            LOGGER.error(
                f"Parallelism per replica ({parallelism_per_replica}) must equal GPUs per replica ({gpus_per_replica})"
            )
            LOGGER.error(f"Total GPUs: {num_gpus}, Num replicas: {num_replicas}, GPUs per replica: {gpus_per_replica}")
            LOGGER.error(
                f"Each replica needs: tensor_parallel({tensor_model_parallel_size}) * "
                f"pipeline_parallel({pipeline_model_parallel_size}) * "
                f"context_parallel({context_parallel_size}) * "
                f"expert_parallel({expert_model_parallel_size}) = {parallelism_per_replica} GPUs"
            )
            sys.exit(1)

        LOGGER.info(f"Configuration: {num_replicas} replicas, {gpus_per_replica} GPUs per replica")

        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        try:
            # Start Ray Serve
            self._start()

            # Create the Multi-Rank Megatron model deployment
            app = MegatronRayDeployable.options(
                num_replicas=num_replicas,
                ray_actor_options={"num_cpus": num_cpus_per_replica},
            ).bind(
                nemo_checkpoint_filepath=nemo_checkpoint,
                num_gpus=gpus_per_replica,
                tensor_model_parallel_size=tensor_model_parallel_size,
                pipeline_model_parallel_size=pipeline_model_parallel_size,
                expert_model_parallel_size=expert_model_parallel_size,
                context_parallel_size=context_parallel_size,
                model_id=model_id,
                enable_cuda_graphs=enable_cuda_graphs,
                enable_flash_decode=enable_flash_decode,
                legacy_ckpt=legacy_ckpt,
                max_batch_size=max_batch_size,
                random_seed=random_seed,
                megatron_checkpoint_filepath=megatron_checkpoint_filepath,
                model_type=model_type,
                model_format=model_format,
                micro_batch_size=micro_batch_size,
                **model_config_kwargs,
            )

            # Deploy the model
            serve.run(app, name=model_id)

            LOGGER.info(f"Megatron model deployed successfully at {self.host}:{self.port}")
            LOGGER.info("Press Ctrl+C to stop the deployment")

            # Keep the deployment running
            while not test_mode:
                signal.pause()

        except Exception as e:
            LOGGER.error(f"Error during deployment: {str(e)}")
            self._stop()
            sys.exit(1)

    def deploy_huggingface_model(
        self,
        hf_model_id_path: str,
        task: str = "text-generation",
        trust_remote_code: bool = True,
        device_map: Optional[str] = None,
        torch_dtype: Optional[str] = "auto",
        max_memory: Optional[str] = None,
        model_id: str = "hf-model",
        num_replicas: int = 1,
        num_cpus_per_replica: float = 8,
        num_gpus_per_replica: int = 1,
        max_ongoing_requests: int = 10,
        use_vllm_backend: bool = False,
        test_mode: bool = False,
    ):
        """Deploy a Hugging Face model using Ray Serve.

        This method handles the complete deployment lifecycle including:
        - Starting Ray Serve
        - Creating and deploying the HFRayDeployable
        - Setting up signal handlers for graceful shutdown
        - Keeping the deployment running until interrupted

        Args:
            hf_model_id_path (str): Path to the HuggingFace model or model identifier.
                Can be a local path or a model ID from HuggingFace Hub.
            task (str, optional): HuggingFace task type. Defaults to "text-generation".
            trust_remote_code (bool, optional): Whether to trust remote code when loading the model. Defaults to True.
            device_map (str, optional): Device mapping strategy for model placement. Defaults to "auto".
            max_memory (str, optional): Maximum memory allocation when using balanced device map. Defaults to None.
            model_id (str, optional): Model identifier for API responses. Defaults to "hf-model".
            num_replicas (int, optional): Number of replicas for deployment. Defaults to 1.
            num_cpus_per_replica (float, optional): CPUs per model replica. Defaults to 8.
            num_gpus_per_replica (int, optional): GPUs per model replica. Defaults to 1.
            max_ongoing_requests (int, optional): Maximum number of ongoing requests per replica. Defaults to 10.
            use_vllm_backend (bool, optional): Whether to use vLLM backend for deployment. If True, exports the HF ckpt
            to vLLM format and uses vLLM backend for inference. Defaults to False.
            test_mode (bool, optional): Enable test mode. Defaults to False.
        Raises:
            Exception: If Ray is not installed or deployment fails.
        """
        if not HAVE_RAY:
            raise UnavailableError(MISSING_RAY_MSG)

        LOGGER.info(
            f"Configuration: {num_replicas} replicas, {num_gpus_per_replica} GPUs per replica, {num_cpus_per_replica} CPUs per replica"
        )

        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        try:
            # Start Ray Serve
            self._start()

            # Create the HuggingFace model deployment
            app = HFRayDeployable.options(
                num_replicas=num_replicas,
                ray_actor_options={
                    "num_cpus": num_cpus_per_replica,
                    "num_gpus": num_gpus_per_replica,
                },
                max_ongoing_requests=max_ongoing_requests,
            ).bind(
                hf_model_id_path=hf_model_id_path,
                task=task,
                trust_remote_code=trust_remote_code,
                device_map=device_map,
                torch_dtype=torch_dtype,
                max_memory=max_memory,
                model_id=model_id,
                use_vllm_backend=use_vllm_backend,
            )

            # Deploy the model
            serve.run(app, name=model_id)

            LOGGER.info(
                f"HuggingFace model '{hf_model_id_path}' deployed successfully at {self.host}:{self.port or 'auto'}"
            )
            LOGGER.info("Press Ctrl+C to stop the deployment")

            # Keep the deployment running
            while not test_mode:
                signal.pause()

        except Exception as e:
            LOGGER.error(f"Error during HuggingFace model deployment: {str(e)}")
            self._stop()
            sys.exit(1)

    def deploy_tensorrt_llm_model(
        self,
        trt_llm_path: str,
        model_id: str = "tensorrt-llm-model",
        use_python_runtime: bool = True,
        multi_block_mode: bool = False,
        lora_ckpt_list: Optional[list] = None,
        enable_chunked_context: bool = False,
        max_tokens_in_paged_kv_cache: Optional[int] = None,
        num_replicas: int = 1,
        num_cpus_per_replica: float = 8,
        num_gpus_per_replica: int = 1,
        max_ongoing_requests: int = 10,
        test_mode: bool = False,
    ):
        """Deploy a TensorRT-LLM model using Ray Serve.

        This method handles the complete deployment lifecycle including:
        - Starting Ray Serve
        - Creating and deploying the TensorRTLLMRayDeployable
        - Setting up signal handlers for graceful shutdown
        - Keeping the deployment running until interrupted

        Note: This method assumes the model is already converted to TensorRT-LLM format.
        The conversion should be done before calling this API.

        Args:
            trt_llm_path (str): Path to the TensorRT-LLM model directory with pre-built engines.
            model_id (str, optional): Model identifier for API responses. Defaults to "tensorrt-llm-model".
            use_python_runtime (bool, optional): Whether to use Python runtime (vs C++ runtime). Defaults to True.
            multi_block_mode (bool, optional): Whether to enable multi-block mode. Defaults to False.
            lora_ckpt_list (list, optional): List of LoRA checkpoint paths. Defaults to None.
            enable_chunked_context (bool, optional): Whether to enable chunked context (C++ runtime only). Defaults to False.
            max_tokens_in_paged_kv_cache (int, optional): Maximum tokens in paged KV cache (C++ runtime only). Defaults to None.
            num_replicas (int, optional): Number of replicas for deployment. Defaults to 1.
            num_cpus_per_replica (float, optional): CPUs per model replica. Defaults to 8.
            num_gpus_per_replica (int, optional): GPUs per model replica. Defaults to 1.
            max_ongoing_requests (int, optional): Maximum number of ongoing requests per replica. Defaults to 10.
            test_mode (bool, optional): Enable test mode. Defaults to False.
        Raises:
            Exception: If Ray is not installed or deployment fails.
            ValueError: If C++ runtime specific options are used with Python runtime.
        """
        if not HAVE_RAY:
            raise UnavailableError(MISSING_RAY_MSG)

        # Validate C++ runtime specific options
        if use_python_runtime and (enable_chunked_context or max_tokens_in_paged_kv_cache):
            raise ValueError(
                "enable_chunked_context and max_tokens_in_paged_kv_cache options "
                "work only with the TensorRT-LLM C++ runtime. Set use_python_runtime=False."
            )

        LOGGER.info(
            f"Configuration: {num_replicas} replicas, {num_gpus_per_replica} GPUs per replica, {num_cpus_per_replica} CPUs per replica"
        )

        # Set up signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

        try:
            # Start Ray Serve
            self._start()

            # Prepare deployment parameters
            deployment_kwargs = {
                "trt_llm_path": trt_llm_path,
                "model_id": model_id,
                "use_python_runtime": use_python_runtime,
                "multi_block_mode": multi_block_mode,
                "lora_ckpt_list": lora_ckpt_list,
            }

            # Add C++ runtime specific options if using C++ runtime
            if not use_python_runtime:
                deployment_kwargs["enable_chunked_context"] = enable_chunked_context
                deployment_kwargs["max_tokens_in_paged_kv_cache"] = max_tokens_in_paged_kv_cache

            # Create the TensorRT-LLM model deployment
            app = TensorRTLLMRayDeployable.options(
                num_replicas=num_replicas,
                ray_actor_options={
                    "num_cpus": num_cpus_per_replica,
                    "num_gpus": num_gpus_per_replica,
                },
                max_ongoing_requests=max_ongoing_requests,
            ).bind(**deployment_kwargs)

            # Deploy the model
            serve.run(app, name=model_id)

            LOGGER.info(f"TensorRT-LLM model deployed successfully at {self.host}:{self.port or 'auto'}")
            LOGGER.info("Press Ctrl+C to stop the deployment")

            # Keep the deployment running
            while not test_mode:
                signal.pause()

        except Exception as e:
            LOGGER.error(f"Error during TensorRT-LLM model deployment: {str(e)}")
            self._stop()
            sys.exit(1)
