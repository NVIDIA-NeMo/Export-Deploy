# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
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
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Literal

import numpy as np

LOGGER = logging.getLogger("NeMo")

from nemo_deploy import ITritonDeployable
from nemo_deploy.utils import cast_output, str_ndarray2list
from nemo_export.utils import is_nemo2_checkpoint
from nemo_export_deploy_common.import_utils import (
    MISSING_NEMO_MSG,
    MISSING_TRITON_MSG,
    MISSING_VLLM_MSG,
    UnavailableError,
)

try:
    from nemo.collections.llm.api import export_ckpt

    HAVE_NeMo2 = True
except (ImportError, ModuleNotFoundError):
    HAVE_NeMo2 = False

try:
    from pytriton.decorators import batch, first_value
    from pytriton.model_config import Tensor

    HAVE_PYTRITON = True
except (ImportError, ModuleNotFoundError):
    from unittest.mock import MagicMock

    batch = MagicMock()
    first_value = MagicMock()
    HAVE_PYTRITON = False

try:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    HAVE_VLLM = True
except (ImportError, ModuleNotFoundError):
    HAVE_VLLM = False


class vLLMExporter(ITritonDeployable):
    """
    vLLMExporter enables deployment of Hugging Face or NeMo2 models using vLLM and Triton.

    This class wraps vLLM APIs to load a model and make it deployable with Triton Inference Server.
    It supports exporting NeMo2 checkpoints to Hugging Face format if needed, and then loads the model
    with vLLM for fast inference.

    Example:
        from nemo_export import vLLMExporter
        from nemo_deploy import DeployPyTriton

        exporter = vLLMExporter()
        exporter.export(model_path_id="/path/to/model/")

        server = DeployPyTriton(
            model=exporter,
            triton_model_name='model'
        )

        server.deploy()
        server.serve()
    """

    def __init__(self):
        """
        Initializes the vLLMExporter instance.

        This constructor sets up the exporter by initializing model and LoRA model attributes.
        It also checks for the availability of required dependencies (vLLM, PyTriton, NeMo2)
        and raises an UnavailableError if any are missing.
        """
        self.model = None
        self.lora_models = None
        if not HAVE_VLLM:
            raise UnavailableError(MISSING_VLLM_MSG)
        if not HAVE_PYTRITON:
            raise UnavailableError(MISSING_TRITON_MSG)
        if not HAVE_NeMo2:
            raise UnavailableError(MISSING_NEMO_MSG)

    def export(
        self,
        model_path_id: str,
        tokenizer: str = None,
        trust_remote_code: bool = False,
        enable_lora: bool = False,
        tensor_parallel_size: int = 1,
        dtype: str = "auto",
        quantization: str = None,
        seed: int = 0,
        gpu_memory_utilization: float = 0.9,
        swap_space: float = 4,
        cpu_offload_gb: float = 0,
        enforce_eager: bool = False,
        max_seq_len_to_capture: int = 8192,
        task: Literal["auto", "generate", "embedding"] = "auto",
    ):
        """
        Exports a Hugging Face or NeMo2 checkpoint to vLLM and initializes the engine.

        Args:
            model_path_id (str): Model name or path to the checkpoint directory. Can be a Hugging Face or NeMo2 checkpoint.
            tokenizer (str, optional): Path to the tokenizer or tokenizer name. Defaults to None.
            trust_remote_code (bool, optional): Whether to trust remote code from Hugging Face Hub. Defaults to False.
            enable_lora (bool, optional): Whether to enable LoRA support. Defaults to False.
            tensor_parallel_size (int, optional): Number of tensor parallel partitions. Defaults to 1.
            dtype (str, optional): Data type for model weights. Defaults to "auto".
            quantization (str, optional): Quantization type. Defaults to None.
            seed (int, optional): Random seed. Defaults to 0.
            gpu_memory_utilization (float, optional): Fraction of GPU memory to use. Defaults to 0.9.
            swap_space (float, optional): Amount of swap space (in GB) to use. Defaults to 4.
            cpu_offload_gb (float, optional): Amount of CPU offload memory (in GB). Defaults to 0.
            enforce_eager (bool, optional): Whether to enforce eager execution. Defaults to False.
            max_seq_len_to_capture (int, optional): Maximum sequence length to capture. Defaults to 8192.
            task (Literal["auto", "generate", "embedding"], optional): Task type for vLLM. Defaults to "auto".

        Raises:
            Exception: If NeMo checkpoint conversion to Hugging Face format fails.
        """
        if Path(model_path_id).exists() and is_nemo2_checkpoint(model_path_id):
            with tempfile.TemporaryDirectory() as tmp_hf_export_dir:
                try:
                    export_ckpt(
                        path=model_path_id,
                        target="hf",
                        output_path=tmp_hf_export_dir,
                        overwrite=True,
                    )
                except Exception as e:
                    raise Exception(
                        f"NeMo checkpoint is not supported. Error occured during Hugging Face conversion. Error message: {e}"
                    )

                if not any(Path(tmp_hf_export_dir).iterdir()):
                    raise Exception("NeMo checkpoint is not supported. Error occured during Hugging Face conversion.")

                self.model = LLM(
                    model=tmp_hf_export_dir,
                    tokenizer=tokenizer,
                    trust_remote_code=trust_remote_code,
                    enable_lora=enable_lora,
                    tensor_parallel_size=tensor_parallel_size,
                    dtype=dtype,
                    quantization=quantization,
                    seed=seed,
                    gpu_memory_utilization=gpu_memory_utilization,
                    swap_space=swap_space,
                    cpu_offload_gb=cpu_offload_gb,
                    enforce_eager=enforce_eager,
                    max_seq_len_to_capture=max_seq_len_to_capture,
                    task=task,
                )
        else:
            self.model = LLM(
                model=model_path_id,
                tokenizer=tokenizer,
                trust_remote_code=trust_remote_code,
                enable_lora=enable_lora,
                tensor_parallel_size=tensor_parallel_size,
                dtype=dtype,
                quantization=quantization,
                seed=seed,
                gpu_memory_utilization=gpu_memory_utilization,
                swap_space=swap_space,
                cpu_offload_gb=cpu_offload_gb,
                enforce_eager=enforce_eager,
                max_seq_len_to_capture=max_seq_len_to_capture,
                task=task,
            )

    def add_lora_models(self, lora_model_name, lora_model):
        """
        Add a LoRA (Low-Rank Adaptation) model to the exporter.

        Args:
            lora_model_name (str): The name or identifier for the LoRA model.
            lora_model: The LoRA model object to be added.
        """
        if self.lora_models is None:
            self.lora_models = {}
        self.lora_models[lora_model_name] = lora_model

    @property
    def get_triton_input(self):
        """
        Returns the expected Triton model input signature for vLLMExporter.

        Returns:
            tuple: A tuple of Tensor objects describing the input fields:
                - prompts (np.bytes_): Input prompt strings.
                - max_tokens (np.int_, optional): Maximum number of tokens to generate.
                - min_tokens (np.int_, optional): Minimum number of tokens to generate.
                - top_k (np.int_, optional): Top-K sampling parameter.
                - top_p (np.single, optional): Top-P (nucleus) sampling parameter.
                - temperature (np.single, optional): Sampling temperature.
                - seed (np.int_, optional): Random seed for generation.
                - n_log_probs (np.int_, optional): Number of log probabilities to return for generated tokens.
                - n_prompt_log_probs (np.int_, optional): Number of log probabilities to return for prompt tokens.
        """
        return (
            Tensor(name="prompts", shape=(-1,), dtype=bytes),
            Tensor(name="max_tokens", shape=(-1,), dtype=np.int_, optional=True),
            Tensor(name="min_tokens", shape=(-1,), dtype=np.int_, optional=True),
            Tensor(name="top_k", shape=(-1,), dtype=np.int_, optional=True),
            Tensor(name="top_p", shape=(-1,), dtype=np.single, optional=True),
            Tensor(name="temperature", shape=(-1,), dtype=np.single, optional=True),
            Tensor(name="seed", shape=(-1,), dtype=np.int_, optional=True),
            Tensor(name="n_log_probs", shape=(-1,), dtype=np.int_, optional=True),
            Tensor(name="n_prompt_log_probs", shape=(-1,), dtype=np.int_, optional=True),
        )

    @property
    def get_triton_output(self):
        """
        Returns the expected Triton model output signature for vLLMExporter.

        Returns:
            tuple: A tuple of Tensor objects describing the output fields:
                - sentences (np.bytes_): Generated text.
                - log_probs (np.bytes_): Log probabilities for generated tokens.
                - prompt_log_probs (np.bytes_): Log probabilities for prompt tokens.
        """
        return (
            Tensor(name="sentences", shape=(-1,), dtype=bytes),
            Tensor(name="log_probs", shape=(-1,), dtype=bytes),
            Tensor(name="prompt_log_probs", shape=(-1,), dtype=bytes),
        )

    @batch
    @first_value(
        "max_tokens", "min_tokens", "n_log_probs", "n_prompt_log_probs", "seed", "top_k", "top_p", "temperature"
    )
    def triton_infer_fn(self, **inputs: np.ndarray):  # pragma: no cover
        """
        Triton inference function for vLLMExporter.

        This function processes input prompts and generates text using vLLM.
        It supports optional parameters for maximum tokens, minimum tokens,
        log probabilities, and random seed.
        """
        try:
            # Convert triton-specific inputs
            prompts = str_ndarray2list(inputs.pop("prompts"))

            # Convert numpy arrays to Python types for triton inputs
            processed_inputs = {}
            for key, value in inputs.items():
                processed_inputs[key] = value

            output_infer = self._infer_fn(prompts, processed_inputs)

            # Format output for triton
            output_infer["sentences"] = cast_output(output_infer["sentences"], np.bytes_)
            if "log_probs" in output_infer.keys():
                output_infer["log_probs"] = cast_output(output_infer["log_probs"], np.bytes_)
            if "prompt_log_probs" in output_infer.keys():
                output_infer["prompt_log_probs"] = cast_output(output_infer["prompt_log_probs"], np.bytes_)

        except Exception as error:
            err_msg = "An error occurred: {0}".format(str(error))
            output_infer = {"sentences": cast_output([err_msg], np.bytes_)}

        return output_infer

    def _infer_fn(self, prompts, inputs):
        """Shared helper function to prepare inference inputs and execute forward pass.

        Args:
            prompts: List of input prompts
            inputs: Dictionary of input parameters

        Returns:
            output_dict: Dictionary containing generated text and optional log probabilities
        """
        infer_input = {"input_texts": prompts}

        # Process common parameters
        if "max_tokens" in inputs:
            infer_input["max_tokens"] = int(inputs["max_tokens"])
        if "min_tokens" in inputs:
            infer_input["min_tokens"] = int(inputs["min_tokens"])
        if "n_log_probs" in inputs:
            infer_input["n_log_probs"] = int(inputs["n_log_probs"])
        if "n_prompt_log_probs" in inputs:
            infer_input["n_prompt_log_probs"] = int(inputs["n_prompt_log_probs"])
        if "seed" in inputs:
            infer_input["seed"] = int(inputs["seed"])
        if "top_k" in inputs:
            infer_input["top_k"] = int(inputs["top_k"])
        if "top_p" in inputs:
            infer_input["top_p"] = float(inputs["top_p"])
        if "temperature" in inputs:
            infer_input["temperature"] = float(inputs["temperature"])
        if "lora_model_name" in inputs:
            infer_input["lora_model_name"] = inputs["lora_model_name"]

        output = self.forward(**infer_input)

        if isinstance(output, dict):
            return output
        else:
            err_msg = "An error occurred: the output format is expected to be a dict."
            return {"sentences": [err_msg]}

    def ray_infer_fn(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Ray inference function that processes input dictionary and returns output without byte casting.

        Args:
            inputs (Dict[str, Any]): Input dictionary containing:
                - prompts: List of input prompts
                - max_tokens: Maximum number of tokens to generate (optional)
                - min_tokens: Minimum number of tokens to generate (optional)
                - top_k: Top-k sampling parameter (optional)
                - top_p: Top-p sampling parameter (optional)
                - temperature: Sampling temperature (optional)
                - seed: Random seed for generation (optional)
                - n_log_probs: Number of log probabilities to return for generated tokens (optional)
                - n_prompt_log_probs: Number of log probabilities to return for prompt tokens (optional)
                - lora_model_name: Name of the LoRA model to use for generation (optional)

        Returns:
            Dict[str, Any]: Output dictionary containing:
                - sentences: List of generated text outputs
                - log_probs: Log probabilities for generated tokens (if requested)
                - prompt_log_probs: Log probabilities for prompt tokens (if requested)
        """
        output_dict = {}

        # Extract prompts - handle both list and single string cases
        prompts = inputs.get("prompts", [])
        if isinstance(prompts, str):
            prompts = [prompts]

        try:
            output_dict = self._infer_fn(prompts, inputs)

        except Exception as error:
            err_msg = f"An error occurred: {str(error)}"
            LOGGER.error(err_msg)
            output_dict["sentences"] = [err_msg] * len(prompts)
            output_dict["error"] = err_msg

        return output_dict

    def _dict_to_str(self, messages):
        """Serializes dict to str."""
        import json

        return json.dumps(messages)

    def forward(
        self,
        input_texts: List[str],
        max_tokens: int = 16,
        min_tokens: int = 0,
        top_k: int = 1,
        top_p: float = 0.1,
        temperature: float = 1.0,
        n_log_probs: int = None,
        n_prompt_log_probs: int = None,
        seed: int = None,
        lora_model_name: str = None,
    ):
        """
        Generate text completions for a list of input prompts using the vLLM model.

        Args:
            input_texts (List[str]): List of input prompt strings.
            max_tokens (int, optional): Maximum number of tokens to generate for each prompt. Defaults to 16.
            min_tokens (int, optional): Minimum number of tokens to generate for each prompt. Defaults to 0.
            top_k (int, optional): The number of highest probability vocabulary tokens to keep for top-k-filtering. Defaults to 1.
            top_p (float, optional): If set to float < 1, only the most probable tokens with probabilities that add up to top_p or higher are kept for generation. Defaults to 0.1.
            temperature (float, optional): Sampling temperature. Defaults to 1.0.
            n_log_probs (int, optional): Number of log probabilities to return for generated tokens. Defaults to None.
            n_prompt_log_probs (int, optional): Number of log probabilities to return for prompt tokens. Defaults to None.
            seed (int, optional): Random seed for generation. Defaults to None.
            lora_model_name (str, optional): Name of the LoRA model to use for generation. Defaults to None.

        Returns:
            dict or list: Generated text completions and optionally log probabilities, depending on the input arguments.
        """
        assert self.model is not None, "Model is not initialized."

        lora_request = None
        if lora_model_name is not None:
            if self.lora_models is None:
                raise Exception("No lora models are available.")
            assert lora_model_name in self.lora_models.keys(), "Lora model was not added before"
            lora_request = LoRARequest(lora_model_name, 1, self.lora_models[lora_model_name])

        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            min_tokens=min_tokens,
            logprobs=n_log_probs,
            prompt_logprobs=n_prompt_log_probs,
            seed=seed,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        request_output = self.model.generate(input_texts, sampling_params, lora_request=lora_request)

        output = []
        top_logprobs = []
        top_prompt_logprobs = []
        for o in request_output:
            output.append(o.outputs[0].text)

            if n_log_probs is not None and n_log_probs > 0:
                lpbs = []
                for lp in o.outputs[0].logprobs:
                    lg_items = {}
                    values = lp.values()
                    values = sorted(values, key=lambda v: v.rank)
                    for v in values:
                        lg_items[str(v.decoded_token)] = v.logprob
                    lpbs.append(self._dict_to_str(lg_items))
                top_logprobs.append(lpbs)

            if n_prompt_log_probs is not None and n_prompt_log_probs > 0:
                lpbs = []
                for lp in o.prompt_logprobs:
                    if lp is not None and isinstance(lp, dict):
                        lg_items = {}
                        values = lp.values()
                        values = sorted(values, key=lambda v: v.rank)
                        for v in values:
                            lg_items[str(v.decoded_token)] = v.logprob

                        lpbs.append(self._dict_to_str(lg_items))

                top_prompt_logprobs.append(lpbs)

        # TODO athitten: extract only the generated text if output contains input prompt + generated text.
        output = {"sentences": output}
        if len(top_logprobs) > 0:
            max_len = max(len(arr) for arr in top_logprobs)
            # Pad each array to the maximum length. Pads 0.
            top_logprobs = np.array([np.pad(arr, (0, max_len - len(arr)), constant_values=0) for arr in top_logprobs])
            output["log_probs"] = top_logprobs
        if len(top_prompt_logprobs) > 0:
            max_len = max(len(arr) for arr in top_prompt_logprobs)
            # Pad each array to the maximum length. Pads 0.
            top_prompt_logprobs = np.array(
                [np.pad(arr, (0, max_len - len(arr)), constant_values=0) for arr in top_prompt_logprobs]
            )
            output["prompt_log_probs"] = top_prompt_logprobs

        return output
