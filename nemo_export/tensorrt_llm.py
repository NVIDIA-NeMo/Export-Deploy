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

import importlib.util
import logging
from typing import Any, Dict, List, Optional

import numpy as np

from nemo_deploy import ITritonDeployable
from nemo_export_deploy_common.import_utils import (
    null_decorator,
)

try:
    from pytriton.decorators import batch, first_value
    from pytriton.model_config import Tensor

    HAVE_PYTRITON = True
except (ImportError, ModuleNotFoundError):
    from unittest.mock import MagicMock

    batch = null_decorator
    first_value = null_decorator
    Tensor = MagicMock()
    HAVE_PYTRITON = False

HAVE_TENSORRT_LLM = importlib.util.find_spec("tensorrt_llm") is not None

LOGGER = logging.getLogger("NeMo")


# pylint: disable=line-too-long
class TensorRTLLM(ITritonDeployable):
    """Exports NeMo checkpoints to TensorRT-LLM and run fast inference.

    This class provides functionality to export NeMo models to TensorRT-LLM
    format and run inference using the exported models. It supports various model architectures
    and provides options for model parallelism, quantization, and inference parameters.

    Note: For HuggingFace model export, use the TensorRTLLMHF class instead.

    Two export methods are available:
    - export(): Standard NeMo export pipeline
    - export_with_hf_fallback(): Tries standard export first, falls back to HF conversion if it fails

    Example:
        from nemo_export.tensorrt_llm import TensorRTLLM

        trt_llm_exporter = TensorRTLLM(model_dir="/path/for/model/files")
        trt_llm_exporter.export(
            nemo_checkpoint_path="/path/for/nemo/checkpoint",
            model_type="llama",
            tensor_parallelism_size=1,
        )

        output = trt_llm_exporter.forward(["Hi, how are you?", "I am good, thanks, how about you?"])
        print("output: ", output)

    Example with fallback:
        trt_llm_exporter = TensorRTLLM(model_dir="/path/for/model/files")
        trt_llm_exporter.export_with_hf_fallback(
            nemo_checkpoint_path="/path/for/nemo/checkpoint",
            model_type="llama",
            tensor_parallelism_size=1,
        )
    """

    def __init__(
        self,
        model_dir: str,
        lora_ckpt_list: List[str] = None,
        load_model: bool = True,
        use_python_runtime: bool = True,
        enable_chunked_context: bool = None,
        max_tokens_in_paged_kv_cache: int = None,
        multi_block_mode: bool = False,
    ):
        """Initialize TensorRTLLM exporter.

        Args:
            model_dir (str): Path for storing the TensorRT-LLM model files.
            lora_ckpt_list (List[str], optional): List of LoRA checkpoint paths. Defaults to None.
            load_model (bool, optional): Load TensorRT-LLM model if engine files exist. Defaults to True.
            use_python_runtime (bool, optional): Whether to use python or c++ runtime. Defaults to True.
            enable_chunked_context (bool, optional): Enable chunked context processing. Defaults to None.
            max_tokens_in_paged_kv_cache (int, optional): Max tokens in paged KV cache. Defaults to None.
            multi_block_mode (bool, optional): Enable faster decoding in multihead attention. Defaults to False.
        """
        raise NotImplementedError("To be implemented")

    def _export_nemo_checkpoint(
        self,
        nemo_checkpoint_path: str,
        model_type: Optional[str] = None,
        delete_existing_files: bool = True,
        tensor_parallelism_size: int = 1,
        pipeline_parallelism_size: int = 1,
        max_input_len: int = 256,
        max_output_len: Optional[int] = None,
        max_batch_size: int = 8,
        use_parallel_embedding: bool = False,
        paged_kv_cache: bool = True,
        remove_input_padding: bool = True,
        use_paged_context_fmha: bool = True,
        dtype: Optional[str] = None,
        load_model: bool = True,
        use_lora_plugin: str = None,
        lora_target_modules: List[str] = None,
        max_lora_rank: int = 64,
        max_num_tokens: Optional[int] = None,
        opt_num_tokens: Optional[int] = None,
        max_seq_len: Optional[int] = 512,
        multiple_profiles: bool = False,
        gpt_attention_plugin: str = "auto",
        gemm_plugin: str = "auto",
        reduce_fusion: bool = True,
        fp8_quantized: Optional[bool] = None,
        fp8_kvcache: Optional[bool] = None,
        build_rank: Optional[int] = 0,
    ):
        """Export nemo checkpoints to TensorRT-LLM format.

        This method exports a NeMo checkpoint to TensorRT-LLM format with various configuration
        options for model parallelism, quantization, and inference parameters.

        Args:
            nemo_checkpoint_path (str): Path to the NeMo checkpoint.
            model_type (Optional[str], optional): Type of the model. Defaults to None.
            delete_existing_files (bool, optional): Delete existing files in model_dir. Defaults to True.
            tensor_parallelism_size (int, optional): Size of tensor parallelism. Defaults to 1.
            pipeline_parallelism_size (int, optional): Size of pipeline parallelism. Defaults to 1.
            max_input_len (int, optional): Maximum input sequence length. Defaults to 256.
            max_output_len (Optional[int], optional): Maximum output sequence length. Defaults to None.
            max_batch_size (int, optional): Maximum batch size. Defaults to 8.
            use_parallel_embedding (bool, optional): Use parallel embedding. Defaults to False.
            paged_kv_cache (bool, optional): Use paged KV cache. Defaults to True.
            remove_input_padding (bool, optional): Remove input padding. Defaults to True.
            use_paged_context_fmha (bool, optional): Use paged context FMHA. Defaults to True.
            dtype (Optional[str], optional): Data type for model weights. Defaults to None.
            load_model (bool, optional): Load model after export. Defaults to True.
            use_lora_plugin (str, optional): Use LoRA plugin. Defaults to None.
            lora_target_modules (List[str], optional): Target modules for LoRA. Defaults to None.
            max_lora_rank (int, optional): Maximum LoRA rank. Defaults to 64.
            max_num_tokens (Optional[int], optional): Maximum number of tokens. Defaults to None.
            opt_num_tokens (Optional[int], optional): Optimal number of tokens. Defaults to None.
            max_seq_len (Optional[int], optional): Maximum sequence length. Defaults to 512.
            multiple_profiles (bool, optional): Use multiple profiles. Defaults to False.
            gpt_attention_plugin (str, optional): GPT attention plugin type. Defaults to "auto".
            gemm_plugin (str, optional): GEMM plugin type. Defaults to "auto".
            reduce_fusion (bool, optional): Enable reduce fusion. Defaults to True.
            fp8_quantized (Optional[bool], optional): Enable FP8 quantization. Defaults to None.
            fp8_kvcache (Optional[bool], optional): Enable FP8 KV cache. Defaults to None.
            build_rank (Optional[int], optional): Rank to build on. Defaults to 0.

        Raises:
            ValueError: If model_type is not supported or dtype cannot be determined.
            Exception: If files cannot be deleted or other export errors occur.
        """
        raise NotImplementedError("To be implemented")

    def export_with_hf(
        self,
        nemo_checkpoint_path: str,
        model_type: Optional[str] = None,
        delete_existing_files: bool = True,
        tensor_parallelism_size: int = 1,
        max_input_len: int = 256,
        max_output_len: Optional[int] = None,
        max_batch_size: int = 8,
        paged_kv_cache: bool = True,
        remove_input_padding: bool = True,
        use_paged_context_fmha: bool = True,
        dtype: Optional[str] = None,
        max_num_tokens: Optional[int] = None,
        opt_num_tokens: Optional[int] = None,
        max_seq_len: Optional[int] = 512,
        multiple_profiles: bool = False,
        gemm_plugin: str = "auto",
        reduce_fusion: bool = False,
    ):
        """Internal method to export via HuggingFace conversion fallback.

        This method converts a NeMo2 checkpoint to HuggingFace format, then exports
        to TensorRT-LLM using the HF export pipeline.

        Args:
            nemo_checkpoint_path (str): Path to the NeMo checkpoint.
            model_type (Optional[str], optional): Type of the model. Defaults to None.
            delete_existing_files (bool, optional): Delete existing files in model_dir. Defaults to True.
            tensor_parallelism_size (int, optional): Size of tensor parallelism. Defaults to 1.
            max_input_len (int, optional): Maximum input sequence length. Defaults to 256.
            max_output_len (Optional[int], optional): Maximum output sequence length. Defaults to None.
            max_batch_size (int, optional): Maximum batch size. Defaults to 8.
            paged_kv_cache (bool, optional): Use paged KV cache. Defaults to True.
            remove_input_padding (bool, optional): Remove input padding. Defaults to True.
            use_paged_context_fmha (bool, optional): Use paged context FMHA. Defaults to True.
            dtype (Optional[str], optional): Data type for model weights. Defaults to None.
            max_num_tokens (Optional[int], optional): Maximum number of tokens. Defaults to None.
            opt_num_tokens (Optional[int], optional): Optimal number of tokens. Defaults to None.
            max_seq_len (Optional[int], optional): Maximum sequence length. Defaults to 512.
            multiple_profiles (bool, optional): Use multiple profiles. Defaults to False.
            gemm_plugin (str, optional): GEMM plugin type. Defaults to "auto".
            reduce_fusion (bool, optional): Enable reduce fusion. Defaults to False.

        Raises:
            Exception: If HF conversion or export fails.
        """
        raise NotImplementedError("To be implemented")

    def export(
        self,
        nemo_checkpoint_path: str,
        model_type: Optional[str] = None,
        delete_existing_files: bool = True,
        tensor_parallelism_size: int = 1,
        pipeline_parallelism_size: int = 1,
        max_input_len: int = 256,
        max_output_len: Optional[int] = None,
        max_batch_size: int = 8,
        use_parallel_embedding: bool = False,
        paged_kv_cache: bool = True,
        remove_input_padding: bool = True,
        use_paged_context_fmha: bool = True,
        dtype: Optional[str] = None,
        load_model: bool = True,
        use_lora_plugin: str = None,
        lora_target_modules: List[str] = None,
        max_lora_rank: int = 64,
        max_num_tokens: Optional[int] = None,
        opt_num_tokens: Optional[int] = None,
        max_seq_len: Optional[int] = 512,
        multiple_profiles: bool = False,
        gpt_attention_plugin: str = "auto",
        gemm_plugin: str = "auto",
        reduce_fusion: bool = True,
        fp8_quantized: Optional[bool] = None,
        fp8_kvcache: Optional[bool] = None,
        build_rank: Optional[int] = 0,
    ):
        """Export nemo checkpoints to TensorRT-LLM with fallback to HF export.

        This method first attempts to export using the standard NeMo export pipeline.
        If that fails, it will convert the NeMo checkpoint to HuggingFace format first,
        then export to TensorRT-LLM using the HF export pipeline.

        Args:
            nemo_checkpoint_path (str): Path to the NeMo checkpoint.
            model_type (Optional[str], optional): Type of the model. Defaults to None.
            delete_existing_files (bool, optional): Delete existing files in model_dir. Defaults to True.
            tensor_parallelism_size (int, optional): Size of tensor parallelism. Defaults to 1.
            pipeline_parallelism_size (int, optional): Size of pipeline parallelism. Defaults to 1.
            max_input_len (int, optional): Maximum input sequence length. Defaults to 256.
            max_output_len (Optional[int], optional): Maximum output sequence length. Defaults to None.
            max_batch_size (int, optional): Maximum batch size. Defaults to 8.
            use_parallel_embedding (bool, optional): Use parallel embedding. Defaults to False.
            paged_kv_cache (bool, optional): Use paged KV cache. Defaults to True.
            remove_input_padding (bool, optional): Remove input padding. Defaults to True.
            use_paged_context_fmha (bool, optional): Use paged context FMHA. Defaults to True.
            dtype (Optional[str], optional): Data type for model weights. Defaults to None.
            load_model (bool, optional): Load model after export. Defaults to True.
            use_lora_plugin (str, optional): Use LoRA plugin. Defaults to None.
            lora_target_modules (List[str], optional): Target modules for LoRA. Defaults to None.
            max_lora_rank (int, optional): Maximum LoRA rank. Defaults to 64.
            max_num_tokens (Optional[int], optional): Maximum number of tokens. Defaults to None.
            opt_num_tokens (Optional[int], optional): Optimal number of tokens. Defaults to None.
            max_seq_len (Optional[int], optional): Maximum sequence length. Defaults to 512.
            multiple_profiles (bool, optional): Use multiple profiles. Defaults to False.
            gpt_attention_plugin (str, optional): GPT attention plugin type. Defaults to "auto".
            gemm_plugin (str, optional): GEMM plugin type. Defaults to "auto".
            reduce_fusion (bool, optional): Enable reduce fusion. Defaults to True.
            fp8_quantized (Optional[bool], optional): Enable FP8 quantization. Defaults to None.
            fp8_kvcache (Optional[bool], optional): Enable FP8 KV cache. Defaults to None.
            build_rank (Optional[int], optional): Rank to build on. Defaults to 0.

        Raises:
            ValueError: If model_type is not supported or dtype cannot be determined.
            Exception: If both NeMo and HF export methods fail.
        """
        raise NotImplementedError("To be implemented")

    def _export_to_nim_format(self, model_config: Dict[str, Any], model_type: str):
        """Exports the model configuration to a specific format required by NIM.

        This method performs the following steps:

        1. Copies the generation_config.json (if present) from the nemo_context directory to the root model directory.
        2. Creates a dummy Hugging Face configuration file based on the provided model configuration and type.

        Args:
            model_config (dict): A dictionary containing the model configuration parameters.
            model_type (str): The type of the model (e.g., "llama").
        """
        raise NotImplementedError("To be implemented")

    def get_transformer_config(self, nemo_model_config):
        """Given nemo model config get transformer config."""
        raise NotImplementedError("To be implemented")

    def forward(
        self,
        input_texts: List[str],
        max_output_len: int = 64,
        top_k: int = 1,
        top_p: float = 0.0,
        temperature: float = 1.0,
        stop_words_list: List[str] = None,
        bad_words_list: List[str] = None,
        no_repeat_ngram_size: int = None,
        lora_uids: List[str] = None,
        output_log_probs: bool = False,
        output_context_logits: bool = False,
        output_generation_logits: bool = False,
        **sampling_kwargs,
    ):
        """Exports nemo checkpoints to TensorRT-LLM.

        Args:
            input_texts (List(str)): list of sentences.
            max_output_len (int): max generated tokens.
            top_k (int): limits us to a certain number (K) of the top tokens to consider.
            top_p (float): limits us to the top tokens within a certain probability mass (p).
            temperature (float): A parameter of the softmax function, which is the last layer in the network.
            stop_words_list (List(str)): list of stop words.
            bad_words_list (List(str)): list of bad words.
            no_repeat_ngram_size (int): no repeat ngram size.
            output_generation_logits (bool): if True returns generation_logits in the outout of generate method.
            sampling_kwargs: Additional kwargs to set in the SamplingConfig.
        """
        raise NotImplementedError("To be implemented")

    def _pad_logits(self, logits_tensor):
        """Pads the logits tensor with 0's on the right."""
        raise NotImplementedError("To be implemented")

    @property
    def get_supported_models_list(self):
        """Supported model list."""
        raise NotImplementedError("To be implemented")

    @property
    def get_hidden_size(self):
        """Get hidden size."""
        raise NotImplementedError("To be implemented")

    @property
    def get_triton_input(self):
        """Get triton input."""
        raise NotImplementedError("To be implemented")

    @property
    def get_triton_output(self):
        raise NotImplementedError("To be implemented")

    def _infer_fn(self, prompts, inputs):
        """Shared helper function to prepare inference inputs and execute forward pass.

        Args:
            prompts: List of input prompts
            inputs: Dictionary of input parameters

        Returns:
            output_texts: List of generated text outputs
        """
        raise NotImplementedError("To be implemented")

    @batch
    @first_value(
        "max_output_len",
        "top_k",
        "top_p",
        "temperature",
        "random_seed",
        "no_repeat_ngram_size",
        "output_generation_logits",
        "output_context_logits",
    )
    def triton_infer_fn(self, **inputs: np.ndarray):  # pragma: no cover
        """Triton infer function for inference."""
        raise NotImplementedError("To be implemented")

    def ray_infer_fn(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Ray inference function that processes input dictionary and returns output without byte casting.

        Args:
            inputs (Dict[str, Any]): Input dictionary containing:
                - prompts: List of input prompts
                - max_output_len: Maximum output length (optional)
                - top_k: Top-k sampling parameter (optional)
                - top_p: Top-p sampling parameter (optional)
                - temperature: Sampling temperature (optional)
                - random_seed: Random seed (optional)
                - stop_words_list: List of stop words (optional)
                - bad_words_list: List of bad words (optional)
                - no_repeat_ngram_size: No repeat ngram size (optional)
                - lora_uids: LoRA UIDs (optional)
                - apply_chat_template: Whether to apply chat template (optional)
                - compute_logprob: Whether to compute log probabilities (optional)

        Returns:
            Dict[str, Any]: Output dictionary containing:
                - sentences: List of generated text outputs
                - log_probs: Log probabilities (if requested)
        """
        raise NotImplementedError("To be implemented")

    def _load_config_file(self):
        raise NotImplementedError("To be implemented")

    def _load(self):
        raise NotImplementedError("To be implemented")

    def unload_engine(self):
        """Unload engine."""
        raise NotImplementedError("To be implemented")
