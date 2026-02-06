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

import json
import logging
import os
import shutil
import tempfile
import warnings
from glob import glob
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from megatron.core.export.data_type import DataType
from megatron.core.export.export_config import ExportConfig
from megatron.core.export.model_type import ModelType
from megatron.core.export.trtllm.model_to_trllm_mapping.default_conversion_dict import (
    DEFAULT_CONVERSION_DICT,
)
from transformers import PreTrainedTokenizerBase

from nemo_deploy import ITritonDeployable
from nemo_deploy.utils import cast_output, str_ndarray2list
from nemo_export.trt_llm.nemo_ckpt_loader.nemo_file import (
    get_model_type,
    get_tokenizer,
    get_weights_dtype,
    load_nemo_model,
)
from nemo_export.trt_llm.qnemo import qnemo_to_tensorrt_llm
from nemo_export.trt_llm.qnemo.utils import is_qnemo_checkpoint
from nemo_export.trt_llm.tensorrt_llm_run import (
    generate,
    load,
    unload_engine,
)
from nemo_export.trt_llm.utils import determine_quantization_settings, is_rank
from nemo_export.utils import prepare_directory_for_export
from nemo_export.utils.constants import TRTLLM_ENGINE_DIR
from nemo_export_deploy_common.import_utils import (
    MISSING_TENSORRT_LLM_MSG,
    MISSING_TRITON_MSG,
    UnavailableError,
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

try:
    import tensorrt_llm
    from tensorrt_llm.layers import MoeConfig

    HAVE_TENSORRT_LLM = True
except (ImportError, ModuleNotFoundError):
    HAVE_TENSORRT_LLM = False

try:
    from nemo.collections.llm.api import export_ckpt

    HAVE_NEMO_EXPORT = True
except (ImportError, ModuleNotFoundError):
    HAVE_NEMO_EXPORT = False

if HAVE_TENSORRT_LLM:
    from megatron.core.export.trtllm.trtllm_helper import TRTLLMHelper

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
        if not HAVE_TENSORRT_LLM:
            raise UnavailableError(MISSING_TENSORRT_LLM_MSG)
        if not HAVE_PYTRITON:
            raise UnavailableError(MISSING_TRITON_MSG)

        if use_python_runtime:
            if enable_chunked_context is not None or max_tokens_in_paged_kv_cache is not None:
                raise Exception(
                    "enable_chunked_context and max_tokens_in_paged_kv_cache options "
                    "work only with the TensorRT-LLM C++ runtime. Please set "
                    "use_python_runtime=False to use these options."
                )

        self.model_dir = model_dir
        self.engine_dir = os.path.join(model_dir, TRTLLM_ENGINE_DIR)
        self.lora_ckpt_list = lora_ckpt_list
        self.use_python_runtime = use_python_runtime
        self.enable_chunked_context = enable_chunked_context if enable_chunked_context is not None else False
        self.max_tokens_in_paged_kv_cache = max_tokens_in_paged_kv_cache
        self.multi_block_mode = multi_block_mode
        self.model = None
        self.tokenizer = None
        self.config = None

        if load_model:
            self._load()

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
        prepare_directory_for_export(
            self.model_dir,
            delete_existing_files=delete_existing_files,
            subdir=TRTLLM_ENGINE_DIR,
        )

        self.model = None

        if max_output_len is not None:
            warnings.warn(
                "Parameter max_output_len is deprecated and will be removed.",
                DeprecationWarning,
                stacklevel=2,
            )
            max_output_len = max_output_len if max_output_len is not None else 256

            if max_seq_len is None:
                max_seq_len = max_input_len + max_output_len
            else:
                warnings.warn(
                    f"Parameter max_output_len will be overwritten by max_seq_len={max_seq_len}.",
                    DeprecationWarning,
                    stacklevel=2,
                )

        max_seq_len = max_seq_len if max_seq_len is not None else 512

        if max_batch_size < 4:
            warnings.warn(
                "TensorRT LLM may hit a runtime issue with batch size is smaller than 4 on some models. Force set to 4",
                stacklevel=2,
            )
            max_batch_size = 4

        is_export_rank = is_rank(build_rank)

        if is_export_rank:
            tmp_dir = tempfile.TemporaryDirectory()
            nemo_export_dir = Path(tmp_dir.name)

            if is_qnemo_checkpoint(nemo_checkpoint_path):
                nemo_export_dir = nemo_checkpoint_path

                self.tokenizer = get_tokenizer(nemo_checkpoint_path)

                model_config = None

                qnemo_to_tensorrt_llm(
                    nemo_checkpoint_path=nemo_checkpoint_path,
                    engine_dir=self.engine_dir,
                    max_input_len=max_input_len,
                    max_seq_len=max_seq_len,
                    max_batch_size=max_batch_size,
                    max_prompt_embedding_table_size=0,
                    tensor_parallel_size=tensor_parallelism_size,
                    pipeline_parallel_size=pipeline_parallelism_size,
                    use_parallel_embedding=use_parallel_embedding,
                    paged_kv_cache=paged_kv_cache,
                    use_paged_context_fmha=use_paged_context_fmha,
                    remove_input_padding=remove_input_padding,
                    use_lora_plugin=use_lora_plugin,
                    lora_target_modules=lora_target_modules,
                    max_lora_rank=max_lora_rank,
                    max_num_tokens=max_num_tokens,
                    opt_num_tokens=opt_num_tokens,
                    multiple_profiles=multiple_profiles,
                    reduce_fusion=reduce_fusion,
                )
            else:
                if model_type is None:
                    # For NeMo 2.0 models we can get model_type from the model class name
                    model_type = get_model_type(nemo_checkpoint_path)

                if model_type is None:
                    raise ValueError(
                        "Parameter model_type needs to be provided and cannot be inferred from the checkpoint. "
                        "Please specify it explicitely."
                    )

                if model_type not in self.get_supported_models_list:
                    raise ValueError(
                        f"Model {model_type} is not currently a supported model type. "
                        f"Supported model types are: {self.get_supported_models_list}."
                    )

                if dtype is None:
                    dtype = get_weights_dtype(nemo_checkpoint_path)

                if dtype is None:
                    raise ValueError(
                        "Parameter dtype needs to be provided and cannot be inferred from the checkpoint. "
                        "Please specify it explicitely."
                    )

                model, model_config, self.tokenizer = load_nemo_model(nemo_checkpoint_path, nemo_export_dir)

                share_embeddings_and_output_weights = model_config.get("share_embeddings_and_output_weights", False)
                fp8_quantized, fp8_kvcache = determine_quantization_settings(model_config, fp8_quantized, fp8_kvcache)

                # We build the transformer config using the nemo model config.
                transformer_config = self.get_transformer_config(model_config)
                input_model_type = getattr(ModelType, model_type)

                # MCore export supports some default conversion dictionaries
                mcore_model_conversion_dict = DEFAULT_CONVERSION_DICT

                # All Mcore conversion dicts start with "decoder.layers.4.blah.blah" , while nemo models start with "model.decoder.layers.4.blahblah". so we append model. to the keys
                nemo_model_conversion_dict = {
                    f"model.{key}": value for key, value in mcore_model_conversion_dict.items()
                } | {  # Mapping for NeMo 2.0
                    f"module.{key}": value for key, value in mcore_model_conversion_dict.items()
                }

                # TODO: Workaround: Gemma uses gated activation, while mcore does not handle openai-gelu
                # as a gated function. Remove once !11614 is merged.
                activation = model_config.get("activation", "gelu")
                if activation == "openai-gelu" and input_model_type.name == "gemma":
                    activation = "geglu"

                trtllm_helper = TRTLLMHelper(
                    transformer_config=transformer_config,
                    model_type=input_model_type,
                    trtllm_conversion_dict=nemo_model_conversion_dict,
                    position_embedding_type=model_config.get("position_embedding_type"),
                    max_position_embeddings=model_config.get("max_position_embeddings"),
                    rotary_percentage=model_config.get("rotary_percentage", 1.0),
                    rotary_base=model_config.get("rotary_base", 10000),
                    moe_tp_mode=model_config.get("moe_tp_mode", 2),
                    multi_query_mode=model_config.get("multi_query_mode", False),
                    activation=activation,
                    seq_len_interpolation_factor=model_config.get("seq_len_interpolation_factor"),
                    moe_renorm_mode=model_config.get(
                        "moe_renorm_mode",
                        MoeConfig.ExpertScaleNormalizationMode.RENORMALIZE,
                    ),
                    share_embeddings_and_output_weights=share_embeddings_and_output_weights,
                )

                input_dtype = getattr(DataType, dtype)
                export_config = ExportConfig(
                    tensor_parallelism_size,
                    pipeline_parallelism_size,
                    use_parallel_embedding,
                    share_embeddings_and_output_weights,
                )

                trtllm_model_weights_list, trtllm_model_config_list = (
                    trtllm_helper.get_trtllm_pretrained_config_and_model_weights(
                        model_state_dict=model,
                        export_config=export_config,
                        dtype=input_dtype,
                        state_dict_split_by_layer_numbers=False,
                        fp8_quantized=fp8_quantized,
                        fp8_kvcache=fp8_kvcache,
                    )
                )

                for trtllm_model_weights, trtllm_model_config in zip(
                    trtllm_model_weights_list, trtllm_model_config_list
                ):
                    trtllm_helper.build_and_save_engine(
                        max_input_len=max_input_len,
                        max_output_len=max_output_len,
                        max_batch_size=max_batch_size,
                        engine_dir=self.engine_dir,
                        trtllm_model_weights=trtllm_model_weights,
                        trtllm_model_config=trtllm_model_config,
                        lora_ckpt_list=self.lora_ckpt_list,
                        use_lora_plugin=use_lora_plugin,
                        max_lora_rank=max_lora_rank,
                        lora_target_modules=lora_target_modules,
                        max_prompt_embedding_table_size=0,
                        paged_kv_cache=paged_kv_cache,
                        remove_input_padding=remove_input_padding,
                        paged_context_fmha=use_paged_context_fmha,  # TODO: rename paged_context_fmha -> use_paged_context_fmha in MCore
                        use_refit=False,
                        max_num_tokens=max_num_tokens,
                        max_seq_len=max_seq_len,
                        opt_num_tokens=opt_num_tokens,
                        max_beam_width=1,
                        tokens_per_block=128,
                        multiple_profiles=multiple_profiles,
                        gpt_attention_plugin=gpt_attention_plugin,
                        gemm_plugin=gemm_plugin,
                    )

            tokenizer_path = os.path.join(nemo_export_dir, "tokenizer.model")
            tokenizer_path_nemo2 = os.path.join(nemo_export_dir, "nemo_context")
            vocab_path = os.path.join(nemo_export_dir, "vocab.json")
            if isinstance(self.tokenizer, PreTrainedTokenizerBase):
                self.tokenizer.save_pretrained(self.model_dir)
            elif os.path.exists(tokenizer_path):
                shutil.copy(tokenizer_path, self.model_dir)
            elif os.path.exists(tokenizer_path_nemo2):
                # Copy HF tokenizer files to root model directory
                for path in glob(os.path.join(tokenizer_path_nemo2, "nemo_tokenizer", "*.json")):
                    shutil.copy(path, self.model_dir)
                # Copy SentencePiece tokenizer.model
                for path in glob(os.path.join(tokenizer_path_nemo2, "*.model")):
                    shutil.copy(path, os.path.join(self.model_dir, "tokenizer.model"))
            elif os.path.exists(vocab_path):
                shutil.copy(vocab_path, os.path.join(self.model_dir, "vocab.json"))

            nemo_model_config = os.path.join(nemo_export_dir, "model_config.yaml")
            if os.path.exists(nemo_model_config):
                shutil.copy(nemo_model_config, self.model_dir)

            tmp_dir.cleanup()

        if is_export_rank and model_config is not None:
            self._export_to_nim_format(model_config, model_type)

        if tensorrt_llm.mpi_world_size() > 1:
            tensorrt_llm.mpi_barrier()

        if is_export_rank and load_model:
            self._load()

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
        # Convert NeMo checkpoint to HF format
        tmp_hf_export_dir = tempfile.TemporaryDirectory()
        hf_model_path = tmp_hf_export_dir.name

        try:
            LOGGER.info(f"Converting NeMo checkpoint to HF format at {hf_model_path}...")
            export_ckpt(
                path=nemo_checkpoint_path,
                target="hf",
                output_path=hf_model_path,
                overwrite=True,
            )

            if not any(Path(hf_model_path).iterdir()):
                raise Exception("HF conversion produced empty directory")

            LOGGER.info("NeMo to HF conversion succeeded. Now exporting HF model to TensorRT-LLM...")

            # Import and use HF export functionality
            from nemo_export.tensorrt_llm_hf import TensorRTLLMHF

            # Create a temporary HF exporter to handle the export
            hf_exporter = TensorRTLLMHF(
                model_dir=self.model_dir,
                lora_ckpt_list=self.lora_ckpt_list,
                load_model=False,
                use_python_runtime=self.use_python_runtime,
                enable_chunked_context=self.enable_chunked_context if self.enable_chunked_context else None,
                max_tokens_in_paged_kv_cache=self.max_tokens_in_paged_kv_cache,
                multi_block_mode=self.multi_block_mode,
            )

            # Handle max_output_len deprecation
            if max_output_len is not None:
                warnings.warn(
                    "Parameter max_output_len is deprecated and will be removed.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                if max_seq_len is None:
                    max_seq_len = max_input_len + max_output_len

            max_seq_len = max_seq_len if max_seq_len is not None else 512

            # Export using HF pipeline
            hf_exporter.export_hf_model(
                hf_model_path=hf_model_path,
                max_batch_size=max_batch_size,
                tensor_parallelism_size=tensor_parallelism_size,
                max_input_len=max_input_len,
                max_output_len=max_output_len if max_output_len is not None else 256,
                max_num_tokens=max_num_tokens,
                opt_num_tokens=opt_num_tokens,
                dtype=dtype,
                max_seq_len=max_seq_len,
                gemm_plugin=gemm_plugin,
                remove_input_padding=remove_input_padding,
                use_paged_context_fmha=use_paged_context_fmha,
                paged_kv_cache=paged_kv_cache,
                multiple_profiles=multiple_profiles,
                reduce_fusion=reduce_fusion,
                model_type=None,
                delete_existing_files=delete_existing_files,
            )

            # Load the TensorRT-LLM engine that was built by the HF exporter
            # Both TensorRTLLM and TensorRTLLMHF share the same model_dir and engine_dir
            self._load()

            LOGGER.info("HuggingFace fallback export succeeded!")

        finally:
            # Always clean up temporary directory
            tmp_hf_export_dir.cleanup()

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
        LOGGER.info("Starting export with HF fallback...")

        # First try the standard NeMo export
        try:
            LOGGER.info("Attempting standard NeMo export...")
            self._export_nemo_checkpoint(
                nemo_checkpoint_path=nemo_checkpoint_path,
                model_type=model_type,
                delete_existing_files=delete_existing_files,
                tensor_parallelism_size=tensor_parallelism_size,
                pipeline_parallelism_size=pipeline_parallelism_size,
                max_input_len=max_input_len,
                max_output_len=max_output_len,
                max_batch_size=max_batch_size,
                use_parallel_embedding=use_parallel_embedding,
                paged_kv_cache=paged_kv_cache,
                remove_input_padding=remove_input_padding,
                use_paged_context_fmha=use_paged_context_fmha,
                dtype=dtype,
                load_model=load_model,
                use_lora_plugin=use_lora_plugin,
                lora_target_modules=lora_target_modules,
                max_lora_rank=max_lora_rank,
                max_num_tokens=max_num_tokens,
                opt_num_tokens=opt_num_tokens,
                max_seq_len=max_seq_len,
                multiple_profiles=multiple_profiles,
                gpt_attention_plugin=gpt_attention_plugin,
                gemm_plugin=gemm_plugin,
                reduce_fusion=reduce_fusion,
                fp8_quantized=fp8_quantized,
                fp8_kvcache=fp8_kvcache,
                build_rank=build_rank,
            )
            LOGGER.info("Standard NeMo export succeeded!")
            return
        except Exception as nemo_export_error:
            LOGGER.warning(f"Standard NeMo export failed: {str(nemo_export_error)}")
            LOGGER.info("Attempting HuggingFace fallback export...")

            # Check if we can do HF export
            if not HAVE_NEMO_EXPORT:
                raise Exception(
                    f"Standard NeMo export failed and NeMo export_ckpt is not available for HF fallback. "
                    f"Original error: {str(nemo_export_error)}"
                )

            # Check if it's a NeMo2 checkpoint
            if not (Path(nemo_checkpoint_path).exists() and (Path(nemo_checkpoint_path) / "context").is_dir()):
                raise Exception(
                    f"Standard NeMo export failed and checkpoint is not a NeMo2 checkpoint. "
                    f"HF fallback only works with NeMo2 checkpoints. "
                    f"Original error: {str(nemo_export_error)}"
                )

            # Try HF export fallback
            try:
                self.export_with_hf(
                    nemo_checkpoint_path=nemo_checkpoint_path,
                    model_type=model_type,
                    delete_existing_files=delete_existing_files,
                    tensor_parallelism_size=tensor_parallelism_size,
                    max_input_len=max_input_len,
                    max_output_len=max_output_len,
                    max_batch_size=max_batch_size,
                    paged_kv_cache=paged_kv_cache,
                    remove_input_padding=remove_input_padding,
                    use_paged_context_fmha=use_paged_context_fmha,
                    dtype=dtype,
                    max_num_tokens=max_num_tokens,
                    opt_num_tokens=opt_num_tokens,
                    max_seq_len=max_seq_len,
                    multiple_profiles=multiple_profiles,
                    gemm_plugin=gemm_plugin,
                    reduce_fusion=reduce_fusion,
                )
            except Exception as hf_export_error:
                raise Exception(
                    f"Both NeMo export and HF fallback export failed.\n"
                    f"NeMo export error: {str(nemo_export_error)}\n"
                    f"HF fallback error: {str(hf_export_error)}"
                )

    def _export_to_nim_format(self, model_config: Dict[str, Any], model_type: str):
        """Exports the model configuration to a specific format required by NIM.

        This method performs the following steps:

        1. Copies the generation_config.json (if present) from the nemo_context directory to the root model directory.
        2. Creates a dummy Hugging Face configuration file based on the provided model configuration and type.

        Args:
            model_config (dict): A dictionary containing the model configuration parameters.
            model_type (str): The type of the model (e.g., "llama").
        """
        generation_config_path = os.path.join(self.model_dir, "nemo_context", "artifacts", "generation_config.json")
        if os.path.isfile(generation_config_path):
            shutil.copy(generation_config_path, self.model_dir)

        # Fields "architectures" and "model_type" are required by HF but not relevant for NIM
        seq_len_interpolation_factor = model_config.get("seq_len_interpolation_factor")
        hf_config = {
            "max_position_embeddings": model_config.get("encoder_seq_length"),
            "architectures": ["LLaMAForCausalLM"],
            "rope_scaling": (
                None
                if seq_len_interpolation_factor is None
                else {
                    "factor": seq_len_interpolation_factor,
                    "rope_type": "default",
                }
            ),
            "model_type": model_type,
        }
        with open(os.path.join(self.model_dir, "config.json"), "w") as f:
            json.dump(hf_config, f, indent=2)
            f.write("\n")

    def get_transformer_config(self, nemo_model_config):
        """Given nemo model config get transformer config."""
        from megatron.core.transformer.transformer_config import TransformerConfig

        normalization = nemo_model_config.get("normalization", "layernorm")
        transformer_config_normalization = "LayerNorm"
        layernorm_zero_centered_gamma = nemo_model_config.get("layernorm_zero_centered_gamma", False)
        if normalization == "layernorm1p":
            layernorm_zero_centered_gamma = True
        elif normalization == "rmsnorm":
            transformer_config_normalization = "RMSNorm"

        num_moe_experts = nemo_model_config.get("num_moe_experts", 0)
        conf = TransformerConfig(
            num_layers=nemo_model_config.get("num_layers"),
            moe_router_topk=nemo_model_config.get("moe_router_topk", 0),
            num_attention_heads=nemo_model_config.get("num_attention_heads"),
            num_query_groups=nemo_model_config.get("num_query_groups", nemo_model_config["num_attention_heads"]),
            kv_channels=nemo_model_config.get("kv_channels", None),
            hidden_size=nemo_model_config.get("hidden_size"),
            ffn_hidden_size=nemo_model_config.get("ffn_hidden_size"),
            layernorm_epsilon=nemo_model_config.get("layernorm_epsilon"),
            add_bias_linear=nemo_model_config.get("bias"),
            num_moe_experts=num_moe_experts if num_moe_experts > 0 else None,
            normalization=transformer_config_normalization,
            layernorm_zero_centered_gamma=layernorm_zero_centered_gamma,
            gated_linear_unit=nemo_model_config.get("gated_linear_unit", False),
        )
        return conf

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
        if self.model is None:
            raise Exception(
                "A nemo checkpoint should be exported to TensorRT-LLM and "
                "then it should be loaded first to run inference."
            )
        else:
            if torch.distributed.is_initialized() or tensorrt_llm.mpi_world_size() > 1:
                multiprocessed_env = True
            else:
                multiprocessed_env = False

            return generate(
                input_texts=input_texts,
                max_output_len=max_output_len,
                host_context=self.model,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                lora_uids=lora_uids,
                stop_words_list=stop_words_list,
                bad_words_list=bad_words_list,
                no_repeat_ngram_size=no_repeat_ngram_size,
                output_log_probs=output_log_probs,
                multiprocessed_env=multiprocessed_env,
                output_context_logits=output_context_logits,
                output_generation_logits=output_generation_logits,
                **sampling_kwargs,
            )

    def _pad_logits(self, logits_tensor):
        """Pads the logits tensor with 0's on the right."""
        padding_len = max([logit_tensor.shape[0] for logit_tensor in logits_tensor])
        for i, tensor in enumerate(logits_tensor):
            tensor_len = tensor.shape[0]
            if tensor_len < padding_len:
                padding_diff = padding_len - tensor_len
                # padding_diff num of rows of zeros are added at the bottom
                logits_tensor[i] = F.pad(tensor, (0, 0, 0, padding_diff), mode="constant", value=0)
        return logits_tensor

    @property
    def get_supported_models_list(self):
        """Supported model list."""
        # gpt and gptnext are the same. Keeping the gptnext due to backward compatibility.
        return ["gpt", "gptnext", "llama", "falcon", "starcoder", "mixtral", "gemma"]

    @property
    def get_hidden_size(self):
        """Get hidden size."""
        if self.config is None:
            return None
        else:
            return self.config["pretrained_config"]["hidden_size"]

    @property
    def get_triton_input(self):
        """Get triton input."""
        inputs = (
            Tensor(name="prompts", shape=(-1,), dtype=bytes),
            Tensor(name="max_output_len", shape=(-1,), dtype=np.int_, optional=True),
            Tensor(name="top_k", shape=(-1,), dtype=np.int_, optional=True),
            Tensor(name="top_p", shape=(-1,), dtype=np.single, optional=True),
            Tensor(name="temperature", shape=(-1,), dtype=np.single, optional=True),
            Tensor(name="random_seed", shape=(-1,), dtype=np.int_, optional=True),
            Tensor(name="stop_words_list", shape=(-1,), dtype=bytes, optional=True),
            Tensor(name="bad_words_list", shape=(-1,), dtype=bytes, optional=True),
            Tensor(name="no_repeat_ngram_size", shape=(-1,), dtype=np.single, optional=True),
            Tensor(name="lora_uids", shape=(-1,), dtype=bytes, optional=True),
            Tensor(
                name="output_context_logits",
                shape=(-1,),
                dtype=np.bool_,
                optional=False,
            ),
            Tensor(
                name="output_generation_logits",
                shape=(-1,),
                dtype=np.bool_,
                optional=False,
            ),
        )
        return inputs

    @property
    def get_triton_output(self):
        outputs = (
            Tensor(name="outputs", shape=(-1,), dtype=bytes),
            Tensor(name="generation_logits", shape=(-1,), dtype=np.single),
            Tensor(name="context_logits", shape=(-1,), dtype=np.single),
        )
        return outputs

    def _infer_fn(self, prompts, inputs):
        """Shared helper function to prepare inference inputs and execute forward pass.

        Args:
            prompts: List of input prompts
            inputs: Dictionary of input parameters

        Returns:
            output_texts: List of generated text outputs
        """
        infer_input = {"input_texts": prompts}

        # Process common parameters
        if "max_output_len" in inputs:
            infer_input["max_output_len"] = inputs["max_output_len"]
        if "top_k" in inputs:
            infer_input["top_k"] = inputs["top_k"]
        if "top_p" in inputs:
            infer_input["top_p"] = inputs["top_p"]
        if "temperature" in inputs:
            infer_input["temperature"] = inputs["temperature"]
        if "random_seed" in inputs:
            infer_input["random_seed"] = inputs["random_seed"]
        if "stop_words_list" in inputs:
            stop_words_list = inputs["stop_words_list"]
            # Ensure proper format for stop words
            if isinstance(stop_words_list, list) and stop_words_list:
                if isinstance(stop_words_list[0], str):
                    infer_input["stop_words_list"] = [[word] for word in stop_words_list]
                else:
                    infer_input["stop_words_list"] = stop_words_list
        if "bad_words_list" in inputs:
            bad_words_list = inputs["bad_words_list"]
            # Ensure proper format for bad words
            if isinstance(bad_words_list, list) and bad_words_list:
                if isinstance(bad_words_list[0], str):
                    infer_input["bad_words_list"] = [[word] for word in bad_words_list]
                else:
                    infer_input["bad_words_list"] = bad_words_list
        if "no_repeat_ngram_size" in inputs:
            infer_input["no_repeat_ngram_size"] = inputs["no_repeat_ngram_size"]
        if "lora_uids" in inputs:
            infer_input["lora_uids"] = inputs["lora_uids"]
        if "output_log_probs" in inputs:
            infer_input["output_log_probs"] = inputs["output_log_probs"]

        output_texts = self.forward(**infer_input)

        return output_texts

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
        output_dict = {}

        # Convert triton-specific inputs
        prompts = str_ndarray2list(inputs.pop("prompts"))

        # Convert numpy arrays to Python types for triton inputs
        processed_inputs = {}
        for key, value in inputs.items():
            if key == "stop_words_list":
                processed_inputs[key] = str_ndarray2list(value)
            elif key == "bad_words_list":
                processed_inputs[key] = str_ndarray2list(value)
            elif key == "lora_uids":
                lora_uids = np.char.decode(value.astype("bytes"), encoding="utf-8")
                processed_inputs[key] = lora_uids[0].tolist()
            else:
                processed_inputs[key] = value

        try:
            output_texts = self._infer_fn(prompts, processed_inputs)
            output_dict["outputs"] = cast_output(output_texts, np.bytes_)

        except Exception as error:
            err_msg = "An error occurred: {0}".format(str(error))
            output_dict["outputs"] = cast_output([err_msg] * len(prompts), np.bytes_)

        return output_dict

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
        output_dict = {}

        # Extract prompts - handle both list and single string cases
        prompts = inputs.get("prompts", [])
        if isinstance(prompts, str):
            prompts = [prompts]

        try:
            output_texts = self._infer_fn(prompts, inputs)
            output_dict["sentences"] = output_texts

        except Exception as error:
            err_msg = f"An error occurred: {str(error)}"
            LOGGER.error(err_msg)
            output_dict["sentences"] = [err_msg] * len(prompts)
            output_dict["error"] = err_msg

        return output_dict

    def _load_config_file(self):
        config_path = Path(self.engine_dir) / "config.json"
        if config_path.exists():
            with open(config_path, "r") as f:
                self.config = json.load(f)
        else:
            raise FileNotFoundError(f"File: {config_path} could not be found.")

    def _load(self):
        self.model = None
        self.tokenizer = None
        self.config = None

        if Path(self.model_dir).exists():
            folders = os.listdir(self.model_dir)
            if len(folders) > 0:
                try:
                    self._load_config_file()
                    self.tokenizer = get_tokenizer(self.model_dir)
                    self.model = load(
                        tokenizer=self.tokenizer,
                        engine_dir=self.engine_dir,
                        lora_ckpt_list=self.lora_ckpt_list,
                        use_python_runtime=self.use_python_runtime,
                        enable_chunked_context=self.enable_chunked_context,
                        max_tokens_in_paged_kv_cache=self.max_tokens_in_paged_kv_cache,
                        multi_block_mode=self.multi_block_mode,
                    )
                except Exception as error:
                    raise RuntimeError(
                        "Files in the TensorRT-LLM folder are corrupted and the model needs to be exported again."
                    ) from error

    def unload_engine(self):
        """Unload engine."""
        unload_engine()
