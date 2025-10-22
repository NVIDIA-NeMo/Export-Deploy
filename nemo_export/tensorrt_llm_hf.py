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
from glob import glob
from pathlib import Path
from typing import List, Optional

from transformers import AutoConfig

from nemo_export.tensorrt_llm import TensorRTLLM
from nemo_export.utils import prepare_directory_for_export
from nemo_export.utils.constants import TRTLLM_ENGINE_DIR
from nemo_export_deploy_common.import_utils import (
    MISSING_TENSORRT_LLM_MSG,
    UnavailableError,
)

try:
    from tensorrt_llm._common import check_max_num_tokens
    from tensorrt_llm.builder import BuildConfig
    from tensorrt_llm.commands.build import build as build_trtllm
    from tensorrt_llm.mapping import Mapping
    from tensorrt_llm.models import (
        BaichuanForCausalLM,
        BertForQuestionAnswering,
        BertForSequenceClassification,
        BertModel,
        BloomForCausalLM,
        ChatGLMForCausalLM,
        CogVLMForCausalLM,
        CohereForCausalLM,
        DbrxForCausalLM,
        DeciLMForCausalLM,
        DecoderModel,
        DeepseekForCausalLM,
        DeepseekV2ForCausalLM,
        DiT,
        EagleForCausalLM,
        EncoderModel,
        FalconForCausalLM,
        GemmaForCausalLM,
        GPTForCausalLM,
        GPTJForCausalLM,
        GPTNeoXForCausalLM,
        GrokForCausalLM,
        LLaMAForCausalLM,
        MambaForCausalLM,
        MedusaForCausalLm,
        MLLaMAForCausalLM,
        MPTForCausalLM,
        OPTForCausalLM,
        Phi3ForCausalLM,
        PhiForCausalLM,
        QWenForCausalLM,
        RecurrentGemmaForCausalLM,
        ReDrafterForLLaMALM,
        ReDrafterForQWenLM,
        RobertaForQuestionAnswering,
        RobertaForSequenceClassification,
        RobertaModel,
        WhisperEncoder,
    )
    from tensorrt_llm.plugin import PluginConfig

    HAVE_TENSORRT_LLM = True
except (ImportError, ModuleNotFoundError):
    HAVE_TENSORRT_LLM = False

LOGGER = logging.getLogger("NeMo")


class TensorRTLLMHF(TensorRTLLM):
    """Exports HuggingFace checkpoints to TensorRT-LLM and run fast inference.

    This class provides functionality to export HuggingFace models to TensorRT-LLM
    format and run inference using the exported models. It inherits from TensorRTLLM
    and adds HuggingFace-specific export capabilities.

    Example:
        from nemo_export.tensorrt_llm_hf import TensorRTLLMHF

        trt_llm_exporter = TensorRTLLMHF(model_dir="/path/for/model/files")
        trt_llm_exporter.export_hf_model(
            hf_model_path="/path/to/huggingface/model",
            max_batch_size=8,
            tensor_parallelism_size=1,
        )

        output = trt_llm_exporter.forward(["Hi, how are you?", "I am good, thanks, how about you?"])
        print("output: ", output)
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
        """Initialize TensorRTLLMHF exporter.

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

        # Call parent class constructor
        super().__init__(
            model_dir=model_dir,
            lora_ckpt_list=lora_ckpt_list,
            load_model=load_model,
            use_python_runtime=use_python_runtime,
            enable_chunked_context=enable_chunked_context,
            max_tokens_in_paged_kv_cache=max_tokens_in_paged_kv_cache,
            multi_block_mode=multi_block_mode,
        )

    def export_hf_model(
        self,
        hf_model_path: str,
        max_batch_size: int = 8,
        tensor_parallelism_size: int = 1,
        max_input_len: int = 256,
        max_output_len: int = 256,
        max_num_tokens: Optional[int] = None,
        opt_num_tokens: Optional[int] = None,
        dtype: Optional[str] = None,
        max_seq_len: Optional[int] = 512,
        gemm_plugin: str = "auto",
        remove_input_padding: bool = True,
        use_paged_context_fmha: bool = True,
        paged_kv_cache: bool = True,
        tokens_per_block: int = 128,
        multiple_profiles: bool = False,
        reduce_fusion: bool = False,
        max_beam_width: int = 1,
        use_refit: bool = False,
        model_type: Optional[str] = None,
        delete_existing_files: bool = True,
    ):
        """Export a Hugging Face model to TensorRT-LLM format.

        This method exports a Hugging Face model to TensorRT-LLM format with various configuration
        options for model parallelism, quantization, and inference parameters.

        Args:
            hf_model_path (str): Path to the Hugging Face model directory.
            max_batch_size (int, optional): Maximum batch size. Defaults to 8.
            tensor_parallelism_size (int, optional): Size of tensor parallelism. Defaults to 1.
            max_input_len (int, optional): Maximum input sequence length. Defaults to 256.
            max_output_len (int, optional): Maximum output sequence length. Defaults to 256.
            max_num_tokens (Optional[int], optional): Maximum number of tokens. Defaults to None.
            opt_num_tokens (Optional[int], optional): Optimal number of tokens. Defaults to None.
            dtype (Optional[str], optional): Data type for model weights. Defaults to None.
            max_seq_len (Optional[int], optional): Maximum sequence length. Defaults to 512.
            gemm_plugin (str, optional): GEMM plugin type. Defaults to "auto".
            remove_input_padding (bool, optional): Remove input padding. Defaults to True.
            use_paged_context_fmha (bool, optional): Use paged context FMHA. Defaults to True.
            paged_kv_cache (bool, optional): Use paged KV cache. Defaults to True.
            tokens_per_block (int, optional): Tokens per block. Defaults to 128.
            multiple_profiles (bool, optional): Use multiple profiles. Defaults to False.
            reduce_fusion (bool, optional): Enable reduce fusion. Defaults to False.
            max_beam_width (int, optional): Maximum beam width. Defaults to 1.
            use_refit (bool, optional): Use refit. Defaults to False.
            model_type (Optional[str], optional): Type of the model. Defaults to None.
            delete_existing_files (bool, optional): Delete existing files. Defaults to True.

        Raises:
            ValueError: If model_type is not supported or dtype cannot be determined.
            FileNotFoundError: If config file is not found.
            RuntimeError: If there are errors reading the config file.
        """
        LOGGER.info("Starting HF export to TRT-LLM")
        if model_type is None:
            model_type = self.get_hf_model_type(hf_model_path)

        if model_type not in self.get_supported_hf_model_mapping:
            raise ValueError(
                f"Model {model_type} is not currently a supported model type. "
                f"Supported model types are: {self.get_supported_hf_model_mapping.keys()}."
            )

        if dtype is None:
            dtype = self.get_hf_model_dtype(hf_model_path)
            if dtype is None:
                raise ValueError("No dtype found in hf model config. Please specify a dtype.")

        prepare_directory_for_export(
            self.model_dir,
            delete_existing_files=delete_existing_files,
            subdir=TRTLLM_ENGINE_DIR,
        )

        if max_batch_size < 4:
            print("TensorRT-LLM may hit runtime issue with batch size is smaller than 4. Force set to 4")
            max_batch_size = 4

        plugin_config = PluginConfig()
        plugin_config.gemm_plugin = gemm_plugin
        if paged_kv_cache:
            plugin_config.enable_paged_kv_cache(tokens_per_block=tokens_per_block)
        else:
            plugin_config.paged_kv_cache = False
        plugin_config.remove_input_padding = remove_input_padding
        plugin_config.use_paged_context_fmha = use_paged_context_fmha
        plugin_config.multiple_profiles = multiple_profiles
        plugin_config.reduce_fusion = reduce_fusion
        max_seq_len = max_input_len + max_output_len
        max_num_tokens, opt_num_tokens = check_max_num_tokens(
            max_num_tokens=max_num_tokens,
            opt_num_tokens=opt_num_tokens,
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            max_input_len=max_input_len,
            max_beam_width=max_beam_width,
            remove_input_padding=remove_input_padding,
            enable_context_fmha=plugin_config.context_fmha,
            tokens_per_block=tokens_per_block,
            multiple_profiles=multiple_profiles,
        )
        build_dict = {
            "max_input_len": max_input_len,
            "max_output_len": max_output_len,
            "max_batch_size": max_batch_size,
            "max_beam_width": max_beam_width,
            "max_seq_len": max_seq_len,
            "max_num_tokens": max_num_tokens,
            "opt_num_tokens": opt_num_tokens,
            "strongly_typed": False,
            "builder_opt": None,
            "multiple_profiles": multiple_profiles,
            "use_refit": use_refit,
        }
        build_config = BuildConfig.from_dict(build_dict, plugin_config=plugin_config)
        for rank in range(tensor_parallelism_size):
            LOGGER.info(f"Iterating over rank:{rank}")
            mapping = Mapping(
                world_size=tensor_parallelism_size,
                rank=rank,
                tp_size=tensor_parallelism_size,
            )
            trtllm_model_class = self.get_supported_hf_model_mapping[model_type]
            model = trtllm_model_class.from_hugging_face(
                hf_model_path,
                dtype,
                mapping=mapping,
            )
            engine = build_trtllm(model, build_config)
            engine.save(self.engine_dir)
        # Copy HF tokenizer files to root model directory
        for path in glob(os.path.join(hf_model_path, "*.json")):
            shutil.copy(path, self.model_dir)
        # Copy sentencepiece model to model directory
        for path in glob(os.path.join(hf_model_path, "*.model")):
            shutil.copy(path, self.model_dir)
        LOGGER.info(f"Generarated TRT-LLM checkpoint at dir:{self.model_dir}")
        LOGGER.info(f"Loading the TRT-LLM checkpoint:{self.model_dir}")
        self._load()

    def get_hf_model_type(self, model_dir: str) -> str:
        """Get the model type from a Hugging Face model directory.

        This method infers the model type from the 'architectures' field in the model's config.json file.

        Args:
            model_dir (str): Path to the Hugging Face model directory or model ID at Hugging Face Hub.

        Returns:
            str: The inferred model type (e.g., "LlamaForCausalLM").

        Raises:
            ValueError: If the architecture choice is ambiguous.
        """
        config = AutoConfig.from_pretrained(model_dir)

        if len(config.architectures) != 1:
            raise ValueError(
                f"Ambiguous architecture choice: {config.architectures}, please specify model_type explicitly."
            )

        return config.architectures[0]

    def get_hf_model_dtype(self, model_dir: str) -> Optional[str]:
        """Get the data type from a Hugging Face model directory.

        This method reads the config file from a Hugging Face model directory and identifies
        the model's data type from various possible locations in the config.

        Args:
            model_dir (str): Path to the Hugging Face model directory.

        Returns:
            Optional[str]: The model's data type if found in config, None otherwise.

        Raises:
            FileNotFoundError: If the config file is not found.
            ValueError: If the config file contains invalid JSON.
            RuntimeError: If there are errors reading the config file.
        """
        config_path = Path(model_dir) / "config.json"

        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found at {config_path}")

        try:
            with open(config_path, "r") as f:
                config = json.load(f)
                # Check for dtype in different possible locations in the config
                if "torch_dtype" in config:
                    return config["torch_dtype"]
                elif "dtype" in config:
                    return config["dtype"]
                elif "pretrained_config" in config and "dtype" in config["pretrained_config"]:
                    return config["pretrained_config"]["dtype"]

                # If no explicit dtype found, check for other indicators
                if "fp16" in config and config["fp16"]:
                    return "float16"
                elif "bf16" in config and config["bf16"]:
                    return "bfloat16"

            return None
        except json.JSONDecodeError:
            raise ValueError(f"Invalid JSON in config file at {config_path}")
        except Exception as e:
            raise RuntimeError(f"Error reading config file: {str(e)}")

    @property
    def get_supported_hf_model_mapping(self):
        """Supported HF Model Mapping."""
        HF_MODEL_CLASS_MAP = {
            "GPT2LMHeadModel": GPTForCausalLM,
            "GPT2LMHeadCustomModel": GPTForCausalLM,
            "GPTBigCodeForCausalLM": GPTForCausalLM,
            "Starcoder2ForCausalLM": GPTForCausalLM,
            "JAISLMHeadModel": GPTForCausalLM,
            "GPTForCausalLM": GPTForCausalLM,
            "NemotronForCausalLM": GPTForCausalLM,
            "OPTForCausalLM": OPTForCausalLM,
            "BloomForCausalLM": BloomForCausalLM,
            "RWForCausalLM": FalconForCausalLM,
            "FalconForCausalLM": FalconForCausalLM,
            "PhiForCausalLM": PhiForCausalLM,
            "Phi3ForCausalLM": Phi3ForCausalLM,
            "Phi3VForCausalLM": Phi3ForCausalLM,
            "Phi3SmallForCausalLM": Phi3ForCausalLM,
            "PhiMoEForCausalLM": Phi3ForCausalLM,
            "MambaForCausalLM": MambaForCausalLM,
            "GPTNeoXForCausalLM": GPTNeoXForCausalLM,
            "GPTJForCausalLM": GPTJForCausalLM,
            "MptForCausalLM": MPTForCausalLM,
            "MPTForCausalLM": MPTForCausalLM,
            "GLMModel": ChatGLMForCausalLM,
            "ChatGLMModel": ChatGLMForCausalLM,
            "ChatGLMForCausalLM": ChatGLMForCausalLM,
            "ChatGLMForConditionalGeneration": ChatGLMForCausalLM,
            "LlamaForCausalLM": LLaMAForCausalLM,
            "LlavaLlamaModel": LLaMAForCausalLM,
            "ExaoneForCausalLM": LLaMAForCausalLM,
            "MistralForCausalLM": LLaMAForCausalLM,
            "MixtralForCausalLM": LLaMAForCausalLM,
            "ArcticForCausalLM": LLaMAForCausalLM,
            "Grok1ModelForCausalLM": GrokForCausalLM,
            "InternLMForCausalLM": LLaMAForCausalLM,
            "InternLM2ForCausalLM": LLaMAForCausalLM,
            "InternLMXComposer2ForCausalLM": LLaMAForCausalLM,
            "GraniteForCausalLM": LLaMAForCausalLM,
            "GraniteMoeForCausalLM": LLaMAForCausalLM,
            "MedusaForCausalLM": MedusaForCausalLm,
            "MedusaLlamaForCausalLM": MedusaForCausalLm,
            "ReDrafterForLLaMALM": ReDrafterForLLaMALM,
            "ReDrafterForQWenLM": ReDrafterForQWenLM,
            "BaichuanForCausalLM": BaichuanForCausalLM,
            "BaiChuanForCausalLM": BaichuanForCausalLM,
            "SkyworkForCausalLM": LLaMAForCausalLM,
            "GEMMA": GemmaForCausalLM,
            "GEMMA2": GemmaForCausalLM,
            "QWenLMHeadModel": QWenForCausalLM,
            "QWenForCausalLM": QWenForCausalLM,
            "Qwen2ForCausalLM": QWenForCausalLM,
            "Qwen2MoeForCausalLM": QWenForCausalLM,
            "Qwen2ForSequenceClassification": QWenForCausalLM,
            "Qwen2VLForConditionalGeneration": QWenForCausalLM,
            "Qwen2VLModel": QWenForCausalLM,
            "WhisperEncoder": WhisperEncoder,
            "EncoderModel": EncoderModel,
            "DecoderModel": DecoderModel,
            "DbrxForCausalLM": DbrxForCausalLM,
            "RecurrentGemmaForCausalLM": RecurrentGemmaForCausalLM,
            "CogVLMForCausalLM": CogVLMForCausalLM,
            "DiT": DiT,
            "DeepseekForCausalLM": DeepseekForCausalLM,
            "DeciLMForCausalLM": DeciLMForCausalLM,
            "DeepseekV2ForCausalLM": DeepseekV2ForCausalLM,
            "EagleForCausalLM": EagleForCausalLM,
            "CohereForCausalLM": CohereForCausalLM,
            "MLLaMAModel": MLLaMAForCausalLM,
            "MllamaForConditionalGeneration": MLLaMAForCausalLM,
            "BertForQuestionAnswering": BertForQuestionAnswering,
            "BertForSequenceClassification": BertForSequenceClassification,
            "BertModel": BertModel,
            "RobertaModel": RobertaModel,
            "RobertaForQuestionAnswering": RobertaForQuestionAnswering,
            "RobertaForSequenceClassification": RobertaForSequenceClassification,
        }
        return HF_MODEL_CLASS_MAP
