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
import os
from unittest.mock import (
    MagicMock,
    mock_open,
    patch,
)

import pytest


@pytest.mark.run_only_on("GPU")
@pytest.mark.unit
def test_tensorrt_llm_hf_initialization():
    """Test TensorRTLLMHF class initialization with various parameters."""
    try:
        import tensorrt_llm  # noqa: F401
    except ImportError:
        pytest.skip("Could not import TRTLLM helpers. tensorrt_llm is likely not installed")
        return

    from nemo_export.tensorrt_llm_hf import TensorRTLLMHF

    # Test basic initialization
    model_dir = "/tmp/test_hf_model_dir"
    trt_llm_hf = TensorRTLLMHF(model_dir=model_dir, load_model=False)
    assert trt_llm_hf.model_dir == model_dir
    assert trt_llm_hf.engine_dir == os.path.join(model_dir, "trtllm_engine")
    assert trt_llm_hf.model is None
    assert trt_llm_hf.tokenizer is None
    assert trt_llm_hf.config is None

    # Test initialization with lora checkpoints
    lora_ckpt_list = ["/path/to/hf_lora1", "/path/to/hf_lora2"]
    trt_llm_hf = TensorRTLLMHF(model_dir=model_dir, lora_ckpt_list=lora_ckpt_list, load_model=False)
    assert trt_llm_hf.lora_ckpt_list == lora_ckpt_list

    # Test initialization with python runtime options
    trt_llm_hf = TensorRTLLMHF(
        model_dir=model_dir,
        use_python_runtime=False,
        enable_chunked_context=True,
        max_tokens_in_paged_kv_cache=2048,
        multi_block_mode=True,
        load_model=False,
    )
    assert trt_llm_hf.use_python_runtime is False
    assert trt_llm_hf.enable_chunked_context is True
    assert trt_llm_hf.max_tokens_in_paged_kv_cache == 2048
    assert trt_llm_hf.multi_block_mode is True


@pytest.mark.run_only_on("GPU")
@pytest.mark.unit
def test_tensorrt_llm_hf_get_hf_model_type():
    """Test getting model type from HF config."""
    try:
        import tensorrt_llm  # noqa: F401
    except ImportError:
        pytest.skip("Could not import TRTLLM helpers. tensorrt_llm is likely not installed")
        return

    from nemo_export.tensorrt_llm_hf import TensorRTLLMHF

    trt_llm_hf = TensorRTLLMHF(model_dir="/tmp/test_model", load_model=False)

    # Test with LlamaForCausalLM architecture
    with patch("transformers.AutoConfig.from_pretrained") as mock_config:
        mock_config.return_value.architectures = ["LlamaForCausalLM"]
        model_type = trt_llm_hf.get_hf_model_type("/tmp/model")
        assert model_type == "LlamaForCausalLM"

    # Test with different model architectures
    test_architectures = [
        "GPT2LMHeadModel",
        "MistralForCausalLM",
        "Phi3ForCausalLM",
        "QWenForCausalLM",
    ]

    for arch in test_architectures:
        with patch("transformers.AutoConfig.from_pretrained") as mock_config:
            mock_config.return_value.architectures = [arch]
            model_type = trt_llm_hf.get_hf_model_type("/tmp/model")
            assert model_type == arch


@pytest.mark.run_only_on("GPU")
@pytest.mark.unit
def test_tensorrt_llm_hf_get_hf_model_type_ambiguous():
    """Test getting model type with ambiguous architecture."""
    try:
        import tensorrt_llm  # noqa: F401
    except ImportError:
        pytest.skip("Could not import TRTLLM helpers. tensorrt_llm is likely not installed")
        return

    from nemo_export.tensorrt_llm_hf import TensorRTLLMHF

    trt_llm_hf = TensorRTLLMHF(model_dir="/tmp/test_model", load_model=False)

    # Test with multiple architectures
    with patch("transformers.AutoConfig.from_pretrained") as mock_config:
        mock_config.return_value.architectures = ["Model1", "Model2"]
        with pytest.raises(ValueError) as exc_info:
            trt_llm_hf.get_hf_model_type("/tmp/model")
        assert "Ambiguous architecture choice" in str(exc_info.value)

    # Test with empty architectures list
    with patch("transformers.AutoConfig.from_pretrained") as mock_config:
        mock_config.return_value.architectures = []
        with pytest.raises(ValueError) as exc_info:
            trt_llm_hf.get_hf_model_type("/tmp/model")
        assert "Ambiguous architecture choice" in str(exc_info.value)


@pytest.mark.run_only_on("GPU")
@pytest.mark.unit
def test_tensorrt_llm_hf_get_hf_model_dtype_torch_dtype():
    """Test getting model dtype from HF config with torch_dtype field."""
    try:
        import tensorrt_llm  # noqa: F401
    except ImportError:
        pytest.skip("Could not import TRTLLM helpers. tensorrt_llm is likely not installed")
        return

    from nemo_export.tensorrt_llm_hf import TensorRTLLMHF

    trt_llm_hf = TensorRTLLMHF(model_dir="/tmp/test_model", load_model=False)

    # Test with torch_dtype field
    mock_config = {"torch_dtype": "float16"}

    with (
        patch("pathlib.Path.exists", return_value=True),
        patch("builtins.open", mock_open(read_data=json.dumps(mock_config))),
    ):
        dtype = trt_llm_hf.get_hf_model_dtype("/tmp/model")
        assert dtype == "float16"

    # Test with bfloat16
    mock_config = {"torch_dtype": "bfloat16"}

    with (
        patch("pathlib.Path.exists", return_value=True),
        patch("builtins.open", mock_open(read_data=json.dumps(mock_config))),
    ):
        dtype = trt_llm_hf.get_hf_model_dtype("/tmp/model")
        assert dtype == "bfloat16"


@pytest.mark.run_only_on("GPU")
@pytest.mark.unit
def test_tensorrt_llm_hf_get_hf_model_dtype_fp16_bf16_flags():
    """Test getting model dtype from HF config with fp16/bf16 flags."""
    try:
        import tensorrt_llm  # noqa: F401
    except ImportError:
        pytest.skip("Could not import TRTLLM helpers. tensorrt_llm is likely not installed")
        return

    from nemo_export.tensorrt_llm_hf import TensorRTLLMHF

    trt_llm_hf = TensorRTLLMHF(model_dir="/tmp/test_model", load_model=False)

    # Test with fp16 flag
    mock_config = {"fp16": True, "bf16": False}

    with (
        patch("pathlib.Path.exists", return_value=True),
        patch("builtins.open", mock_open(read_data=json.dumps(mock_config))),
    ):
        dtype = trt_llm_hf.get_hf_model_dtype("/tmp/model")
        assert dtype == "float16"

    # Test with bf16 flag
    mock_config = {"fp16": False, "bf16": True}

    with (
        patch("pathlib.Path.exists", return_value=True),
        patch("builtins.open", mock_open(read_data=json.dumps(mock_config))),
    ):
        dtype = trt_llm_hf.get_hf_model_dtype("/tmp/model")
        assert dtype == "bfloat16"


@pytest.mark.run_only_on("GPU")
@pytest.mark.unit
def test_tensorrt_llm_hf_get_hf_model_dtype_direct_dtype_field():
    """Test getting model dtype from HF config with direct dtype field."""
    try:
        import tensorrt_llm  # noqa: F401
    except ImportError:
        pytest.skip("Could not import TRTLLM helpers. tensorrt_llm is likely not installed")
        return

    from nemo_export.tensorrt_llm_hf import TensorRTLLMHF

    trt_llm_hf = TensorRTLLMHF(model_dir="/tmp/test_model", load_model=False)

    # Test with direct dtype field
    mock_config = {"dtype": "float32"}

    with (
        patch("pathlib.Path.exists", return_value=True),
        patch("builtins.open", mock_open(read_data=json.dumps(mock_config))),
    ):
        dtype = trt_llm_hf.get_hf_model_dtype("/tmp/model")
        assert dtype == "float32"


@pytest.mark.run_only_on("GPU")
@pytest.mark.unit
def test_tensorrt_llm_hf_get_hf_model_dtype_pretrained_config():
    """Test getting model dtype from HF config with pretrained_config field."""
    try:
        import tensorrt_llm  # noqa: F401
    except ImportError:
        pytest.skip("Could not import TRTLLM helpers. tensorrt_llm is likely not installed")
        return

    from nemo_export.tensorrt_llm_hf import TensorRTLLMHF

    trt_llm_hf = TensorRTLLMHF(model_dir="/tmp/test_model", load_model=False)

    # Test with pretrained_config field
    mock_config = {"pretrained_config": {"dtype": "float16"}}

    with (
        patch("pathlib.Path.exists", return_value=True),
        patch("builtins.open", mock_open(read_data=json.dumps(mock_config))),
    ):
        dtype = trt_llm_hf.get_hf_model_dtype("/tmp/model")
        assert dtype == "float16"


@pytest.mark.run_only_on("GPU")
@pytest.mark.unit
def test_tensorrt_llm_hf_get_hf_model_dtype_not_found():
    """Test getting model dtype when config file doesn't exist."""
    try:
        import tensorrt_llm  # noqa: F401
    except ImportError:
        pytest.skip("Could not import TRTLLM helpers. tensorrt_llm is likely not installed")
        return

    from nemo_export.tensorrt_llm_hf import TensorRTLLMHF

    trt_llm_hf = TensorRTLLMHF(model_dir="/tmp/test_model", load_model=False)

    with patch("pathlib.Path.exists", return_value=False):
        with pytest.raises(FileNotFoundError) as exc_info:
            trt_llm_hf.get_hf_model_dtype("/tmp/model")
        assert "Config file not found" in str(exc_info.value)


@pytest.mark.run_only_on("GPU")
@pytest.mark.unit
def test_tensorrt_llm_hf_get_hf_model_dtype_no_dtype():
    """Test getting model dtype when no dtype information is available."""
    try:
        import tensorrt_llm  # noqa: F401
    except ImportError:
        pytest.skip("Could not import TRTLLM helpers. tensorrt_llm is likely not installed")
        return

    from nemo_export.tensorrt_llm_hf import TensorRTLLMHF

    trt_llm_hf = TensorRTLLMHF(model_dir="/tmp/test_model", load_model=False)

    # Test with config that has no dtype information
    mock_config = {"model_type": "llama"}

    with (
        patch("pathlib.Path.exists", return_value=True),
        patch("builtins.open", mock_open(read_data=json.dumps(mock_config))),
    ):
        dtype = trt_llm_hf.get_hf_model_dtype("/tmp/model")
        assert dtype is None


@pytest.mark.run_only_on("GPU")
@pytest.mark.unit
def test_tensorrt_llm_hf_get_hf_model_dtype_invalid_json():
    """Test getting model dtype with invalid JSON in config file."""
    try:
        import tensorrt_llm  # noqa: F401
    except ImportError:
        pytest.skip("Could not import TRTLLM helpers. tensorrt_llm is likely not installed")
        return

    from nemo_export.tensorrt_llm_hf import TensorRTLLMHF

    trt_llm_hf = TensorRTLLMHF(model_dir="/tmp/test_model", load_model=False)

    with (
        patch("pathlib.Path.exists", return_value=True),
        patch("builtins.open", mock_open(read_data="invalid json {")),
    ):
        with pytest.raises(ValueError) as exc_info:
            trt_llm_hf.get_hf_model_dtype("/tmp/model")
        assert "Invalid JSON in config file" in str(exc_info.value)


@pytest.mark.run_only_on("GPU")
@pytest.mark.unit
def test_tensorrt_llm_hf_supported_models():
    """Test supported HF models mapping."""
    try:
        import tensorrt_llm  # noqa: F401
    except ImportError:
        pytest.skip("Could not import TRTLLM helpers. tensorrt_llm is likely not installed")
        return

    from nemo_export.tensorrt_llm_hf import TensorRTLLMHF

    model_dir = "/tmp/test_model_dir"
    trt_llm_hf = TensorRTLLMHF(model_dir=model_dir, load_model=False)

    # Test HF model mapping
    hf_mapping = trt_llm_hf.get_supported_hf_model_mapping
    assert isinstance(hf_mapping, dict)
    assert len(hf_mapping) > 0

    # Test specific model mappings
    expected_models = [
        "LlamaForCausalLM",
        "MistralForCausalLM",
        "GPT2LMHeadModel",
        "Phi3ForCausalLM",
        "QWenForCausalLM",
        "GEMMA",
        "FalconForCausalLM",
        "MambaForCausalLM",
    ]

    for model in expected_models:
        assert model in hf_mapping, f"Model {model} not found in supported HF models"

    # Verify all values are valid TensorRT-LLM model classes
    for key, value in hf_mapping.items():
        assert value is not None
        assert hasattr(value, "__name__")


@pytest.mark.run_only_on("GPU")
@pytest.mark.unit
def test_tensorrt_llm_hf_export_hf_model_unsupported_model():
    """Test exporting an unsupported HF model type."""
    try:
        import tensorrt_llm  # noqa: F401
    except ImportError:
        pytest.skip("Could not import TRTLLM helpers. tensorrt_llm is likely not installed")
        return

    from nemo_export.tensorrt_llm_hf import TensorRTLLMHF

    trt_llm_hf = TensorRTLLMHF(model_dir="/tmp/test_model", load_model=False)

    with (
        patch.object(trt_llm_hf, "get_hf_model_type", return_value="UnsupportedModel"),
        pytest.raises(ValueError) as exc_info,
    ):
        trt_llm_hf.export_hf_model(hf_model_path="/tmp/hf_model", model_type="UnsupportedModel")

    assert "is not currently a supported model type" in str(exc_info.value)


@pytest.mark.run_only_on("GPU")
@pytest.mark.unit
def test_tensorrt_llm_hf_export_hf_model_no_dtype():
    """Test exporting HF model when dtype cannot be determined."""
    try:
        import tensorrt_llm  # noqa: F401
    except ImportError:
        pytest.skip("Could not import TRTLLM helpers. tensorrt_llm is likely not installed")
        return

    from nemo_export.tensorrt_llm_hf import TensorRTLLMHF

    trt_llm_hf = TensorRTLLMHF(model_dir="/tmp/test_model", load_model=False)

    with (
        patch.object(trt_llm_hf, "get_hf_model_type", return_value="LlamaForCausalLM"),
        patch.object(trt_llm_hf, "get_hf_model_dtype", return_value=None),
        pytest.raises(ValueError) as exc_info,
    ):
        trt_llm_hf.export_hf_model(hf_model_path="/tmp/hf_model")

    assert "No dtype found in hf model config" in str(exc_info.value)


@pytest.mark.run_only_on("GPU")
@pytest.mark.unit
def test_tensorrt_llm_hf_export_hf_model_basic():
    """Test basic HF model export functionality."""
    try:
        import tensorrt_llm  # noqa: F401
    except ImportError:
        pytest.skip("Could not import TRTLLM helpers. tensorrt_llm is likely not installed")
        return

    from nemo_export.tensorrt_llm_hf import TensorRTLLMHF

    trt_llm_hf = TensorRTLLMHF(model_dir="/tmp/test_model", load_model=False)

    mock_model = MagicMock()
    mock_engine = MagicMock()

    with (
        patch.object(trt_llm_hf, "get_hf_model_type", return_value="LlamaForCausalLM"),
        patch.object(trt_llm_hf, "get_hf_model_dtype", return_value="float16"),
        patch("nemo_export.tensorrt_llm_hf.prepare_directory_for_export"),
        patch("nemo_export.tensorrt_llm_hf.build_trtllm", return_value=mock_engine),
        patch("nemo_export.tensorrt_llm_hf.LLaMAForCausalLM.from_hugging_face", return_value=mock_model),
        patch("glob.glob", return_value=[]),
        patch.object(trt_llm_hf, "_load"),
    ):
        trt_llm_hf.export_hf_model(
            hf_model_path="/tmp/hf_model",
            max_batch_size=8,
            tensor_parallelism_size=1,
            max_input_len=256,
            max_output_len=256,
        )

        # Verify engine was saved
        mock_engine.save.assert_called_once()


@pytest.mark.run_only_on("GPU")
@pytest.mark.unit
def test_tensorrt_llm_hf_export_hf_model_with_params():
    """Test HF model export with various parameters."""
    try:
        import tensorrt_llm  # noqa: F401
    except ImportError:
        pytest.skip("Could not import TRTLLM helpers. tensorrt_llm is likely not installed")
        return

    from nemo_export.tensorrt_llm_hf import TensorRTLLMHF

    trt_llm_hf = TensorRTLLMHF(model_dir="/tmp/test_model", load_model=False)

    mock_model = MagicMock()
    mock_engine = MagicMock()

    with (
        patch.object(trt_llm_hf, "get_hf_model_type", return_value="MistralForCausalLM"),
        patch.object(trt_llm_hf, "get_hf_model_dtype", return_value="bfloat16"),
        patch("nemo_export.tensorrt_llm_hf.prepare_directory_for_export"),
        patch("nemo_export.tensorrt_llm_hf.build_trtllm", return_value=mock_engine),
        patch("nemo_export.tensorrt_llm_hf.LLaMAForCausalLM.from_hugging_face", return_value=mock_model),
        patch("glob.glob", return_value=[]),
        patch.object(trt_llm_hf, "_load"),
    ):
        trt_llm_hf.export_hf_model(
            hf_model_path="/tmp/hf_model",
            max_batch_size=16,
            tensor_parallelism_size=2,
            max_input_len=512,
            max_output_len=512,
            dtype="bfloat16",
            gemm_plugin="auto",
            remove_input_padding=True,
            use_paged_context_fmha=True,
            paged_kv_cache=True,
            tokens_per_block=64,
            multiple_profiles=True,
            reduce_fusion=True,
            max_beam_width=4,
            use_refit=True,
        )


@pytest.mark.run_only_on("GPU")
@pytest.mark.unit
def test_tensorrt_llm_hf_export_hf_model_batch_size_adjustment():
    """Test HF model export with batch size < 4 gets adjusted to 4."""
    try:
        import tensorrt_llm  # noqa: F401
    except ImportError:
        pytest.skip("Could not import TRTLLM helpers. tensorrt_llm is likely not installed")
        return

    from nemo_export.tensorrt_llm_hf import TensorRTLLMHF

    trt_llm_hf = TensorRTLLMHF(model_dir="/tmp/test_model", load_model=False)

    mock_model = MagicMock()
    mock_engine = MagicMock()

    with (
        patch.object(trt_llm_hf, "get_hf_model_type", return_value="LlamaForCausalLM"),
        patch.object(trt_llm_hf, "get_hf_model_dtype", return_value="float16"),
        patch("nemo_export.tensorrt_llm_hf.prepare_directory_for_export"),
        patch("nemo_export.tensorrt_llm_hf.build_trtllm", return_value=mock_engine),
        patch("nemo_export.tensorrt_llm_hf.LLaMAForCausalLM.from_hugging_face", return_value=mock_model),
        patch("glob.glob", return_value=[]),
        patch.object(trt_llm_hf, "_load"),
        patch("builtins.print") as mock_print,
    ):
        trt_llm_hf.export_hf_model(
            hf_model_path="/tmp/hf_model",
            max_batch_size=2,  # Less than 4
        )

        # Verify warning was printed
        mock_print.assert_called_once()
        assert "Force set to 4" in str(mock_print.call_args)


@pytest.mark.run_only_on("GPU")
@pytest.mark.unit
def test_tensorrt_llm_hf_export_hf_model_multi_rank():
    """Test HF model export with multiple ranks (tensor parallelism)."""
    try:
        import tensorrt_llm  # noqa: F401
    except ImportError:
        pytest.skip("Could not import TRTLLM helpers. tensorrt_llm is likely not installed")
        return

    from nemo_export.tensorrt_llm_hf import TensorRTLLMHF

    trt_llm_hf = TensorRTLLMHF(model_dir="/tmp/test_model", load_model=False)

    mock_model = MagicMock()
    mock_engine = MagicMock()

    with (
        patch.object(trt_llm_hf, "get_hf_model_type", return_value="LlamaForCausalLM"),
        patch.object(trt_llm_hf, "get_hf_model_dtype", return_value="float16"),
        patch("nemo_export.tensorrt_llm_hf.prepare_directory_for_export"),
        patch("nemo_export.tensorrt_llm_hf.build_trtllm", return_value=mock_engine),
        patch("nemo_export.tensorrt_llm_hf.LLaMAForCausalLM.from_hugging_face", return_value=mock_model),
        patch("glob.glob", return_value=[]),
        patch.object(trt_llm_hf, "_load"),
    ):
        trt_llm_hf.export_hf_model(
            hf_model_path="/tmp/hf_model",
            tensor_parallelism_size=4,  # Test with 4 ranks
        )

        # Verify engine was saved 4 times (once per rank)
        assert mock_engine.save.call_count == 4


@pytest.mark.run_only_on("GPU")
@pytest.mark.unit
def test_tensorrt_llm_hf_export_copies_tokenizer_files():
    """Test that HF model export copies tokenizer files."""
    try:
        import tensorrt_llm  # noqa: F401
    except ImportError:
        pytest.skip("Could not import TRTLLM helpers. tensorrt_llm is likely not installed")
        return

    from nemo_export.tensorrt_llm_hf import TensorRTLLMHF

    trt_llm_hf = TensorRTLLMHF(model_dir="/tmp/test_model", load_model=False)

    mock_model = MagicMock()
    mock_engine = MagicMock()

    with (
        patch.object(trt_llm_hf, "get_hf_model_type", return_value="LlamaForCausalLM"),
        patch.object(trt_llm_hf, "get_hf_model_dtype", return_value="float16"),
        patch("nemo_export.tensorrt_llm_hf.prepare_directory_for_export"),
        patch("nemo_export.tensorrt_llm_hf.build_trtllm", return_value=mock_engine),
        patch("nemo_export.tensorrt_llm_hf.LLaMAForCausalLM.from_hugging_face", return_value=mock_model),
        patch(
            "glob.glob",
            side_effect=lambda x: (
                ["/tmp/hf_model/tokenizer.json"] if "*.json" in x else ["/tmp/hf_model/tokenizer.model"]
            ),
        ),
        patch("shutil.copy"),
        patch.object(trt_llm_hf, "_load"),
    ):
        trt_llm_hf.export_hf_model(hf_model_path="/tmp/hf_model")


@pytest.mark.run_only_on("GPU")
@pytest.mark.unit
def test_tensorrt_llm_hf_inherits_parent_methods():
    """Test that TensorRTLLMHF inherits methods from TensorRTLLM."""
    try:
        import tensorrt_llm  # noqa: F401
    except ImportError:
        pytest.skip("Could not import TRTLLM helpers. tensorrt_llm is likely not installed")
        return

    from nemo_export.tensorrt_llm_hf import TensorRTLLMHF

    trt_llm_hf = TensorRTLLMHF(model_dir="/tmp/test_model", load_model=False)

    # Verify inherited methods exist
    assert hasattr(trt_llm_hf, "forward")
    assert hasattr(trt_llm_hf, "_infer_fn")
    assert hasattr(trt_llm_hf, "ray_infer_fn")
    assert hasattr(trt_llm_hf, "unload_engine")
    assert hasattr(trt_llm_hf, "_load")
    assert hasattr(trt_llm_hf, "get_triton_input")
    assert hasattr(trt_llm_hf, "get_triton_output")
    assert hasattr(trt_llm_hf, "_pad_logits")


@pytest.mark.run_only_on("GPU")
@pytest.mark.unit
def test_tensorrt_llm_hf_unavailable_error():
    """Test that TensorRTLLMHF raises UnavailableError when TensorRT-LLM is not installed."""
    try:
        import tensorrt_llm  # noqa: F401

        pytest.skip("TensorRT-LLM is installed, skipping unavailable test")
    except ImportError:
        pass

    from nemo_export_deploy_common.import_utils import UnavailableError

    # Mock HAVE_TENSORRT_LLM to be False
    with patch("nemo_export.tensorrt_llm_hf.HAVE_TENSORRT_LLM", False):
        from nemo_export.tensorrt_llm_hf import TensorRTLLMHF

        with pytest.raises(UnavailableError):
            TensorRTLLMHF(model_dir="/tmp/test_model", load_model=False)
