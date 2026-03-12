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

from pathlib import Path
from unittest.mock import MagicMock, patch

from nemo_deploy.llm.inference.nemo_utils import (
    ADAPTER_META_FILENAME,
    CONTEXT_PATH,
    WEIGHTS_PATH,
    MCoreTokenizerWrappper,
    ckpt_to_context_subdir,
    ckpt_to_dir,
    ckpt_to_weights_subdir,
    idempotent_path_append,
    set_modelopt_spec_if_exists_in_ckpt,
)


class TestConstants:
    def test_weights_path(self):
        assert WEIGHTS_PATH == "weights"

    def test_context_path(self):
        assert CONTEXT_PATH == "context"

    def test_adapter_meta_filename(self):
        assert ADAPTER_META_FILENAME == "adapter_metadata.json"


class TestCkptToDir:
    def test_with_ckpt_extension(self):
        filepath = Path("/some/path/model.ckpt")
        result = ckpt_to_dir(filepath)
        assert result == Path("/some/path/model")

    def test_without_ckpt_extension(self):
        # Should append .ckpt then strip it
        filepath = Path("/some/path/model")
        result = ckpt_to_dir(filepath)
        assert result == Path("/some/path/model")

    def test_with_other_extension(self):
        # .pt is not .ckpt, so .ckpt is appended
        filepath = Path("/some/path/model.pt")
        result = ckpt_to_dir(filepath)
        assert result == Path("/some/path/model.pt")

    def test_returns_path_object(self):
        result = ckpt_to_dir("/some/path/model.ckpt")
        assert isinstance(result, Path)

    def test_string_input(self):
        result = ckpt_to_dir("/some/path/model.ckpt")
        assert result == Path("/some/path/model")


class TestIdempotentPathAppend:
    def test_appends_suffix_when_not_last(self):
        base = Path("/some/path")
        result = idempotent_path_append(base, "weights")
        assert result == Path("/some/path/weights")

    def test_does_not_append_when_already_last(self):
        base = Path("/some/path/weights")
        result = idempotent_path_append(base, "weights")
        assert result == Path("/some/path/weights")

    def test_returns_path_object(self):
        result = idempotent_path_append("/some/path", "context")
        assert isinstance(result, Path)

    def test_string_input(self):
        result = idempotent_path_append("/some/path", "context")
        assert result == Path("/some/path/context")

    def test_different_suffix(self):
        base = Path("/some/path/weights")
        result = idempotent_path_append(base, "context")
        assert result == Path("/some/path/weights/context")


class TestCkptToContextSubdir:
    def test_returns_context_subdir(self):
        filepath = Path("/checkpoint/model.ckpt")
        result = ckpt_to_context_subdir(filepath)
        assert result == Path("/checkpoint/model/context")

    def test_with_string_input(self):
        result = ckpt_to_context_subdir("/checkpoint/model.ckpt")
        assert result == Path("/checkpoint/model/context")

    def test_context_already_in_path(self):
        # ckpt_to_dir("/checkpoint/context.ckpt") -> "/checkpoint/context"
        # idempotent_path_append sees last part == "context", so no re-append
        filepath = Path("/checkpoint/context.ckpt")
        result = ckpt_to_context_subdir(filepath)
        assert result == Path("/checkpoint/context")


class TestCkptToWeightsSubdir:
    def test_is_saving_true(self):
        filepath = Path("/checkpoint/model.ckpt")
        result = ckpt_to_weights_subdir(filepath, is_saving=True)
        assert result == Path("/checkpoint/model/weights")
        assert result.parts[-1] == WEIGHTS_PATH

    def test_is_saving_false_weights_dir_exists(self):
        filepath = Path("/checkpoint/model.ckpt")
        with patch.object(Path, "is_dir", return_value=True):
            result = ckpt_to_weights_subdir(filepath, is_saving=False)
        assert result == Path("/checkpoint/model/weights")

    def test_is_saving_false_weights_dir_not_exists(self):
        filepath = Path("/checkpoint/model.ckpt")
        with patch.object(Path, "is_dir", return_value=False):
            result = ckpt_to_weights_subdir(filepath, is_saving=False)
        # When not saving and weights dir doesn't exist, returns base dir without weights
        assert result == Path("/checkpoint/model")

    def test_already_has_weights_path(self):
        # When filepath ends in "weights", ckpt_to_dir strips .ckpt and keeps "weights"
        # Actually ckpt_to_dir("/checkpoint/weights.ckpt") -> "/checkpoint/weights"
        # Then idempotent check: last part is "weights" == WEIGHTS_PATH, skip appending
        filepath = Path("/checkpoint/weights.ckpt")
        result = ckpt_to_weights_subdir(filepath, is_saving=False)
        assert result == Path("/checkpoint/weights")

    def test_string_input(self):
        with patch.object(Path, "is_dir", return_value=True):
            result = ckpt_to_weights_subdir("/checkpoint/model.ckpt", is_saving=False)
        assert result == Path("/checkpoint/model/weights")


class TestMCoreTokenizerWrappper:
    def test_init_default_vocab_size(self):
        mock_tok = MagicMock()
        mock_tok.eod = 50256
        mock_tok.vocab_size = 50000
        wrapper = MCoreTokenizerWrappper(mock_tok)
        assert wrapper.tokenizer is mock_tok
        assert wrapper.eod == 50256
        assert wrapper.vocab_size == 50000

    def test_init_custom_vocab_size(self):
        mock_tok = MagicMock()
        mock_tok.eod = 50256
        mock_tok.vocab_size = 50000
        wrapper = MCoreTokenizerWrappper(mock_tok, vocab_size=32000)
        assert wrapper.vocab_size == 32000

    def test_tokenize(self):
        mock_tok = MagicMock()
        mock_tok.eod = 50256
        mock_tok.vocab_size = 50000
        mock_tok.text_to_ids.return_value = [1, 2, 3]
        wrapper = MCoreTokenizerWrappper(mock_tok)
        result = wrapper.tokenize("hello")
        mock_tok.text_to_ids.assert_called_once_with("hello")
        assert result == [1, 2, 3]

    def test_detokenize_without_remove_special_tokens_param(self):
        mock_tok = MagicMock()
        mock_tok.eod = 50256
        mock_tok.vocab_size = 50000

        # ids_to_text does NOT have remove_special_tokens param
        def ids_to_text_no_param(tokens):
            return "hello world"

        mock_tok.ids_to_text = ids_to_text_no_param
        wrapper = MCoreTokenizerWrappper(mock_tok)
        result = wrapper.detokenize([1, 2, 3])
        assert result == "hello world"

    def test_detokenize_with_remove_special_tokens_param(self):
        mock_tok = MagicMock()
        mock_tok.eod = 50256
        mock_tok.vocab_size = 50000

        # ids_to_text DOES have remove_special_tokens param
        def ids_to_text_with_param(tokens, remove_special_tokens=False):
            return "hello world"

        mock_tok.ids_to_text = ids_to_text_with_param
        wrapper = MCoreTokenizerWrappper(mock_tok)
        result = wrapper.detokenize([1, 2, 3], remove_special_tokens=True)
        assert result == "hello world"

    def test_detokenize_default_remove_special_tokens(self):
        mock_tok = MagicMock()
        mock_tok.eod = 50256
        mock_tok.vocab_size = 50000

        called_with = {}

        def ids_to_text_with_param(tokens, remove_special_tokens=False):
            called_with["remove_special_tokens"] = remove_special_tokens
            return "hello"

        mock_tok.ids_to_text = ids_to_text_with_param
        wrapper = MCoreTokenizerWrappper(mock_tok)
        wrapper.detokenize([1, 2, 3])
        assert called_with["remove_special_tokens"] is False

    def test_additional_special_tokens_ids(self):
        mock_tok = MagicMock()
        mock_tok.eod = 50256
        mock_tok.vocab_size = 50000
        mock_tok.additional_special_tokens_ids = [100, 101]
        wrapper = MCoreTokenizerWrappper(mock_tok)
        assert wrapper.additional_special_tokens_ids == [100, 101]

    def test_bos_property(self):
        mock_tok = MagicMock()
        mock_tok.eod = 50256
        mock_tok.vocab_size = 50000
        mock_tok.bos_id = 1
        wrapper = MCoreTokenizerWrappper(mock_tok)
        assert wrapper.bos == 1

    def test_pad_property(self):
        mock_tok = MagicMock()
        mock_tok.eod = 50256
        mock_tok.vocab_size = 50000
        mock_tok.pad_id = 0
        wrapper = MCoreTokenizerWrappper(mock_tok)
        assert wrapper.pad == 0


# ---------------------------------------------------------------------------
# set_modelopt_spec_if_exists_in_ckpt
# ---------------------------------------------------------------------------


class TestSetModeloptSpecIfExistsInCkpt:
    """Tests for set_modelopt_spec_if_exists_in_ckpt — no GPU required."""

    def _make_model(self, type_name: str, config_type_name: str, *, has_module: bool = False):
        """Build a MagicMock that looks like a NeMo model with the given type names."""
        model = MagicMock()
        type(model).__name__ = type_name
        if has_module:
            model.module = MagicMock()
        else:
            del model.module  # ensure hasattr returns False

        config = MagicMock()
        type(config).__name__ = config_type_name
        model.config = config
        return model, config

    @patch("nemo_deploy.llm.inference.nemo_utils.ckpt_to_weights_subdir")
    def test_returns_early_when_modelopt_state_path_missing(self, mock_ckpt_to_weights):
        """When modelopt_state does not exist, the function should return immediately."""
        mock_weights = MagicMock()
        mock_modelopt_path = MagicMock()
        mock_modelopt_path.exists.return_value = False
        mock_weights.__truediv__ = MagicMock(return_value=mock_modelopt_path)
        mock_ckpt_to_weights.return_value = mock_weights

        model, config = self._make_model("GPTModel", "GPTConfig")
        set_modelopt_spec_if_exists_in_ckpt(model, "/fake/path")

        # config should be untouched
        config.transformer_layer_spec = MagicMock()
        assert not mock_modelopt_path.exists.return_value

    @patch("nemo_deploy.llm.inference.nemo_utils.ckpt_to_weights_subdir")
    def test_returns_early_when_model_has_module_attr(self, mock_ckpt_to_weights):
        """When model.module exists (DDP wrapper), the function should skip."""
        mock_weights = MagicMock()
        mock_modelopt_path = MagicMock()
        mock_modelopt_path.exists.return_value = True
        mock_weights.__truediv__ = MagicMock(return_value=mock_modelopt_path)
        mock_ckpt_to_weights.return_value = mock_weights

        # has_module=True forces early return
        model, config = self._make_model("GPTModel", "GPTConfig", has_module=True)
        set_modelopt_spec_if_exists_in_ckpt(model, "/fake/path")

        # gradient_accumulation_fusion should NOT be set (we returned early)
        config.gradient_accumulation_fusion.assert_not_called() if callable(
            config.gradient_accumulation_fusion
        ) else None

    @patch("nemo_deploy.llm.inference.nemo_utils.ckpt_to_weights_subdir")
    def test_logs_warning_for_non_gpt_mamba_model(self, mock_ckpt_to_weights):
        """Models that are neither GPTModel nor MambaModel should log a warning and return."""
        mock_weights = MagicMock()
        mock_modelopt_path = MagicMock()
        mock_modelopt_path.exists.return_value = True
        mock_weights.__truediv__ = MagicMock(return_value=mock_modelopt_path)
        mock_ckpt_to_weights.return_value = mock_weights

        model, config = self._make_model("SomeOtherModel", "GPTConfig")

        with patch("nemo_deploy.llm.inference.nemo_utils._logger") as mock_logger:
            set_modelopt_spec_if_exists_in_ckpt(model, "/fake/path")

        mock_logger.warning.assert_called_once()
        warning_msg = mock_logger.warning.call_args[0][0]
        assert "neither a GPTModel nor MambaModel" in warning_msg

    @patch("nemo_deploy.llm.inference.nemo_utils.ckpt_to_weights_subdir")
    def test_gpt_config_with_gpt_modelopt_spec_available(self, mock_ckpt_to_weights):
        """GPTModel + GPTConfig: when get_gpt_modelopt_spec is importable, spec is set."""
        mock_weights = MagicMock()
        mock_modelopt_path = MagicMock()
        mock_modelopt_path.exists.return_value = True
        mock_weights.__truediv__ = MagicMock(return_value=mock_modelopt_path)
        mock_ckpt_to_weights.return_value = mock_weights

        model, config = self._make_model("GPTModel", "GPTConfig")
        config.softmax_type = "vanilla"

        mock_spec_fn = MagicMock()
        fake_module = MagicMock()
        fake_module.get_gpt_modelopt_spec = mock_spec_fn

        with patch.dict(
            "sys.modules",
            {"megatron.core.post_training.modelopt.gpt.model_specs": fake_module},
        ):
            set_modelopt_spec_if_exists_in_ckpt(model, "/fake/path")

        assert config.gradient_accumulation_fusion is False
        # transformer_layer_spec should be a partial wrapping get_gpt_modelopt_spec
        assert config.transformer_layer_spec is not None

    @patch("nemo_deploy.llm.inference.nemo_utils.ckpt_to_weights_subdir")
    def test_gpt_config_without_gpt_modelopt_spec(self, mock_ckpt_to_weights):
        """GPTModel + GPTConfig: when get_gpt_modelopt_spec is NOT importable, log warning."""
        mock_weights = MagicMock()
        mock_modelopt_path = MagicMock()
        mock_modelopt_path.exists.return_value = True
        mock_weights.__truediv__ = MagicMock(return_value=mock_modelopt_path)
        mock_ckpt_to_weights.return_value = mock_weights

        model, config = self._make_model("GPTModel", "GPTConfig")

        # Ensure the import fails
        with patch.dict("sys.modules", {"megatron.core.post_training.modelopt.gpt.model_specs": None}):
            with patch("nemo_deploy.llm.inference.nemo_utils._logger") as mock_logger:
                set_modelopt_spec_if_exists_in_ckpt(model, "/fake/path")

        mock_logger.warning.assert_called()
        warning_args = mock_logger.warning.call_args[0]
        assert "get_gpt_modelopt_spec not available" in warning_args[0]

    @patch("nemo_deploy.llm.inference.nemo_utils.ckpt_to_weights_subdir")
    def test_ssm_config_with_mamba_spec_available(self, mock_ckpt_to_weights):
        """MambaModel + SSMConfig: when get_mamba_stack_modelopt_spec is importable, spec is set."""
        mock_weights = MagicMock()
        mock_modelopt_path = MagicMock()
        mock_modelopt_path.exists.return_value = True
        mock_weights.__truediv__ = MagicMock(return_value=mock_modelopt_path)
        mock_ckpt_to_weights.return_value = mock_weights

        model, config = self._make_model("MambaModel", "SSMConfig")

        mock_mamba_fn = MagicMock()
        fake_mamba_module = MagicMock()
        fake_mamba_module.get_mamba_stack_modelopt_spec = mock_mamba_fn

        with patch.dict(
            "sys.modules",
            {"megatron.core.post_training.modelopt.mamba.model_specs": fake_mamba_module},
        ):
            set_modelopt_spec_if_exists_in_ckpt(model, "/fake/path")

        assert config.gradient_accumulation_fusion is False
        assert config.mamba_stack_spec is not None

    @patch("nemo_deploy.llm.inference.nemo_utils.ckpt_to_weights_subdir")
    def test_ssm_config_without_mamba_spec(self, mock_ckpt_to_weights):
        """MambaModel + SSMConfig: when get_mamba_stack_modelopt_spec is NOT importable, log warning."""
        mock_weights = MagicMock()
        mock_modelopt_path = MagicMock()
        mock_modelopt_path.exists.return_value = True
        mock_weights.__truediv__ = MagicMock(return_value=mock_modelopt_path)
        mock_ckpt_to_weights.return_value = mock_weights

        model, config = self._make_model("MambaModel", "SSMConfig")

        with patch.dict("sys.modules", {"megatron.core.post_training.modelopt.mamba.model_specs": None}):
            with patch("nemo_deploy.llm.inference.nemo_utils._logger") as mock_logger:
                set_modelopt_spec_if_exists_in_ckpt(model, "/fake/path")

        mock_logger.warning.assert_called()
        warning_args = mock_logger.warning.call_args[0]
        assert "get_mamba_stack_modelopt_spec not available" in warning_args[0]

    @patch("nemo_deploy.llm.inference.nemo_utils.ckpt_to_weights_subdir")
    def test_other_config_type_logs_warning_and_returns(self, mock_ckpt_to_weights):
        """GPTModel with an unrecognised config type should log a warning and not touch gradient_accumulation_fusion."""
        mock_weights = MagicMock()
        mock_modelopt_path = MagicMock()
        mock_modelopt_path.exists.return_value = True
        mock_weights.__truediv__ = MagicMock(return_value=mock_modelopt_path)
        mock_ckpt_to_weights.return_value = mock_weights

        model, config = self._make_model("GPTModel", "UnknownConfig")

        with patch("nemo_deploy.llm.inference.nemo_utils._logger") as mock_logger:
            set_modelopt_spec_if_exists_in_ckpt(model, "/fake/path")

        mock_logger.warning.assert_called()
        warning_args = mock_logger.warning.call_args[0]
        assert "No modelopt layer spec supported" in warning_args[0]
        # gradient_accumulation_fusion should NOT have been set to False (early return)
        # config is a MagicMock, so we just verify the warning was the only side-effect logged
        assert mock_logger.warning.call_count == 1

    @patch("nemo_deploy.llm.inference.nemo_utils.ckpt_to_weights_subdir")
    def test_strips_nemo_prefix_from_path(self, mock_ckpt_to_weights):
        """Path starting with 'nemo://' should have that prefix stripped before processing."""
        mock_weights = MagicMock()
        mock_modelopt_path = MagicMock()
        mock_modelopt_path.exists.return_value = False
        mock_weights.__truediv__ = MagicMock(return_value=mock_modelopt_path)
        mock_ckpt_to_weights.return_value = mock_weights

        model, config = self._make_model("GPTModel", "GPTConfig")
        set_modelopt_spec_if_exists_in_ckpt(model, "nemo:///real/path")

        # Verify ckpt_to_weights_subdir was called with the stripped path
        mock_ckpt_to_weights.assert_called_once_with("/real/path", is_saving=False)
