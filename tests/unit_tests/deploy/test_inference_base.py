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

import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch
from megatron.core.inference.engines.mcore_engine import MCoreEngine
from megatron.core.inference.model_inference_wrappers.gpt.gpt_inference_wrapper import (
    GPTInferenceWrapper,
)
from megatron.core.transformer.module import MegatronModule
from nemo.collections.llm.gpt.model.base import GPTConfig
from nemo.collections.llm.inference.base import MCoreTokenizerWrappper

from nemo_deploy.llm.inference.inference_base import (
    MCoreEngineWithCleanup,
    _load_dist_shards_into_model,
    cleanup_distributed,
    create_mcore_engine,
    initialize_megatron_for_inference,
    load_nemo_checkpoint_to_tron_model,
    peel,
    setup_megatron_model_and_tokenizer_for_inference,
    setup_model_and_tokenizer_for_inference,
)
from nemo_deploy.llm.inference.tron_utils import DistributedInitConfig, RNGConfig
from nemo_export_deploy_common.import_utils import UnavailableError


@pytest.mark.run_only_on("GPU")
class TestInferenceBase(unittest.TestCase):
    def setUp(self):
        # Mock common objects needed for tests
        self.mock_model = MagicMock(spec=MegatronModule)
        self.mock_model_list = [self.mock_model]
        self.mock_path = Path("/fake/checkpoint/path")
        self.mock_weights_dir = Path("/fake/weights/dir")

        # Create a more complete mock tokenizer with required attributes
        self.mock_tokenizer = MagicMock(spec=MCoreTokenizerWrappper)
        self.mock_tokenizer.vocab_size = 50000
        self.mock_tokenizer.eod = 50256  # End of document token ID
        self.mock_tokenizer.pad = 50257  # Padding token ID

        # Setup model config
        self.model_config = GPTConfig(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            context_parallel_size=1,
            expert_model_parallel_size=1,
            vocab_size=50000,
            hidden_size=768,
            num_attention_heads=12,
            num_layers=12,
            params_dtype=torch.float16,
        )

        # Setup distributed and RNG configs
        self.dist_config = DistributedInitConfig(distributed_backend="nccl")
        self.rng_config = RNGConfig(inference_rng_tracker=True)

    @patch("nemo_deploy.llm.inference.inference_base.dist_ckpt.load")
    @patch("nemo_deploy.llm.inference.inference_base.get_default_load_sharded_strategy")
    @patch("megatron.core.transformer.module.MegatronModule.sharded_state_dict")
    def test_load_dist_shards_into_model_single_model(self, mock_sharded_state_dict, mock_get_strategy, mock_load):
        # Setup mocks
        mock_sharded_state_dict.return_value = {"fake_key": "fake_value"}
        mock_get_strategy.return_value = "fake_strategy"

        # Call the function
        _load_dist_shards_into_model(self.mock_model_list, self.mock_weights_dir)

        # Verify calls
        mock_sharded_state_dict.assert_called_once_with(self.mock_model)
        mock_get_strategy.assert_called_once_with(str(self.mock_weights_dir))
        mock_load.assert_called_once()
        self.mock_model.load_state_dict.assert_called_once()

    @patch("nemo_deploy.llm.inference.inference_base.dist_ckpt.load")
    @patch("nemo_deploy.llm.inference.inference_base.get_default_load_sharded_strategy")
    @patch("megatron.core.transformer.module.MegatronModule.sharded_state_dict")
    def test_load_dist_shards_into_model_multiple_models(self, mock_sharded_state_dict, mock_get_strategy, mock_load):
        # Setup multiple models
        mock_model1 = MagicMock(spec=MegatronModule)
        mock_model2 = MagicMock(spec=MegatronModule)
        mock_model_list = [mock_model1, mock_model2]
        mock_sharded_state_dict.side_effect = [
            {"fake_key1": "fake_value1"},
            {"fake_key2": "fake_value2"},
        ]

        # Call the function
        _load_dist_shards_into_model(mock_model_list, self.mock_weights_dir)

        # Verify calls
        self.assertEqual(mock_sharded_state_dict.call_count, 2)
        mock_get_strategy.assert_called_once_with(str(self.mock_weights_dir))
        mock_load.assert_called_once()
        mock_model1.load_state_dict.assert_called_once()
        mock_model2.load_state_dict.assert_called_once()

    @patch("torch.distributed.is_initialized")
    @patch("torch.distributed.destroy_process_group")
    def test_cleanup_distributed_initialized(self, mock_destroy, mock_is_initialized):
        # Setup mock
        mock_is_initialized.return_value = True

        # Call the function
        cleanup_distributed()

        # Verify calls
        mock_is_initialized.assert_called_once()
        mock_destroy.assert_called_once()

    @patch("torch.distributed.is_initialized")
    @patch("torch.distributed.destroy_process_group")
    def test_cleanup_distributed_not_initialized(self, mock_destroy, mock_is_initialized):
        # Setup mock
        mock_is_initialized.return_value = False

        # Call the function
        cleanup_distributed()

        # Verify calls
        mock_is_initialized.assert_called_once()
        mock_destroy.assert_not_called()

    @patch("nemo_deploy.llm.inference.inference_base.initialize_distributed")
    @patch("nemo_deploy.llm.inference.inference_base._set_random_seed")
    @patch("nemo_deploy.llm.inference.inference_base._initialize_tp_communicators")
    def test_initialize_megatron_for_inference(self, mock_tp_comm, mock_seed, mock_init_dist):
        # Setup mocks
        self.model_config.tp_comm_overlap = True
        micro_batch_size = 4

        # Call the function
        initialize_megatron_for_inference(self.model_config, self.dist_config, self.rng_config, micro_batch_size)

        # Verify calls
        mock_init_dist.assert_called_once()
        mock_seed.assert_called_once()
        mock_tp_comm.assert_called_once_with(self.model_config, micro_batch_size)

    def test_peel_unwrapped_module(self):
        # Setup a simple module
        module = torch.nn.Linear(10, 10)

        # Call the function
        result = peel(module)

        # Verify the result
        self.assertEqual(result, module)

    def test_peel_wrapped_module(self):
        # Setup a wrapped module (nested)
        inner_module = torch.nn.Linear(10, 10)
        middle_wrapper = MagicMock()
        middle_wrapper.module = inner_module
        outer_wrapper = MagicMock()
        outer_wrapper.module = middle_wrapper

        # Call the function
        result = peel(outer_wrapper)

        # Verify the result
        self.assertEqual(result, inner_module)

    @patch("nemo_deploy.llm.inference.inference_base._load_dist_shards_into_model")
    @patch("nemo_deploy.llm.inference.inference_base.ckpt_to_weights_subdir")
    def test_load_nemo_checkpoint_to_tron_model(self, mock_ckpt_to_weights, mock_load_shards):
        # Setup mocks
        mock_ckpt_to_weights.return_value = self.mock_weights_dir

        # Call the function
        load_nemo_checkpoint_to_tron_model(self.mock_model_list, self.mock_path)

        # Verify calls
        mock_ckpt_to_weights.assert_called_once_with(self.mock_path, is_saving=False)
        mock_load_shards.assert_called_once_with(self.mock_model_list, self.mock_weights_dir, False)

    @patch("nemo_deploy.llm.inference.inference_base.HAVE_NEMO", True)
    @patch("nemo_deploy.llm.inference.inference_base.set_modelopt_spec_if_exists_in_ckpt")
    @patch("nemo_deploy.llm.inference.inference_base.torch_distributed_init")
    @patch("nemo_deploy.llm.inference.inference_base.io.load_context")
    @patch("nemo_deploy.llm.inference.inference_base.check_is_distributed_checkpoint")
    @patch("nemo_deploy.llm.inference.inference_base.ckpt_to_weights_subdir")
    @patch("nemo_deploy.llm.inference.inference_base.ckpt_to_context_subdir")
    @patch("nemo_deploy.llm.inference.inference_base.initialize_megatron_for_inference")
    @patch("nemo_deploy.llm.inference.inference_base.get_model_from_config")
    @patch("nemo_deploy.llm.inference.inference_base.load_nemo_checkpoint_to_tron_model")
    @patch("nemo_deploy.llm.inference.inference_base.peel")
    @patch("nemo_deploy.llm.inference.inference_base.MCoreTokenizerWrappper")
    def test_setup_model_and_tokenizer_for_inference(
        self,
        mock_tokenizer_wrapper,
        mock_peel,
        mock_load_ckpt,
        mock_get_model,
        mock_init_megatron,
        mock_context_subdir,
        mock_weights_subdir,
        mock_check_dist,
        mock_load_context,
        mock_torch_dist_init,
        mock_set_modelopt,
    ):
        # Setup mocks
        mock_context = MagicMock()
        mock_context.config = self.model_config
        mock_context.tokenizer = self.mock_tokenizer
        mock_load_context.return_value = mock_context
        mock_check_dist.return_value = True
        mock_get_model.return_value = self.mock_model_list
        mock_peel.return_value = self.mock_model

        # Create a mock tokenizer wrapper that will be returned
        mock_tokenizer_instance = MagicMock()
        mock_tokenizer_wrapper.return_value = mock_tokenizer_instance

        # Call the function
        result = setup_model_and_tokenizer_for_inference(
            checkpoint_path=self.mock_path,
            tensor_model_parallel_size=2,
            enable_flash_decode=True,
        )

        # Verify calls and result
        self.assertEqual(len(result), 2)
        mock_check_dist.assert_called_once()
        mock_init_megatron.assert_called_once()
        mock_get_model.assert_called_once()
        mock_load_ckpt.assert_called_once()
        mock_peel.assert_called_once()
        mock_tokenizer_wrapper.assert_called_once()
        mock_torch_dist_init.assert_called_once()
        mock_set_modelopt.assert_called_once()

    @patch("nemo_deploy.llm.inference.inference_base.HAVE_NEMO", True)
    @patch("nemo_deploy.llm.inference.inference_base.set_modelopt_spec_if_exists_in_ckpt")
    @patch("nemo_deploy.llm.inference.inference_base.torch_distributed_init")
    @patch("nemo_deploy.llm.inference.inference_base.io.load_context")
    @patch("nemo_deploy.llm.inference.inference_base.check_is_distributed_checkpoint")
    @patch("nemo_deploy.llm.inference.inference_base.ckpt_to_weights_subdir")
    @patch("nemo_deploy.llm.inference.inference_base.ckpt_to_context_subdir")
    @patch("nemo_deploy.llm.inference.inference_base.initialize_megatron_for_inference")
    @patch("nemo_deploy.llm.inference.inference_base.get_model_from_config")
    @patch("nemo_deploy.llm.inference.inference_base.load_nemo_checkpoint_to_tron_model")
    def test_setup_model_calls_configure_model(
        self,
        mock_load_ckpt,
        mock_get_model,
        mock_init_megatron,
        mock_context_subdir,
        mock_weights_subdir,
        mock_check_dist,
        mock_load_context,
        mock_torch_dist_init,
        mock_set_modelopt,
    ):
        # Ensure distributed checkpoint path
        mock_check_dist.return_value = True
        # Context with tokenizer
        mock_context = MagicMock()
        mock_context.config = self.model_config
        mock_context.tokenizer = self.mock_tokenizer
        mock_load_context.return_value = mock_context

        # Model list with a module having configure_model
        self.mock_model.configure_model = MagicMock()
        mock_get_model.return_value = self.mock_model_list

        # Call the function under test
        from nemo_deploy.llm.inference.inference_base import setup_model_and_tokenizer_for_inference

        setup_model_and_tokenizer_for_inference(checkpoint_path=self.mock_path)

        # Verify that configure_model(tokenizer) was invoked
        self.mock_model.configure_model.assert_called_once_with(self.mock_tokenizer)

    @patch("nemo_deploy.llm.inference.inference_base.HAVE_NEMO", True)
    @patch("nemo_deploy.llm.inference.inference_base.calculate_padded_vocab_size")
    @patch("nemo_deploy.llm.inference.inference_base.GPTInferenceWrapper")
    @patch("nemo_deploy.llm.inference.inference_base.TextGenerationController")
    @patch("nemo_deploy.llm.inference.inference_base.MCoreEngine")
    @patch("nemo_deploy.llm.inference.inference_base.StaticInferenceContext")
    @patch("nemo_deploy.llm.inference.inference_base.setup_megatron_model_and_tokenizer_for_inference")
    def test_create_mcore_engine_megatron_with_mlm_args(
        self,
        mock_setup_meg,
        mock_static_ctx,
        mock_engine_class,
        mock_tg_ctrl_class,
        mock_gpt_wrapper_class,
        mock_calc_pad_vocab,
    ):
        # Prepare model.config
        mock_model = MagicMock()
        mock_model.config = MagicMock()
        mock_model.config.hidden_size = 256
        mock_model.config.vocab_size = 32000
        mock_model.config.make_vocab_size_divisible_by = 128
        mock_model.config.tensor_model_parallel_size = 1

        mock_tokenizer = MagicMock()

        # mlm_args with explicit padded_vocab_size
        mlm_args = MagicMock()
        mlm_args.padded_vocab_size = 1234

        mock_setup_meg.return_value = ([mock_model], mock_tokenizer, mlm_args)
        mock_static_ctx.from_config.return_value = MagicMock()

        from nemo_deploy.llm.inference.inference_base import create_mcore_engine

        create_mcore_engine(path=self.mock_path, model_format="megatron")

        # Ensure we did NOT compute padded vocab when mlm_args provides it
        mock_calc_pad_vocab.assert_not_called()

        # Validate GPTInferenceWrapper config
        mock_gpt_wrapper_class.assert_called_once_with(mock_model, mock_static_ctx.return_value)
        mock_static_ctx.assert_called_once_with(8, 4096)

    @patch("nemo_deploy.llm.inference.inference_base.HAVE_NEMO", True)
    @patch("nemo_deploy.llm.inference.inference_base.calculate_padded_vocab_size")
    @patch("nemo_deploy.llm.inference.inference_base.GPTInferenceWrapper")
    @patch("nemo_deploy.llm.inference.inference_base.TextGenerationController")
    @patch("nemo_deploy.llm.inference.inference_base.MCoreEngine")
    @patch("nemo_deploy.llm.inference.inference_base.StaticInferenceContext")
    @patch("nemo_deploy.llm.inference.inference_base.setup_megatron_model_and_tokenizer_for_inference")
    def test_create_mcore_engine_megatron_without_mlm_args_uses_calculated_padded_vocab(
        self,
        mock_setup_meg,
        mock_static_ctx,
        mock_engine_class,
        mock_tg_ctrl_class,
        mock_gpt_wrapper_class,
        mock_calc_pad_vocab,
    ):
        # Prepare model.config
        mock_model = MagicMock()
        mock_model.config = MagicMock()
        mock_model.config.hidden_size = 512
        mock_model.config.vocab_size = 30000
        mock_model.config.make_vocab_size_divisible_by = 128
        mock_model.config.tensor_model_parallel_size = 2

        mock_tokenizer = MagicMock()

        mock_setup_meg.return_value = ([mock_model], mock_tokenizer, None)
        mock_static_ctx.from_config.return_value = MagicMock()
        mock_calc_pad_vocab.return_value = 24576

        from nemo_deploy.llm.inference.inference_base import create_mcore_engine

        create_mcore_engine(path=self.mock_path, model_format="megatron")

        # Ensure padded vocab was computed with expected args
        mock_calc_pad_vocab.assert_called_once_with(
            mock_model.config.vocab_size,
            mock_model.config.make_vocab_size_divisible_by,
            mock_model.config.tensor_model_parallel_size,
        )

        # Validate GPTInferenceWrapper config
        mock_gpt_wrapper_class.assert_called_once_with(mock_model, mock_static_ctx.return_value)
        mock_static_ctx.assert_called_once_with(8, 4096)

    @patch("nemo_deploy.llm.inference.inference_base.check_is_distributed_checkpoint")
    @patch("nemo_deploy.llm.inference.inference_base.ckpt_to_weights_subdir")
    @patch("nemo_deploy.llm.inference.inference_base.ckpt_to_context_subdir")
    @patch("nemo_deploy.llm.inference.inference_base.io.load_context")
    def test_setup_model_and_tokenizer_not_dist_ckpt(
        self,
        mock_load_context,
        mock_context_subdir,
        mock_weights_subdir,
        mock_check_dist,
    ):
        # Setup mocks
        mock_check_dist.return_value = False
        mock_weights_subdir.return_value = "/fake/weights/path"
        mock_context_subdir.return_value = "/fake/context/path"

        # In the actual implementation, load_context is called before the distributed checkpoint check
        # so we need to return a valid object here
        mock_context = MagicMock()
        mock_context.config = self.model_config
        mock_context.tokenizer = self.mock_tokenizer
        mock_load_context.return_value = mock_context

        # Call the function and expect exception
        with self.assertRaises(ValueError):
            setup_model_and_tokenizer_for_inference(checkpoint_path=self.mock_path)

        # Verify mocks were called
        mock_context_subdir.assert_called_once()
        mock_load_context.assert_called_once()
        mock_weights_subdir.assert_called_once()
        mock_check_dist.assert_called_once()

    def test_mcore_engine_with_cleanup(self):
        # Create mocks for the engine and wrapper
        mock_engine = MagicMock(spec=MCoreEngine)
        mock_wrapper = MagicMock(spec=GPTInferenceWrapper)

        # Create the wrapper
        engine_wrapper = MCoreEngineWithCleanup(mock_engine, mock_wrapper, self.mock_tokenizer)

        # Test attribute delegation - mock the attribute access directly instead of using __getattr__
        # Define the attribute directly on the mock
        mock_engine.some_attribute = "attribute_value"
        attribute_value = engine_wrapper.some_attribute
        self.assertEqual(attribute_value, "attribute_value")

        # Test method delegation - create a method on the mock
        mock_engine.some_method = MagicMock(return_value="method_result")
        result = engine_wrapper.some_method()
        self.assertEqual(result, "method_result")
        mock_engine.some_method.assert_called_once()

    @patch("nemo_deploy.llm.inference.inference_base.cleanup_distributed")
    def test_mcore_engine_with_cleanup_del(self, mock_cleanup):
        # Create mocks
        mock_engine = MagicMock(spec=MCoreEngine)
        mock_wrapper = MagicMock(spec=GPTInferenceWrapper)

        # Create the wrapper
        engine_wrapper = MCoreEngineWithCleanup(mock_engine, mock_wrapper, self.mock_tokenizer)

        # Call __del__
        engine_wrapper.__del__()

        # Verify cleanup was called
        mock_cleanup.assert_called_once()

    @patch("nemo_deploy.llm.inference.inference_base.HAVE_NEMO", True)
    def test_create_mcore_engine_unknown_format_raises(self):
        with self.assertRaises(ValueError):
            create_mcore_engine(path=self.mock_path, model_format="unknown")

    @patch("nemo_deploy.llm.inference.inference_base.HAVE_NEMO", False)
    def test_create_mcore_engine_unavailable_nemo_raises(self):
        with self.assertRaises(UnavailableError):
            create_mcore_engine(path=self.mock_path)

    @patch("nemo_deploy.llm.inference.inference_base.torch_distributed_init")
    @patch("nemo_deploy.llm.inference.inference_base.load_model_config")
    @patch("nemo_deploy.llm.inference.inference_base.initialize_megatron_for_inference")
    @patch("nemo_deploy.llm.inference.inference_base.build_and_load_model")
    @patch("nemo_deploy.llm.inference.inference_base.load_tokenizer")
    def test_attention_backend_conversion_fused(
        self, mock_load_tokenizer, mock_build_model, mock_init_megatron, mock_load_config, mock_torch_dist
    ):
        """Test that attention_backend string 'AttnBackend.fused' is converted to enum."""
        from megatron.core.transformer.enums import AttnBackend

        # Setup model config with string attention_backend
        mock_config = MagicMock()
        mock_config.attention_backend = "AttnBackend.fused"
        mock_config.tensor_model_parallel_size = 1
        mock_config.pipeline_model_parallel_size = 1
        mock_config.context_parallel_size = 1
        mock_config.expert_model_parallel_size = 1

        mock_mlm_args = MagicMock()
        mock_load_config.return_value = (mock_config, mock_mlm_args)

        mock_model = MagicMock()
        mock_build_model.return_value = [mock_model]
        mock_tokenizer = MagicMock()
        mock_load_tokenizer.return_value = mock_tokenizer

        # Call the function
        setup_megatron_model_and_tokenizer_for_inference(
            checkpoint_path=self.mock_path,
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
        )

        # Verify that attention_backend was converted to enum
        self.assertEqual(mock_config.attention_backend, AttnBackend.fused)

    @patch("nemo_deploy.llm.inference.inference_base.torch_distributed_init")
    @patch("nemo_deploy.llm.inference.inference_base.load_model_config")
    @patch("nemo_deploy.llm.inference.inference_base.initialize_megatron_for_inference")
    @patch("nemo_deploy.llm.inference.inference_base.build_and_load_model")
    @patch("nemo_deploy.llm.inference.inference_base.load_tokenizer")
    def test_attention_backend_conversion_flash(
        self, mock_load_tokenizer, mock_build_model, mock_init_megatron, mock_load_config, mock_torch_dist
    ):
        """Test that attention_backend string 'AttnBackend.flash' is converted to enum."""
        from megatron.core.transformer.enums import AttnBackend

        # Setup model config with string attention_backend
        mock_config = MagicMock()
        mock_config.attention_backend = "AttnBackend.flash"
        mock_config.tensor_model_parallel_size = 1
        mock_config.pipeline_model_parallel_size = 1
        mock_config.context_parallel_size = 1
        mock_config.expert_model_parallel_size = 1

        mock_mlm_args = MagicMock()
        mock_load_config.return_value = (mock_config, mock_mlm_args)

        mock_model = MagicMock()
        mock_build_model.return_value = [mock_model]
        mock_tokenizer = MagicMock()
        mock_load_tokenizer.return_value = mock_tokenizer

        # Call the function
        setup_megatron_model_and_tokenizer_for_inference(
            checkpoint_path=self.mock_path,
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
        )

        # Verify that attention_backend was converted to enum
        self.assertEqual(mock_config.attention_backend, AttnBackend.flash)

    @patch("nemo_deploy.llm.inference.inference_base.torch_distributed_init")
    @patch("nemo_deploy.llm.inference.inference_base.load_model_config")
    @patch("nemo_deploy.llm.inference.inference_base.initialize_megatron_for_inference")
    @patch("nemo_deploy.llm.inference.inference_base.build_and_load_model")
    @patch("nemo_deploy.llm.inference.inference_base.load_tokenizer")
    def test_attention_backend_conversion_unfused(
        self, mock_load_tokenizer, mock_build_model, mock_init_megatron, mock_load_config, mock_torch_dist
    ):
        """Test that attention_backend string 'AttnBackend.unfused' is converted to enum."""
        from megatron.core.transformer.enums import AttnBackend

        # Setup model config with string attention_backend
        mock_config = MagicMock()
        mock_config.attention_backend = "AttnBackend.unfused"
        mock_config.tensor_model_parallel_size = 1
        mock_config.pipeline_model_parallel_size = 1
        mock_config.context_parallel_size = 1
        mock_config.expert_model_parallel_size = 1

        mock_mlm_args = MagicMock()
        mock_load_config.return_value = (mock_config, mock_mlm_args)

        mock_model = MagicMock()
        mock_build_model.return_value = [mock_model]
        mock_tokenizer = MagicMock()
        mock_load_tokenizer.return_value = mock_tokenizer

        # Call the function
        setup_megatron_model_and_tokenizer_for_inference(
            checkpoint_path=self.mock_path,
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
        )

        # Verify that attention_backend was converted to enum
        self.assertEqual(mock_config.attention_backend, AttnBackend.unfused)

    @patch("nemo_deploy.llm.inference.inference_base.torch_distributed_init")
    @patch("nemo_deploy.llm.inference.inference_base.load_model_config")
    @patch("nemo_deploy.llm.inference.inference_base.initialize_megatron_for_inference")
    @patch("nemo_deploy.llm.inference.inference_base.build_and_load_model")
    @patch("nemo_deploy.llm.inference.inference_base.load_tokenizer")
    def test_attention_backend_conversion_local(
        self, mock_load_tokenizer, mock_build_model, mock_init_megatron, mock_load_config, mock_torch_dist
    ):
        """Test that attention_backend string 'AttnBackend.local' is converted to enum."""
        from megatron.core.transformer.enums import AttnBackend

        # Setup model config with string attention_backend
        mock_config = MagicMock()
        mock_config.attention_backend = "AttnBackend.local"
        mock_config.tensor_model_parallel_size = 1
        mock_config.pipeline_model_parallel_size = 1
        mock_config.context_parallel_size = 1
        mock_config.expert_model_parallel_size = 1

        mock_mlm_args = MagicMock()
        mock_load_config.return_value = (mock_config, mock_mlm_args)

        mock_model = MagicMock()
        mock_build_model.return_value = [mock_model]
        mock_tokenizer = MagicMock()
        mock_load_tokenizer.return_value = mock_tokenizer

        # Call the function
        setup_megatron_model_and_tokenizer_for_inference(
            checkpoint_path=self.mock_path,
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
        )

        # Verify that attention_backend was converted to enum
        self.assertEqual(mock_config.attention_backend, AttnBackend.local)

    @patch("nemo_deploy.llm.inference.inference_base.torch_distributed_init")
    @patch("nemo_deploy.llm.inference.inference_base.load_model_config")
    @patch("nemo_deploy.llm.inference.inference_base.initialize_megatron_for_inference")
    @patch("nemo_deploy.llm.inference.inference_base.build_and_load_model")
    @patch("nemo_deploy.llm.inference.inference_base.load_tokenizer")
    def test_attention_backend_conversion_auto(
        self, mock_load_tokenizer, mock_build_model, mock_init_megatron, mock_load_config, mock_torch_dist
    ):
        """Test that attention_backend string 'AttnBackend.auto' is converted to enum."""
        from megatron.core.transformer.enums import AttnBackend

        # Setup model config with string attention_backend
        mock_config = MagicMock()
        mock_config.attention_backend = "AttnBackend.auto"
        mock_config.tensor_model_parallel_size = 1
        mock_config.pipeline_model_parallel_size = 1
        mock_config.context_parallel_size = 1
        mock_config.expert_model_parallel_size = 1

        mock_mlm_args = MagicMock()
        mock_load_config.return_value = (mock_config, mock_mlm_args)

        mock_model = MagicMock()
        mock_build_model.return_value = [mock_model]
        mock_tokenizer = MagicMock()
        mock_load_tokenizer.return_value = mock_tokenizer

        # Call the function
        setup_megatron_model_and_tokenizer_for_inference(
            checkpoint_path=self.mock_path,
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
        )

        # Verify that attention_backend was converted to enum
        self.assertEqual(mock_config.attention_backend, AttnBackend.auto)

    @patch("nemo_deploy.llm.inference.inference_base.torch_distributed_init")
    @patch("nemo_deploy.llm.inference.inference_base.load_model_config")
    @patch("nemo_deploy.llm.inference.inference_base.initialize_megatron_for_inference")
    @patch("nemo_deploy.llm.inference.inference_base.build_and_load_model")
    @patch("nemo_deploy.llm.inference.inference_base.load_tokenizer")
    def test_attention_backend_already_enum(
        self, mock_load_tokenizer, mock_build_model, mock_init_megatron, mock_load_config, mock_torch_dist
    ):
        """Test that attention_backend enum is not modified if already an enum."""
        from megatron.core.transformer.enums import AttnBackend

        # Setup model config with enum attention_backend (already converted)
        mock_config = MagicMock()
        mock_config.attention_backend = AttnBackend.flash
        mock_config.tensor_model_parallel_size = 1
        mock_config.pipeline_model_parallel_size = 1
        mock_config.context_parallel_size = 1
        mock_config.expert_model_parallel_size = 1

        mock_mlm_args = MagicMock()
        mock_load_config.return_value = (mock_config, mock_mlm_args)

        mock_model = MagicMock()
        mock_build_model.return_value = [mock_model]
        mock_tokenizer = MagicMock()
        mock_load_tokenizer.return_value = mock_tokenizer

        # Call the function
        setup_megatron_model_and_tokenizer_for_inference(
            checkpoint_path=self.mock_path,
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
        )

        # Verify that attention_backend remains the same enum
        self.assertEqual(mock_config.attention_backend, AttnBackend.flash)

    @patch("nemo_deploy.llm.inference.inference_base.torch_distributed_init")
    @patch("nemo_deploy.llm.inference.inference_base.load_model_config")
    @patch("nemo_deploy.llm.inference.inference_base.initialize_megatron_for_inference")
    @patch("nemo_deploy.llm.inference.inference_base.build_and_load_model")
    @patch("nemo_deploy.llm.inference.inference_base.load_tokenizer")
    def test_attention_backend_missing_attribute(
        self, mock_load_tokenizer, mock_build_model, mock_init_megatron, mock_load_config, mock_torch_dist
    ):
        """Test that missing attention_backend attribute doesn't cause an error."""
        # Setup model config WITHOUT attention_backend attribute
        mock_config = MagicMock(spec=[])  # spec=[] means no attributes
        mock_config.tensor_model_parallel_size = 1
        mock_config.pipeline_model_parallel_size = 1
        mock_config.context_parallel_size = 1
        mock_config.expert_model_parallel_size = 1

        # Remove the attention_backend attribute explicitly
        if hasattr(mock_config, "attention_backend"):
            delattr(mock_config, "attention_backend")

        mock_mlm_args = MagicMock()
        mock_load_config.return_value = (mock_config, mock_mlm_args)

        mock_model = MagicMock()
        mock_build_model.return_value = [mock_model]
        mock_tokenizer = MagicMock()
        mock_load_tokenizer.return_value = mock_tokenizer

        # Call the function - should not raise an error
        result = setup_megatron_model_and_tokenizer_for_inference(
            checkpoint_path=self.mock_path,
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
        )

        # Verify function completed successfully
        self.assertIsNotNone(result)
        self.assertEqual(len(result), 3)  # Returns (model, tokenizer, mlm_args)


if __name__ == "__main__":
    unittest.main()
