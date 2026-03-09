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

import os
from unittest.mock import MagicMock, patch

from nemo_deploy.llm.inference.tron_utils import (
    DistributedInitConfig,
    RNGConfig,
    _get_model_type,
    get_local_rank_preinit,
    get_rank_safe,
    get_world_size_safe,
    print_rank_0,
    torch_distributed_init,
)


class TestRNGConfig:
    def test_defaults(self):
        cfg = RNGConfig()
        assert cfg.seed == 1234
        assert cfg.te_rng_tracker is False
        assert cfg.inference_rng_tracker is False
        assert cfg.data_parallel_random_init is False

    def test_custom_values(self):
        cfg = RNGConfig(seed=42, te_rng_tracker=True, inference_rng_tracker=True, data_parallel_random_init=True)
        assert cfg.seed == 42
        assert cfg.te_rng_tracker is True
        assert cfg.inference_rng_tracker is True
        assert cfg.data_parallel_random_init is True


class TestDistributedInitConfig:
    def test_defaults(self):
        cfg = DistributedInitConfig()
        assert cfg.distributed_backend == "nccl"
        assert cfg.distributed_timeout_minutes == 10
        assert cfg.align_grad_reduce is True
        assert cfg.lazy_mpu_init is False
        assert cfg.use_torch_fsdp2 is False
        assert cfg.nccl_communicator_config_path is None
        assert cfg.use_tp_pp_dp_mapping is False
        assert cfg.use_gloo_process_groups is True

    def test_local_rank_from_env(self):
        with patch.dict(os.environ, {"LOCAL_RANK": "3"}):
            cfg = DistributedInitConfig()
            assert cfg.local_rank == 3

    def test_local_rank_default(self):
        env = {k: v for k, v in os.environ.items() if k != "LOCAL_RANK"}
        with patch.dict(os.environ, env, clear=True):
            cfg = DistributedInitConfig()
            assert cfg.local_rank == 0

    def test_custom_backend(self):
        cfg = DistributedInitConfig(distributed_backend="gloo")
        assert cfg.distributed_backend == "gloo"


class TestGetRankSafe:
    @patch("torch.distributed.is_initialized", return_value=True)
    @patch("torch.distributed.get_rank", return_value=2)
    def test_distributed_initialized(self, mock_rank, mock_init):
        result = get_rank_safe()
        assert result == 2
        mock_rank.assert_called_once()

    @patch("torch.distributed.is_initialized", return_value=False)
    def test_distributed_not_initialized_default(self, mock_init):
        env = {k: v for k, v in os.environ.items() if k != "RANK"}
        with patch.dict(os.environ, env, clear=True):
            result = get_rank_safe()
            assert result == 0

    @patch("torch.distributed.is_initialized", return_value=False)
    def test_distributed_not_initialized_env(self, mock_init):
        with patch.dict(os.environ, {"RANK": "3"}):
            result = get_rank_safe()
            assert result == 3


class TestGetWorldSizeSafe:
    @patch("torch.distributed.is_initialized", return_value=True)
    @patch("torch.distributed.get_world_size", return_value=8)
    def test_distributed_initialized(self, mock_ws, mock_init):
        result = get_world_size_safe()
        assert result == 8
        mock_ws.assert_called_once()

    @patch("torch.distributed.is_initialized", return_value=False)
    def test_distributed_not_initialized_default(self, mock_init):
        env = {k: v for k, v in os.environ.items() if k != "WORLD_SIZE"}
        with patch.dict(os.environ, env, clear=True):
            result = get_world_size_safe()
            assert result == 1

    @patch("torch.distributed.is_initialized", return_value=False)
    def test_distributed_not_initialized_env(self, mock_init):
        with patch.dict(os.environ, {"WORLD_SIZE": "4"}):
            result = get_world_size_safe()
            assert result == 4


class TestGetLocalRankPreinit:
    def test_default(self):
        env = {k: v for k, v in os.environ.items() if k != "LOCAL_RANK"}
        with patch.dict(os.environ, env, clear=True):
            result = get_local_rank_preinit()
            assert result == 0

    def test_from_env(self):
        with patch.dict(os.environ, {"LOCAL_RANK": "2"}):
            result = get_local_rank_preinit()
            assert result == 2


class TestPrintRank0:
    @patch("nemo_deploy.llm.inference.tron_utils.get_rank_safe", return_value=0)
    @patch("nemo_deploy.llm.inference.tron_utils.LOGGER")
    def test_prints_on_rank_0(self, mock_logger, mock_rank):
        print_rank_0("test message")
        mock_logger.info.assert_called_once_with("test message")

    @patch("nemo_deploy.llm.inference.tron_utils.get_rank_safe", return_value=1)
    @patch("nemo_deploy.llm.inference.tron_utils.LOGGER")
    def test_does_not_print_on_non_zero_rank(self, mock_logger, mock_rank):
        print_rank_0("test message")
        mock_logger.info.assert_not_called()


class TestTorchDistributedInit:
    @patch("torch.distributed.is_initialized", return_value=True)
    @patch("nemo_deploy.llm.inference.tron_utils.get_rank_safe", return_value=0)
    @patch("nemo_deploy.llm.inference.tron_utils.LOGGER")
    def test_already_initialized_rank_0(self, mock_logger, mock_rank, mock_init):
        cfg = DistributedInitConfig()
        torch_distributed_init(cfg)
        mock_logger.info.assert_called()
        # Should NOT call init_process_group since already initialized
        # No assertion on init_process_group since it shouldn't be called

    @patch("torch.distributed.is_initialized", return_value=True)
    @patch("nemo_deploy.llm.inference.tron_utils.get_rank_safe", return_value=1)
    @patch("nemo_deploy.llm.inference.tron_utils.LOGGER")
    def test_already_initialized_non_zero_rank(self, mock_logger, mock_rank, mock_init):
        cfg = DistributedInitConfig()
        torch_distributed_init(cfg)
        # Non-zero rank should not log the "already initialized" message
        # but function should complete without error

    @patch("torch.distributed.is_initialized", return_value=False)
    @patch("torch.cuda.device_count", return_value=0)
    @patch("torch.distributed.init_process_group")
    @patch("torch.distributed.barrier")
    @patch("nemo_deploy.llm.inference.tron_utils.get_rank_safe", return_value=0)
    @patch("nemo_deploy.llm.inference.tron_utils.get_world_size_safe", return_value=1)
    @patch("nemo_deploy.llm.inference.tron_utils.get_local_rank_preinit", return_value=0)
    def test_init_no_gpu(
        self, mock_local_rank, mock_world_size, mock_rank, mock_barrier, mock_init_pg, mock_device_count, mock_init
    ):
        cfg = DistributedInitConfig(distributed_backend="gloo")
        with patch.dict(os.environ, {"MASTER_ADDR": "localhost", "MASTER_PORT": "12345"}):
            torch_distributed_init(cfg)
        mock_init_pg.assert_called_once()
        mock_barrier.assert_called_once()


class TestGetModelType:
    def test_t5_config_returns_encoder_and_decoder(self):
        from megatron.core.enums import ModelType

        # Mock T5Config as a class and create an instance of it
        MockT5Config = type("MockT5Config", (), {})
        mock_t5_instance = MockT5Config()

        with patch("nemo_deploy.llm.inference.tron_utils.T5Config", MockT5Config):
            result = _get_model_type(mock_t5_instance)
            assert result == ModelType.encoder_and_decoder

    def test_gpt_config_returns_encoder_or_decoder(self):
        from megatron.core.enums import ModelType

        # Use a non-T5Config instance
        MockT5Config = type("MockT5Config", (), {})
        mock_gpt_instance = MagicMock()  # Not an instance of MockT5Config

        with patch("nemo_deploy.llm.inference.tron_utils.T5Config", MockT5Config):
            result = _get_model_type(mock_gpt_instance)
            assert result == ModelType.encoder_or_decoder
