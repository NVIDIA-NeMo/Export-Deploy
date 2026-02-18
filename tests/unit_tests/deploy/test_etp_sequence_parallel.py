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

"""
Unit tests for the expert_tensor_parallel_size (ETP) and sequence_parallel parameters
introduced in commit "feat: Pass ETP to Ray inframework".

Covers:
- DeployRay.deploy_inframework_model passes params to MegatronRayDeployable
- setup_megatron_model_and_tokenizer_for_inference applies params to model_config
- setup_model_and_tokenizer_for_inference applies params to model_config
- create_mcore_engine defaults both params to 1 when None and passes them down
- MegatronLLMDeployable.__init__ passes params to create_mcore_engine
- CLI argument parsers in deploy scripts accept both new flags
"""

import importlib.util
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_script(script_path: str, module_name: str):
    """Load a script file as a module without executing __main__."""
    spec = importlib.util.spec_from_file_location(module_name, script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_REPO_ROOT = Path(__file__).parents[3]


# ---------------------------------------------------------------------------
# DeployRay.deploy_inframework_model
# ---------------------------------------------------------------------------


class TestDeployRayETPSequenceParallel(unittest.TestCase):
    """Tests that DeployRay.deploy_inframework_model passes ETP and SP through."""

    def setUp(self):
        self._have_ray_patcher = patch("nemo_deploy.deploy_ray.HAVE_RAY", True)
        self._have_ray_patcher.start()

    def tearDown(self):
        self._have_ray_patcher.stop()

    @patch("nemo_deploy.deploy_ray.ray")
    @patch("nemo_deploy.deploy_ray.serve")
    @patch("nemo_deploy.deploy_ray.MegatronRayDeployable")
    @patch("nemo_deploy.deploy_ray.signal.signal")
    @patch("nemo_deploy.deploy_ray.DeployRay._start")
    def test_deploy_inframework_passes_expert_tensor_parallel_size(
        self, mock_start, mock_signal, mock_megatron, mock_serve, mock_ray
    ):
        """expert_tensor_parallel_size is forwarded to MegatronRayDeployable.bind()."""
        from nemo_deploy.deploy_ray import DeployRay

        mock_app = MagicMock()
        mock_options = MagicMock()
        mock_options.bind.return_value = mock_app
        mock_megatron.options.return_value = mock_options

        deploy = DeployRay()
        deploy.deploy_inframework_model(
            megatron_checkpoint="/path/to/model.megatron",
            num_gpus=4,
            tensor_model_parallel_size=2,
            expert_tensor_parallel_size=4,
            test_mode=True,
        )

        _, bind_kwargs = mock_options.bind.call_args
        assert bind_kwargs["expert_tensor_parallel_size"] == 4

    @patch("nemo_deploy.deploy_ray.ray")
    @patch("nemo_deploy.deploy_ray.serve")
    @patch("nemo_deploy.deploy_ray.MegatronRayDeployable")
    @patch("nemo_deploy.deploy_ray.signal.signal")
    @patch("nemo_deploy.deploy_ray.DeployRay._start")
    def test_deploy_inframework_passes_sequence_parallel(
        self, mock_start, mock_signal, mock_megatron, mock_serve, mock_ray
    ):
        """sequence_parallel=True is forwarded to MegatronRayDeployable.bind()."""
        from nemo_deploy.deploy_ray import DeployRay

        mock_app = MagicMock()
        mock_options = MagicMock()
        mock_options.bind.return_value = mock_app
        mock_megatron.options.return_value = mock_options

        deploy = DeployRay()
        deploy.deploy_inframework_model(
            megatron_checkpoint="/path/to/model.megatron",
            num_gpus=4,
            sequence_parallel=True,
            test_mode=True,
        )

        _, bind_kwargs = mock_options.bind.call_args
        assert bind_kwargs["sequence_parallel"] is True

    @patch("nemo_deploy.deploy_ray.ray")
    @patch("nemo_deploy.deploy_ray.serve")
    @patch("nemo_deploy.deploy_ray.MegatronRayDeployable")
    @patch("nemo_deploy.deploy_ray.signal.signal")
    @patch("nemo_deploy.deploy_ray.DeployRay._start")
    def test_deploy_inframework_default_etp_and_sp(
        self, mock_start, mock_signal, mock_megatron, mock_serve, mock_ray
    ):
        """Defaults: expert_tensor_parallel_size=1, sequence_parallel=False."""
        from nemo_deploy.deploy_ray import DeployRay

        mock_app = MagicMock()
        mock_options = MagicMock()
        mock_options.bind.return_value = mock_app
        mock_megatron.options.return_value = mock_options

        deploy = DeployRay()
        deploy.deploy_inframework_model(
            megatron_checkpoint="/path/to/model.megatron",
            num_gpus=1,
            test_mode=True,
        )

        _, bind_kwargs = mock_options.bind.call_args
        assert bind_kwargs["expert_tensor_parallel_size"] == 1
        assert bind_kwargs["sequence_parallel"] is False


# ---------------------------------------------------------------------------
# setup_megatron_model_and_tokenizer_for_inference
# ---------------------------------------------------------------------------


class TestSetupMegatronModelETPSequenceParallel(unittest.TestCase):
    """Tests that ETP and SP are applied to model_config in the megatron inference path."""

    def _common_patches(self):
        return [
            patch("nemo_deploy.llm.inference.inference_base.torch_distributed_init"),
            patch("nemo_deploy.llm.inference.inference_base.load_model_config"),
            patch("nemo_deploy.llm.inference.inference_base.initialize_megatron_for_inference"),
            patch("nemo_deploy.llm.inference.inference_base.build_and_load_model"),
            patch("nemo_deploy.llm.inference.inference_base.load_tokenizer"),
        ]

    def _mock_config(self):
        cfg = MagicMock()
        cfg.attention_backend = None
        cfg.tensor_model_parallel_size = 1
        cfg.pipeline_model_parallel_size = 1
        cfg.context_parallel_size = 1
        cfg.expert_model_parallel_size = 1
        return cfg

    def test_expert_tensor_parallel_size_applied_to_model_config(self):
        """expert_tensor_parallel_size is set on model_config when provided."""
        from nemo_deploy.llm.inference.inference_base import (
            setup_megatron_model_and_tokenizer_for_inference,
        )

        patches = self._common_patches()
        mocks = [p.start() for p in patches]
        _torch_dist, mock_load_config, _init_meg, mock_build, mock_load_tok = mocks

        cfg = self._mock_config()
        mock_load_config.return_value = (cfg, MagicMock())
        mock_build.return_value = [MagicMock()]
        mock_load_tok.return_value = MagicMock()

        try:
            setup_megatron_model_and_tokenizer_for_inference(
                checkpoint_path=Path("/fake/path"),
                expert_tensor_parallel_size=4,
            )
            assert cfg.expert_tensor_parallel_size == 4
        finally:
            for p in patches:
                p.stop()

    def test_sequence_parallel_applied_to_model_config(self):
        """sequence_parallel is set on model_config when provided."""
        from nemo_deploy.llm.inference.inference_base import (
            setup_megatron_model_and_tokenizer_for_inference,
        )

        patches = self._common_patches()
        mocks = [p.start() for p in patches]
        _torch_dist, mock_load_config, _init_meg, mock_build, mock_load_tok = mocks

        cfg = self._mock_config()
        mock_load_config.return_value = (cfg, MagicMock())
        mock_build.return_value = [MagicMock()]
        mock_load_tok.return_value = MagicMock()

        try:
            setup_megatron_model_and_tokenizer_for_inference(
                checkpoint_path=Path("/fake/path"),
                sequence_parallel=True,
            )
            assert cfg.sequence_parallel is True
        finally:
            for p in patches:
                p.stop()

    def test_etp_not_overwritten_when_none(self):
        """Pre-existing expert_tensor_parallel_size is not overwritten when None is passed."""
        from nemo_deploy.llm.inference.inference_base import (
            setup_megatron_model_and_tokenizer_for_inference,
        )

        patches = self._common_patches()
        mocks = [p.start() for p in patches]
        _torch_dist, mock_load_config, _init_meg, mock_build, mock_load_tok = mocks

        cfg = self._mock_config()
        sentinel = object()
        cfg.expert_tensor_parallel_size = sentinel
        mock_load_config.return_value = (cfg, MagicMock())
        mock_build.return_value = [MagicMock()]
        mock_load_tok.return_value = MagicMock()

        try:
            setup_megatron_model_and_tokenizer_for_inference(
                checkpoint_path=Path("/fake/path"),
                expert_tensor_parallel_size=None,
            )
            assert cfg.expert_tensor_parallel_size is sentinel
        finally:
            for p in patches:
                p.stop()

    def test_sp_not_overwritten_when_none(self):
        """Pre-existing sequence_parallel is not overwritten when None is passed."""
        from nemo_deploy.llm.inference.inference_base import (
            setup_megatron_model_and_tokenizer_for_inference,
        )

        patches = self._common_patches()
        mocks = [p.start() for p in patches]
        _torch_dist, mock_load_config, _init_meg, mock_build, mock_load_tok = mocks

        cfg = self._mock_config()
        sentinel = object()
        cfg.sequence_parallel = sentinel
        mock_load_config.return_value = (cfg, MagicMock())
        mock_build.return_value = [MagicMock()]
        mock_load_tok.return_value = MagicMock()

        try:
            setup_megatron_model_and_tokenizer_for_inference(
                checkpoint_path=Path("/fake/path"),
                sequence_parallel=None,
            )
            assert cfg.sequence_parallel is sentinel
        finally:
            for p in patches:
                p.stop()


# ---------------------------------------------------------------------------
# setup_model_and_tokenizer_for_inference (NeMo checkpoint path)
# ---------------------------------------------------------------------------


class TestSetupModelETPSequenceParallel(unittest.TestCase):
    """Tests that ETP and SP are applied to model_config in the NeMo inference path."""

    def _common_patches(self):
        return [
            patch("nemo_deploy.llm.inference.inference_base.HAVE_NEMO", True),
            patch("nemo_deploy.llm.inference.inference_base.set_modelopt_spec_if_exists_in_ckpt"),
            patch("nemo_deploy.llm.inference.inference_base.torch_distributed_init"),
            patch("nemo_deploy.llm.inference.inference_base.io.load_context"),
            patch("nemo_deploy.llm.inference.inference_base.check_is_distributed_checkpoint"),
            patch("nemo_deploy.llm.inference.inference_base.ckpt_to_weights_subdir"),
            patch("nemo_deploy.llm.inference.inference_base.ckpt_to_context_subdir"),
            patch("nemo_deploy.llm.inference.inference_base.initialize_megatron_for_inference"),
            patch("nemo_deploy.llm.inference.inference_base.get_model_from_config"),
            patch("nemo_deploy.llm.inference.inference_base.load_nemo_checkpoint_to_tron_model"),
            patch("nemo_deploy.llm.inference.inference_base.peel"),
            patch("nemo_deploy.llm.inference.inference_base.MCoreTokenizerWrappper"),
        ]

    def _build_context(self):
        cfg = MagicMock()
        cfg.tensor_model_parallel_size = 1
        cfg.pipeline_model_parallel_size = 1
        cfg.context_parallel_size = 1
        cfg.expert_model_parallel_size = 1
        ctx = MagicMock()
        ctx.config = cfg
        ctx.tokenizer = MagicMock()
        return ctx, cfg

    def test_expert_tensor_parallel_size_applied(self):
        """expert_tensor_parallel_size is assigned to model_config in the NeMo path."""
        from nemo_deploy.llm.inference.inference_base import setup_model_and_tokenizer_for_inference

        patches = self._common_patches()
        mocks = [p.start() for p in patches]
        (
            _have_nemo,
            _set_modelopt,
            _torch_dist,
            mock_load_ctx,
            mock_check_dist,
            _weights,
            _context,
            _init_meg,
            mock_get_model,
            _load_ckpt,
            mock_peel,
            _tok_wrapper,
        ) = mocks

        ctx, cfg = self._build_context()
        mock_load_ctx.return_value = ctx
        mock_check_dist.return_value = True
        mock_get_model.return_value = [MagicMock()]
        mock_peel.return_value = MagicMock()

        try:
            setup_model_and_tokenizer_for_inference(
                checkpoint_path=Path("/fake/path"),
                expert_tensor_parallel_size=8,
            )
            assert cfg.expert_tensor_parallel_size == 8
        finally:
            for p in patches:
                p.stop()

    def test_sequence_parallel_applied(self):
        """sequence_parallel is assigned to model_config in the NeMo path."""
        from nemo_deploy.llm.inference.inference_base import setup_model_and_tokenizer_for_inference

        patches = self._common_patches()
        mocks = [p.start() for p in patches]
        (
            _have_nemo,
            _set_modelopt,
            _torch_dist,
            mock_load_ctx,
            mock_check_dist,
            _weights,
            _context,
            _init_meg,
            mock_get_model,
            _load_ckpt,
            mock_peel,
            _tok_wrapper,
        ) = mocks

        ctx, cfg = self._build_context()
        mock_load_ctx.return_value = ctx
        mock_check_dist.return_value = True
        mock_get_model.return_value = [MagicMock()]
        mock_peel.return_value = MagicMock()

        try:
            setup_model_and_tokenizer_for_inference(
                checkpoint_path=Path("/fake/path"),
                sequence_parallel=True,
            )
            assert cfg.sequence_parallel is True
        finally:
            for p in patches:
                p.stop()


# ---------------------------------------------------------------------------
# create_mcore_engine – default / pass-through behaviour
# ---------------------------------------------------------------------------


class TestCreateMcoreEngineETPSequenceParallel(unittest.TestCase):
    """Tests that create_mcore_engine handles ETP/SP defaults and passes them down."""

    @patch("nemo_deploy.llm.inference.inference_base.HAVE_NEMO", True)
    @patch("nemo_deploy.llm.inference.inference_base.setup_model_and_tokenizer_for_inference")
    @patch("nemo_deploy.llm.inference.inference_base.MCoreEngine")
    def test_etp_defaults_to_1_when_none(self, mock_engine_cls, mock_setup):
        """expert_tensor_parallel_size=None is normalised to 1 before forwarding."""
        from nemo_deploy.llm.inference.inference_base import create_mcore_engine

        mock_setup.return_value = ([MagicMock()], MagicMock())
        mock_engine_cls.return_value = MagicMock()

        create_mcore_engine(path=Path("/fake"), model_format="nemo", expert_tensor_parallel_size=None)

        _, kwargs = mock_setup.call_args
        assert kwargs["expert_tensor_parallel_size"] == 1

    @patch("nemo_deploy.llm.inference.inference_base.HAVE_NEMO", True)
    @patch("nemo_deploy.llm.inference.inference_base.setup_model_and_tokenizer_for_inference")
    @patch("nemo_deploy.llm.inference.inference_base.MCoreEngine")
    def test_sp_defaults_to_1_when_none(self, mock_engine_cls, mock_setup):
        """sequence_parallel=None is normalised to 1 before forwarding."""
        from nemo_deploy.llm.inference.inference_base import create_mcore_engine

        mock_setup.return_value = ([MagicMock()], MagicMock())
        mock_engine_cls.return_value = MagicMock()

        create_mcore_engine(path=Path("/fake"), model_format="nemo", sequence_parallel=None)

        _, kwargs = mock_setup.call_args
        assert kwargs["sequence_parallel"] == 1

    @patch("nemo_deploy.llm.inference.inference_base.HAVE_NEMO", True)
    @patch("nemo_deploy.llm.inference.inference_base.setup_model_and_tokenizer_for_inference")
    @patch("nemo_deploy.llm.inference.inference_base.MCoreEngine")
    def test_explicit_etp_passed_through(self, mock_engine_cls, mock_setup):
        """An explicit expert_tensor_parallel_size value is forwarded unchanged."""
        from nemo_deploy.llm.inference.inference_base import create_mcore_engine

        mock_setup.return_value = ([MagicMock()], MagicMock())
        mock_engine_cls.return_value = MagicMock()

        create_mcore_engine(path=Path("/fake"), model_format="nemo", expert_tensor_parallel_size=4)

        _, kwargs = mock_setup.call_args
        assert kwargs["expert_tensor_parallel_size"] == 4

    @patch("nemo_deploy.llm.inference.inference_base.HAVE_NEMO", True)
    @patch("nemo_deploy.llm.inference.inference_base.setup_model_and_tokenizer_for_inference")
    @patch("nemo_deploy.llm.inference.inference_base.MCoreEngine")
    def test_explicit_sp_passed_through(self, mock_engine_cls, mock_setup):
        """An explicit sequence_parallel=True value is forwarded unchanged."""
        from nemo_deploy.llm.inference.inference_base import create_mcore_engine

        mock_setup.return_value = ([MagicMock()], MagicMock())
        mock_engine_cls.return_value = MagicMock()

        create_mcore_engine(path=Path("/fake"), model_format="nemo", sequence_parallel=True)

        _, kwargs = mock_setup.call_args
        assert kwargs["sequence_parallel"] is True


# ---------------------------------------------------------------------------
# MegatronLLMDeployable.__init__
# ---------------------------------------------------------------------------


class TestMegatronLLMDeployableETPSequenceParallel(unittest.TestCase):
    """Tests that MegatronLLMDeployable passes ETP and SP to create_mcore_engine."""

    @patch("nemo_deploy.llm.megatronllm_deployable.HAVE_TRITON", True)
    @patch("nemo_deploy.llm.megatronllm_deployable.create_mcore_engine")
    def test_expert_tensor_parallel_size_forwarded(self, mock_create):
        """expert_tensor_parallel_size is forwarded to create_mcore_engine."""
        from nemo_deploy.llm.megatronllm_deployable import MegatronLLMDeployable

        mock_create.return_value = (MagicMock(), MagicMock(), MagicMock())

        MegatronLLMDeployable(
            megatron_checkpoint_filepath="model.ckpt",
            expert_tensor_parallel_size=4,
        )

        kwargs = mock_create.call_args.kwargs
        assert kwargs["expert_tensor_parallel_size"] == 4

    @patch("nemo_deploy.llm.megatronllm_deployable.HAVE_TRITON", True)
    @patch("nemo_deploy.llm.megatronllm_deployable.create_mcore_engine")
    def test_sequence_parallel_forwarded(self, mock_create):
        """sequence_parallel is forwarded to create_mcore_engine."""
        from nemo_deploy.llm.megatronllm_deployable import MegatronLLMDeployable

        mock_create.return_value = (MagicMock(), MagicMock(), MagicMock())

        MegatronLLMDeployable(
            megatron_checkpoint_filepath="model.ckpt",
            sequence_parallel=True,
        )

        kwargs = mock_create.call_args.kwargs
        assert kwargs["sequence_parallel"] is True

    @patch("nemo_deploy.llm.megatronllm_deployable.HAVE_TRITON", True)
    @patch("nemo_deploy.llm.megatronllm_deployable.create_mcore_engine")
    def test_defaults_etp_1_and_sp_false(self, mock_create):
        """Defaults: expert_tensor_parallel_size=1, sequence_parallel=False."""
        from nemo_deploy.llm.megatronllm_deployable import MegatronLLMDeployable

        mock_create.return_value = (MagicMock(), MagicMock(), MagicMock())

        MegatronLLMDeployable(megatron_checkpoint_filepath="model.ckpt")

        kwargs = mock_create.call_args.kwargs
        assert kwargs["expert_tensor_parallel_size"] == 1
        assert kwargs["sequence_parallel"] is False


# ---------------------------------------------------------------------------
# Script argument-parser tests
# ---------------------------------------------------------------------------


class TestDeployRayInframeworkScriptArgs(unittest.TestCase):
    """Tests that deploy_ray_inframework.py parse_args accepts the new flags."""

    @classmethod
    def setUpClass(cls):
        script = str(_REPO_ROOT / "scripts" / "deploy" / "nlp" / "deploy_ray_inframework.py")
        cls._module = _load_script(script, "_nlp_ray_inframework")

    def _parse(self, extra_args=None):
        argv = extra_args or []
        with patch("sys.argv", ["prog"] + argv):
            return self._module.parse_args()

    def test_expert_tensor_parallel_size_default(self):
        args = self._parse()
        assert args.expert_tensor_parallel_size == 1

    def test_expert_tensor_parallel_size_custom(self):
        args = self._parse(["--expert_tensor_parallel_size", "4"])
        assert args.expert_tensor_parallel_size == 4

    def test_sequence_parallel_default_false(self):
        args = self._parse()
        assert args.sequence_parallel is False

    def test_sequence_parallel_enabled(self):
        args = self._parse(["--sequence_parallel"])
        assert args.sequence_parallel is True

    def test_sequence_parallel_disabled_explicitly(self):
        args = self._parse(["--no-sequence_parallel"])
        assert args.sequence_parallel is False


class TestMbridgeDeployRayScriptArgs(unittest.TestCase):
    """Tests that scripts/deploy/llm/mbridge/deploy_ray.py parse_args accepts the new flags."""

    @classmethod
    def setUpClass(cls):
        script = str(_REPO_ROOT / "scripts" / "deploy" / "llm" / "mbridge" / "deploy_ray.py")
        cls._module = _load_script(script, "_mbridge_deploy_ray")

    def _parse(self, extra_args=None):
        argv = extra_args or []
        with patch("sys.argv", ["prog"] + argv):
            return self._module.parse_args()

    def test_expert_tensor_parallel_size_default(self):
        args = self._parse()
        assert args.expert_tensor_parallel_size == 1

    def test_expert_tensor_parallel_size_custom(self):
        args = self._parse(["--expert_tensor_parallel_size", "8"])
        assert args.expert_tensor_parallel_size == 8

    def test_sequence_parallel_default_false(self):
        args = self._parse()
        assert args.sequence_parallel is False

    def test_sequence_parallel_enabled(self):
        args = self._parse(["--sequence_parallel"])
        assert args.sequence_parallel is True

    def test_sequence_parallel_disabled_explicitly(self):
        args = self._parse(["--no-sequence_parallel"])
        assert args.sequence_parallel is False


class TestDeployInframeworkTritonScriptArgs(unittest.TestCase):
    """Tests that scripts/deploy/nlp/deploy_inframework_triton.py get_args accepts the new flags."""

    @classmethod
    def setUpClass(cls):
        script = str(_REPO_ROOT / "scripts" / "deploy" / "nlp" / "deploy_inframework_triton.py")
        cls._module = _load_script(script, "_inframework_triton")

    def _get_args(self, extra_args=None):
        # --triton_model_name is required
        argv = ["--triton_model_name", "test-model"] + (extra_args or [])
        return self._module.get_args(argv)

    def test_expert_tensor_parallel_size_default(self):
        args = self._get_args()
        assert args.expert_tensor_parallel_size == 1

    def test_expert_tensor_parallel_size_custom(self):
        args = self._get_args(["--expert_tensor_parallel_size", "2"])
        assert args.expert_tensor_parallel_size == 2

    def test_sequence_parallel_default_false(self):
        args = self._get_args()
        assert args.sequence_parallel is False

    def test_sequence_parallel_enabled(self):
        args = self._get_args(["--sequence_parallel"])
        assert args.sequence_parallel is True

    def test_sequence_parallel_disabled_explicitly(self):
        args = self._get_args(["--no-sequence_parallel"])
        assert args.sequence_parallel is False


if __name__ == "__main__":
    unittest.main()
