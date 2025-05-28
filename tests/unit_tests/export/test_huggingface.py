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

import torch
import pytest
from pathlib import Path
import tempfile
from nemo_export import huggingface as hf_export
from nemo_export.huggingface.utils import load_config, torch_dtype_from_mcore_config


HF_FORMAT = "hf"

@pytest.mark.unit
def test_get_exporter():
    """ Test that the Gemma exporter is registered via the decorator."""
    assert hf_export.get_exporter("GemmaModel", HF_FORMAT) is hf_export.HFGemmaExporter


@pytest.mark.unit
def test_load_gemma_connector():
    """ Test that the Gemma connector is loaded correctly from a checkpoint path."""
    from nemo.collections.llm import GemmaModel, GemmaConfig2B
    model = GemmaModel(GemmaConfig2B())

    with tempfile.TemporaryDirectory() as temp_dir:
        create_gemma_context(temp_dir, model)
        connector = hf_export.load_connector(temp_dir, HF_FORMAT)
        assert isinstance(connector, hf_export.HFGemmaExporter)


@pytest.mark.unit
def test_load_gemma2b_config():
    """ Test that the Gemma 2B config is loaded correctly.
    Asserts that parameters required by the exporter are present.
    """
    from nemo.collections.llm import GemmaModel, GemmaConfig2B
    with tempfile.TemporaryDirectory() as temp_dir:
        model = GemmaModel(GemmaConfig2B(fp16=True))
        create_gemma_context(temp_dir, model)
        config = load_config(temp_dir)
        assert config.num_layers == 18
        assert config.hidden_size == 2048
        assert config.num_attention_heads == 8
        assert config.ffn_hidden_size == 16384
        assert config.num_query_groups == 1
        assert config.kv_channels == 256
        assert torch_dtype_from_mcore_config(config) == torch.float16


def create_gemma_context(output_path: Path | str, model):
    """ Creates a context for a Gemma model. """
    from nemo.lightning import MegatronStrategy, Trainer
    from nemo.lightning.io.pl import TrainerContext
    from nemo.utils.get_rank import is_global_rank_zero

    output_path = Path(output_path)

    _trainer = Trainer(
        devices=1,
        accelerator="cpu",
        strategy=MegatronStrategy(ckpt_save_optimizer=False, always_save_context=True)
    )
    _trainer.state.fn = "fit"
    _trainer.strategy.connect(model)
    _trainer.strategy.setup_environment()

    output_path.mkdir(parents=True, exist_ok=True)
    if is_global_rank_zero():
        TrainerContext.from_trainer(_trainer).io_dump(output_path / "context", yaml_attrs=["model"])

