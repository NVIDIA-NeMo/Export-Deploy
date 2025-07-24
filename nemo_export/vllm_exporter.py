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


from typing import List, Literal
import tempfile
from pathlib import Path
import numpy as np
from nemo_deploy import ITritonDeployable
from nemo_deploy.utils import cast_output, str_ndarray2list
from nemo_export.utils import is_nemo2_checkpoint
from nemo_export_deploy_common.import_utils import MISSING_TRITON_MSG, MISSING_NEMO_MSG, MISSING_VLLM_MSG, UnavailableError

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
    """The Exporter class uses vLLM APIs to convert a HF model to vLLM and makes the class, deployable with Triton server.

    Example:
        from nemo_export import vLLMExporter
        from nemo_deploy import DeployPyTriton

        exporter = vLLMExporter()
        exporter.export(model="/path/to/model/")

        server = DeployPyTriton(
            model=exporter,
            triton_model_name='model'
        )

        server.deploy()
        server.serve()
    """

    def __init__(self):
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
        task: Literal['auto', 'generate', 'embedding'] = 'auto'
    ):
        """Exports the HF checkpoint to vLLM and initializes the engine.

        Args:
            model (str): model name or the path
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
                    raise Exception(f"NeMo checkpoint is not supported. Error occured during Hugging Face conversion. Error message: {e}")

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
        if self.lora_models is None:
            self.lora_models = {}
        self.lora_models[lora_model_name] = lora_model

    @property
    def get_triton_input(self):
        inputs = (
            Tensor(name="prompts", shape=(-1,), dtype=bytes),
            Tensor(name="max_tokens", shape=(-1,), dtype=np.int_, optional=True),
            Tensor(name="min_tokens", shape=(-1,), dtype=np.int_, optional=True),
            Tensor(name="top_k", shape=(-1,), dtype=np.int_, optional=True),
            Tensor(name="top_p", shape=(-1,), dtype=np.single, optional=True),
            Tensor(name="temperature", shape=(-1,), dtype=np.single, optional=True),
            Tensor(name="seed", shape=(-1,), dtype=np.int_, optional=True),            
            Tensor(name="logprobs", shape=(-1,), dtype=np.int_, optional=True),            
        )
        return inputs

    @property
    def get_triton_output(self):
        return (
            Tensor(name="sentences", shape=(-1,), dtype=bytes),
            Tensor(name="logits", shape=(-1,), dtype=np.single),            
        )
        return outputs

    @batch
    def triton_infer_fn(self, **inputs: np.ndarray):  # pragma: no cover
        try:
            infer_input = {"input_texts": str_ndarray2list(inputs.pop("prompts"))}
            if "max_tokens" in inputs:
                infer_input["max_tokens"] = int(inputs.pop("max_tokens")[0][0]) 
            if "min_tokens" in inputs:
                infer_input["min_tokens"] = int(inputs.pop("min_tokens")[0][0])
            if "logprobs" in inputs:
                infer_input["logprobs"] = int(inputs.pop("logprobs")[0][0])
            if "seed" in inputs:
                infer_input["seed"] = int(inputs.pop("seed")[0][0])
            if "top_k" in inputs:
                infer_input["top_k"] = int(inputs.pop("top_k")[0][0])
            if "top_p" in inputs:
                infer_input["top_p"] = float(inputs.pop("top_p")[0][0])
            if "temperature" in inputs:
                infer_input["temperature"] = float(inputs.pop("temperature")[0][0])

            output = self.forward(**infer_input)
            if isinstance(output, dict):
                output_infer = {"sentences": cast_output(output["sentences"], np.bytes_)}                
                if "logits" in output.keys():
                    output_infer["logits"] = cast_output(output["logits"], np.single)
            else:
                output_infer = {"sentences": cast_output(output, np.bytes_)}
        except Exception as error:
            err_msg = "An error occurred: {0}".format(str(error))
            output_infer = {"sentences": cast_output([err_msg], np.bytes_)}

        return output_infer

    def forward(
        self,
        input_texts: List[str],
        max_tokens: int = 16,
        min_tokens: int = 0,
        top_k: int = 1,
        top_p: float = 0.1,
        temperature: float = 1.0,
        logprobs: int = None,
        seed: int = None,
        lora_model_name: str = None,
    ):
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
            logprobs=logprobs,
            seed=seed,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
        )

        request_output = self.model.generate(input_texts, sampling_params, lora_request=lora_request)
        print(request_output)
        output = []
        for o in request_output:
            output.append(o.outputs[0].text)

        return output
