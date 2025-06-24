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


import csv
import json
import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from transformers import PreTrainedTokenizer

from nemo_export_deploy_common.import_utils import (
    MISSING_MPI_MSG,
    MISSING_TENSORRT_MSG,
    UnavailableError,
)

try:
    from mpi4py.futures import MPIPoolExecutor

    HAVE_MPI = True
except (ImportError, ModuleNotFoundError):
    from unittest.mock import MagicMock

    MPIPoolExecutor = MagicMock()
    HAVE_MPI = False

try:
    import tensorrt as trt

    HAVE_TRT = True
except (ImportError, ModuleNotFoundError):
    HAVE_TRT = False

try:
    import tensorrt_llm
    from tensorrt_llm.lora_manager import LoraManager
    from tensorrt_llm.runtime import (
        ModelRunner,
        ModelRunnerCpp,
        SamplingConfig,
    )
except (ImportError, ModuleNotFoundError):
    from unittest.mock import MagicMock

    Engine = MagicMock()
    LoraManager = MagicMock()
    QuantMode = MagicMock()
    ModelConfig = MagicMock()
    ModelRunner = MagicMock()
    ModelRunnerCpp = MagicMock()
    SamplingConfig = MagicMock()
    HAVE_TRT_LLM = False

LOGGER = logging.getLogger("NeMo")


@dataclass
class TensorrtLLMHostContext:
    """The host side context for TRT LLM inference."""

    executor: MPIPoolExecutor = None
    world_size: int = 1
    tokenizer: PreTrainedTokenizer = None
    max_batch_size: int = 0
    max_input_len: int = 0
    add_bos: bool = False


@dataclass
class TensorrtLLMWorkerContext:
    """The MPI worker side context for TRT LLM inference."""

    decoder: ModelRunner | ModelRunnerCpp = None
    sampling_config: SamplingConfig = None
    lora_manager: LoraManager = None
    max_batch_size: int = 0
    max_input_len: int = 0


# This is a global context that will be initialized during the model loading process as MPI worker.
tensorrt_llm_worker_context = TensorrtLLMWorkerContext()


def _load(
    tokenizer: PreTrainedTokenizer,
    engine_dir,
    lora_ckpt_list=None,
    num_beams=1,
    use_python_runtime: bool = True,
    enable_chunked_context: bool = False,
    max_tokens_in_paged_kv_cache: int = None,
    multi_block_mode: bool = False,
):
    """The impl of `load` API for on a single GPU worker."""
    try:
        tensorrt_llm.logger.set_level("info")

        engine_dir = Path(engine_dir)
        config_path = engine_dir / "config.json"
        # model_config, world_size, tp_size, pp_size, dtype, max_input_len, max_batch_size = _read_config(config_path)

        with open(config_path, "r") as f:
            config = json.load(f)

        max_batch_size = config["build_config"]["max_batch_size"]
        max_input_len = config["build_config"]["max_input_len"]
        # max_output_len = config["build_config"]["max_output_len"]
        max_beam_width = config["build_config"]["max_beam_width"]

        runtime_rank = tensorrt_llm.mpi_rank()

        if use_python_runtime:
            if enable_chunked_context:
                logging.warning("enable_chunked_context is disabled when using python runtime")
            if multi_block_mode:
                logging.warning("multi_block_mode is disabled when using python runtime")

            decoder = ModelRunner.from_dir(
                engine_dir=engine_dir,
                lora_dir=lora_ckpt_list,
                lora_ckpt_source="nemo",
                rank=runtime_rank,
                debug_mode=False,
            )
        else:
            decoder = ModelRunnerCpp.from_dir(
                engine_dir=engine_dir,
                lora_dir=lora_ckpt_list,
                lora_ckpt_source="nemo",
                rank=runtime_rank,
                max_batch_size=max_batch_size,
                max_input_len=max_input_len,
                # max_output_len=max_output_len,
                max_beam_width=max_beam_width,
                enable_chunked_context=enable_chunked_context,
                max_tokens_in_paged_kv_cache=max_tokens_in_paged_kv_cache,
                multi_block_mode=multi_block_mode,
                debug_mode=False,
            )

        sampling_config = SamplingConfig(
            end_id=tokenizer.eos_token_id,
            pad_id=tokenizer.eos_token_id,
            num_beams=num_beams,
        )

        # Initialize the global context so it can be used during `run` API.
        global tensorrt_llm_worker_context
        tensorrt_llm_worker_context.decoder = decoder
        tensorrt_llm_worker_context.sampling_config = sampling_config
        tensorrt_llm_worker_context.max_batch_size = max_batch_size
        tensorrt_llm_worker_context.max_input_len = max_input_len

    except Exception as e:
        print(e)
        raise e


def _forward(
    input_tensors: List[torch.IntTensor],
    max_output_len: int,
    top_k: int = 1,
    top_p: float = 0.0,
    temperature: float = 1.0,
    prompt_table=None,
    task_vocab_size=None,  # noqa: ARG001
    task_ids: List[int] = None,
    lora_uids: List[str] = None,
    stop_words_list=None,
    bad_words_list=None,
    no_repeat_ngram_size=None,  # noqa: ARG001
    streaming: bool = False,
    multiprocessed_env=False,
    **sampling_kwargs,
) -> Optional[torch.IntTensor]:
    """The impl of `forward` API for on a single GPU worker with tensor as IO.

    Returns:
        the output tokens tensor with shape [batch_size, num_beams, output_len].
    """
    try:
        # Loading the global context initialized from the `load` API.
        global tensorrt_llm_worker_context
        decoder = tensorrt_llm_worker_context.decoder
        assert decoder is not None, "Invalid worker context, decoder is not loaded."
        sampling_config = tensorrt_llm_worker_context.sampling_config
        max_batch_size = tensorrt_llm_worker_context.max_batch_size
        max_input_len = tensorrt_llm_worker_context.max_input_len

        batch_size = len(input_tensors)
        assert batch_size <= max_batch_size, f"batch size {batch_size} exceedng max batch size {max_batch_size}"
        input_lengths = [t.shape[0] for t in input_tensors]
        max_length = max(input_lengths)
        assert max_length <= max_input_len, f"input length {max_length} exceedng max input length {max_input_len}"
        pad_id = sampling_config.pad_id
        end_id = sampling_config.end_id
        num_beams = sampling_config.num_beams

        for k in sampling_kwargs.keys():
            if not hasattr(sampling_config, k):
                raise TypeError(f"Unknown sampling args '{k}'")

        with torch.no_grad():
            prompt_tasks = None if task_ids is None else ",".join(str(task) for task in task_ids)

            if prompt_table is not None:
                prompt_table = prompt_table.reshape(1, *prompt_table.shape)
                tmp_dir = tempfile.TemporaryDirectory()
                prompt_table_path = os.path.join(tmp_dir.name, "prompt_table.npy")
                np.save(prompt_table_path, prompt_table.cpu().float().numpy())
                prompt_table = prompt_table_path

            outputs = decoder.generate(
                input_tensors,
                max_new_tokens=max_output_len,
                end_id=end_id,
                pad_id=pad_id,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                num_beams=num_beams,
                stop_words_list=stop_words_list,
                bad_words_list=bad_words_list,
                lora_uids=lora_uids,
                prompt_table_path=prompt_table,
                prompt_table=prompt_table,
                prompt_tasks=prompt_tasks,
                streaming=streaming,
                output_sequence_lengths=True,
                return_dict=True,
                **sampling_kwargs,
            )

            torch.cuda.synchronize()

            if prompt_table is not None:
                tmp_dir.cleanup()

        runtime_rank = tensorrt_llm.mpi_rank()
        if runtime_rank == 0 or multiprocessed_env:
            return outputs
        else:
            return None

    except Exception as e:
        print(e)
        raise e


def load(
    tokenizer: PreTrainedTokenizer,
    engine_dir: str,
    lora_ckpt_list: List[str] = None,
    num_beams: int = 1,
    use_python_runtime: bool = True,
    enable_chunked_context: bool = False,
    max_tokens_in_paged_kv_cache: int = None,
    multi_block_mode: bool = False,
) -> TensorrtLLMHostContext:
    """Loaded the compiled LLM model and run it.

    It also supports running the TRT LLM model on multi-GPU.
    """
    # the parent dir of the engine_dir
    config_path = os.path.join(engine_dir, "config.json")
    with open(config_path, "r") as f:
        config = json.load(f)
    world_size = config["pretrained_config"]["mapping"]["world_size"]
    if world_size == 1:
        _load(
            tokenizer,
            engine_dir,
            lora_ckpt_list,
            num_beams,
            use_python_runtime,
            enable_chunked_context,
            max_tokens_in_paged_kv_cache,
            multi_block_mode,
        )
        executor = None
    elif tensorrt_llm.mpi_world_size() > 1:
        _load(
            tokenizer,
            engine_dir,
            lora_ckpt_list,
            num_beams,
            use_python_runtime,
            enable_chunked_context,
            max_tokens_in_paged_kv_cache,
        )
        executor = None
        tensorrt_llm.mpi_barrier()
    else:
        if not HAVE_MPI:
            raise UnavailableError(MISSING_MPI_MSG)

        executor = MPIPoolExecutor(max_workers=world_size)
        futures = []
        for _ in range(world_size):
            future = executor.submit(
                _load,
                tokenizer,
                engine_dir,
                lora_ckpt_list,
                num_beams,
                use_python_runtime,
                enable_chunked_context,
                max_tokens_in_paged_kv_cache,
            )
            futures.append(future)
        for future in futures:
            future.result()

    max_batch_size = config["build_config"]["max_batch_size"]
    max_input_len = config["build_config"]["max_input_len"]
    architectures_that_need_bos_token = [
        "GemmaForCausalLM",
        "LLaMAForCausalLM",
        "MistralForCausalLM",
        "MixtralForCausalLM",
    ]
    add_bos = config["pretrained_config"]["architecture"] in architectures_that_need_bos_token

    return TensorrtLLMHostContext(
        executor=executor,
        world_size=world_size,
        tokenizer=tokenizer,
        max_batch_size=max_batch_size,
        max_input_len=max_input_len,
        add_bos=add_bos,
    )


def forward(
    input_tensors: List[torch.IntTensor],
    max_output_len: int,
    host_context: TensorrtLLMHostContext,
    top_k: int = 1,
    top_p: float = 0.0,
    temperature: float = 1.0,
    prompt_table=None,
    task_vocab_size=None,  # noqa: ARG001
    task_ids: List[int] = None,
    lora_uids: List[str] = None,
    stop_words_list=None,
    bad_words_list=None,
    no_repeat_ngram_size=None,  # noqa: ARG001
    streaming: bool = False,
    multiprocessed_env=False,
    **sampling_kwargs,
) -> Optional[torch.IntTensor]:
    """Run the loaded model with the host_context provided from the `load` API."""
    batch_size = len(input_tensors)
    max_batch_size = host_context.max_batch_size
    assert batch_size <= max_batch_size, f"batch size {batch_size} exceedng max batch size {max_batch_size}"
    max_length = max([t.shape[0] for t in input_tensors])
    max_input_len = host_context.max_input_len
    assert max_length <= max_input_len, f"input length {max_length} exceedng max input length {max_input_len}"

    world_size = host_context.world_size
    if world_size == 1 or multiprocessed_env:
        return _forward(
            input_tensors=input_tensors,
            max_output_len=max_output_len,
            top_k=top_k,
            top_p=top_p,
            temperature=temperature,
            prompt_table=prompt_table,
            task_vocab_size=task_vocab_size,
            task_ids=task_ids,
            lora_uids=lora_uids,
            stop_words_list=stop_words_list,
            bad_words_list=bad_words_list,
            no_repeat_ngram_size=no_repeat_ngram_size,
            streaming=streaming,
            multiprocessed_env=multiprocessed_env,
            **sampling_kwargs,
        )
    else:
        executor = host_context.executor
        futures = []
        for _ in range(world_size):
            future = executor.submit(
                _forward,
                input_tensors=input_tensors,
                max_output_len=max_output_len,
                top_k=top_k,
                top_p=top_p,
                temperature=temperature,
                prompt_table=prompt_table,
                task_vocab_size=task_vocab_size,
                task_ids=task_ids,
                lora_uids=lora_uids,
                stop_words_list=stop_words_list,
                bad_words_list=bad_words_list,
                no_repeat_ngram_size=no_repeat_ngram_size,
                streaming=streaming,
                **sampling_kwargs,
            )
            futures.append(future)
        for future in futures:
            result = future.result()
            if result is not None:
                return result

        raise RuntimeError("Internal error")


def maybe_cast_to_trt_dtype(dtype):
    """Cast input dtype to TensorRT dtype if applicable.

    Args:
        dtype: Input dtype (torch.dtype or trt.DataType)

    Returns:
        trt.DataType: Corresponding TensorRT dtype
    """
    if not HAVE_TRT:
        raise UnavailableError(MISSING_TENSORRT_MSG)

    if isinstance(dtype, trt.DataType):
        return dtype
    elif isinstance(dtype, torch.dtype):
        return tensorrt_llm._utils.torch_dtype_to_trt(dtype)
    else:
        raise NotImplementedError(f"Expects the type to be a tensorrt.DataType or torch.dtype, but got {type(dtype)=}")


def unload_engine():
    """Deletes the ModelRunner which should free up device memory."""
    global tensorrt_llm_worker_context
    decoder = tensorrt_llm_worker_context.decoder
    if not isinstance(decoder, ModelRunner):
        raise ValueError(
            f"unload_engine is only supported with ModelRunner, but export has been configured with {type(decoder)=}"
        )

    logging.info("Unloading engine...")
    del tensorrt_llm_worker_context.decoder
    tensorrt_llm_worker_context.decoder = None
    logging.info("Engine unloaded!")


def prepare_input_tensors(
    input_texts: List[str],
    host_context: TensorrtLLMHostContext,
    prompt_table=None,
    task_vtoken_counts: List[int] = None,
    task_ids: List[int] = None,
):
    """Prepare input tensors from text input.

    Args:
        input_texts: List of input text strings
        host_context: Context containing tokenizer and configuration
        prompt_table: a lookup table containing trained embeddings for vtoken used in p-tuning
        task_vtoken_counts: Optional list of vtoken counts per task
        task_ids: Optional list of task IDs

    Returns:
        dict: Prepared input tensors for model
    """
    tokenizer = host_context.tokenizer

    if host_context.add_bos:
        bos_tokens = [tokenizer.bos_token_id]
    else:
        bos_tokens = []

    input_tokens = [bos_tokens + tokenizer.encode(t) for t in input_texts]

    # If p-tuning is used, we need to prepend vtokens to each input.
    if prompt_table is not None:
        # Go over the tokenized prompts and prepend vtokens.
        # The number of vtokens could be different for each task.
        for prompt_index in range(len(input_texts)):
            # Find out the number of vtokens to generate
            task_id = task_ids[prompt_index]
            num_vtokens = task_vtoken_counts[task_id]

            # Create a tensor with vtokens, e.g. 32000, 32001, 32002... when vocab_size=32000
            # TRT-LLM will convert each vtoken into its corresponding embedding row from the prompt table.
            vocab_size = tokenizer.vocab_size
            vtokens = list(range(vocab_size, vocab_size + num_vtokens))

            # Concatenate the vtokens with the real tokens
            real_tokens = input_tokens[prompt_index]
            input_tokens[prompt_index] = vtokens + real_tokens

    # Convert input token lists to tensors
    input_tensors = [torch.IntTensor(token_list) for token_list in input_tokens]

    return input_tensors


def generate(
    input_texts: List[str],
    max_output_len: int,
    host_context: TensorrtLLMHostContext,
    top_k: int = 1,
    top_p: float = 0.0,
    temperature: float = 1.0,
    prompt_table=None,
    task_vocab_size=None,  # noqa: ARG001
    task_vtoken_counts: List[int] = None,
    task_ids: List[int] = None,
    lora_uids: List[str] = None,
    stop_words_list=None,
    bad_words_list=None,
    no_repeat_ngram_size=None,  # noqa: ARG001
    streaming: bool = False,
    output_log_probs=False,  # noqa: ARG001
    multiprocessed_env=False,
    output_context_logits=False,
    output_generation_logits=False,
    **sampling_kwargs,
) -> Optional[List[List[str]]]:
    """Generate the output sequence from the input sequence.

    Returns a 2D string list with shape [batch_size, num_beams].
    """
    tokenizer = host_context.tokenizer
    input_tensors = prepare_input_tensors(input_texts, host_context, prompt_table, task_vtoken_counts, task_ids)

    stop_words_list_tensors = None
    if stop_words_list is not None:
        stop_words_arrays = to_word_list_format(stop_words_list, tokenizer)
        stop_words_list_tensors = (
            torch.Tensor(stop_words_arrays).to(torch.int32).to(torch.cuda.current_device()).contiguous()
        )

    bad_words_list_tensors = None
    if bad_words_list is not None:
        bad_words_arrays = to_word_list_format(bad_words_list, tokenizer)
        bad_words_list_tensors = (
            torch.Tensor(bad_words_arrays).to(torch.int32).to(torch.cuda.current_device()).contiguous()
        )

    if no_repeat_ngram_size is not None:
        no_repeat_ngram_size = torch.IntTensor(no_repeat_ngram_size).to(torch.cuda.current_device())

    outputs = forward(
        input_tensors=input_tensors,
        max_output_len=max_output_len,
        host_context=host_context,
        top_k=top_k,
        top_p=top_p,
        temperature=temperature,
        prompt_table=prompt_table,
        task_vocab_size=task_vocab_size,
        task_ids=task_ids,
        lora_uids=lora_uids,
        stop_words_list=stop_words_list_tensors,
        bad_words_list=bad_words_list_tensors,
        no_repeat_ngram_size=no_repeat_ngram_size,
        streaming=False,
        output_log_probs=output_log_probs,
        multiprocessed_env=multiprocessed_env,
        **sampling_kwargs,
    )

    assert outputs is not None
    if tensorrt_llm.mpi_rank() != 0:
        return None

    output_ids = outputs["output_ids"]
    sequence_lengths = outputs["sequence_lengths"]
    input_lengths = [t.shape[0] for t in input_tensors]

    output_lines_list = [
        tokenizer.batch_decode(output_ids[b, :, input_lengths[b] : sequence_lengths[b][0]])
        for b in range(output_ids.shape[0])
    ]

    if output_generation_logits:
        return output_lines_list, outputs["generation_logits"]
    elif output_context_logits:
        return output_lines_list, outputs["context_logits"]
    return output_lines_list


def unload(host_context: TensorrtLLMHostContext):
    """Frees the GPU resource from the TensorrtLLMHostContext and reset the host_context."""
    if host_context.executor is not None:
        host_context.executor.shutdown(wait=True)
        host_context.executor = None
        return

    global tensorrt_llm_worker_context
    tensorrt_llm_worker_context.decoder = None
    tensorrt_llm_worker_context = TensorrtLLMWorkerContext()


def to_word_list_format(
    word_dict: List[List[str]],
    tokenizer=None,
    ref_str="<extra_id_1>",
):
    """Format of word_dict.

    len(word_dict) should be same to batch_size
    word_dict[i] means the words for batch i
    len(word_dict[i]) must be 1, which means it only contains 1 string
    This string can contains several sentences and split by ",".
    For example, if word_dict[2] = " I am happy, I am sad", then this function will return
    the ids for two short sentences " I am happy" and " I am sad".
    """
    assert tokenizer is not None, "need to set tokenizer"

    flat_ids = []
    offsets = []
    # The encoding of a single word can't always be trusted. See
    #   https://github.com/NVIDIA/NeMo/blob/bb575b72fd0be51ae10cc77d9f89ddb9e9d3b96d/nemo/collections/nlp/modules/common/text_generation_strategy.py#L229  # pylint: disable=C0301
    ids_ref = tokenizer.encode(ref_str)
    for word_dict_item in word_dict:
        item_flat_ids = []
        item_offsets = []

        if isinstance(word_dict_item[0], bytes):
            word_dict_item = [word_dict_item[0].decode()]

        words = list(csv.reader(word_dict_item))[0]
        for word in words:
            ids = tokenizer.encode(f"{ref_str}{word}")
            if ids[0 : len(ids_ref)] == ids_ref:
                # It worked! We can obtain the token(s) associated to `word` by stripping the prefix tokens.
                ids = ids[len(ids_ref) :]
            else:
                # Unfortunately the prefix was merged with `word`. We could try with a different prefix, but
                # for now we just use the basic encoding since this should be a very rare edge case.
                ids = tokenizer.encode(word)
                logging.warning(f"The encoding of word '{word}' into tokens {ids} might be incorrect")

            if len(ids) == 0:
                continue

            item_flat_ids += ids
            item_offsets.append(len(ids))

        flat_ids.append(np.array(item_flat_ids))
        offsets.append(np.cumsum(np.array(item_offsets)))

    pad_to = max(1, max(len(ids) for ids in flat_ids))

    for i, (ids, offs) in enumerate(zip(flat_ids, offsets)):
        flat_ids[i] = np.pad(ids, (0, pad_to - len(ids)), constant_values=0)
        offsets[i] = np.pad(offs, (0, pad_to - len(offs)), constant_values=-1)

    return np.array([flat_ids, offsets], dtype="int32").transpose((1, 0, 2))
