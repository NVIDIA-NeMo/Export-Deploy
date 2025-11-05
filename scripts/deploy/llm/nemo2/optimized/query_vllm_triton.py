# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
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

import argparse
import sys
import typing

import numpy as np

from nemo_deploy.llm import NemoQueryvLLM


def get_args(argv):
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Sends a single query to an LLM hosted on a Triton server.",
    )
    parser.add_argument("-u", "--url", default="0.0.0.0", type=str, help="url for the triton server")
    parser.add_argument("-mn", "--model-name", required=True, type=str, help="Name of the triton model")
    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument("-p", "--prompt", required=False, type=str, help="Prompt")
    prompt_group.add_argument(
        "-pf",
        "--prompt-file",
        required=False,
        type=str,
        help="File to read the prompt from",
    )
    parser.add_argument(
        "-mat",
        "--max-tokens",
        default=16,
        type=int,
        help="Max output token length",
    )
    parser.add_argument(
        "-mit",
        "--min-tokens",
        default=0,
        type=int,
        help="Min output token length",
    )
    parser.add_argument(
        "-nlp",
        "--n-log-probs",
        default=None,
        type=int,
        help="Number of log probabilities to return per output token.",
    )
    parser.add_argument(
        "-nplp",
        "--n-prompt-log-probs",
        default=None,
        type=int,
        help="Number of log probabilities to return per prompt token.",
    )
    parser.add_argument(
        "-s",
        "--seed",
        default=None,
        type=int,
        help="Random seed to use for the generation.",
    )
    parser.add_argument("-tk", "--top_k", default=1, type=int, help="top_k")
    parser.add_argument("-tpp", "--top_p", default=0.1, type=float, help="top_p")
    parser.add_argument("-t", "--temperature", default=1.0, type=float, help="temperature")
    parser.add_argument(
        "-lt",
        "--lora-task-uids",
        default=None,
        type=str,
        nargs="+",
        help="The list of LoRA task uids; use -1 to disable the LoRA module",
    )
    parser.add_argument(
        "-it",
        "--init-timeout",
        default=60.0,
        type=float,
        help="init timeout for the triton server",
    )

    args = parser.parse_args(argv)
    return args


def str_list2numpy(str_list: typing.List[str]) -> np.ndarray:
    str_ndarray = np.array(str_list)[..., np.newaxis]
    return np.char.encode(str_ndarray, "utf-8")


def query(argv):
    args = get_args(argv)

    if args.prompt_file is not None:
        with open(args.prompt_file, "r") as f:
            args.prompt = f.read()

    nq = NemoQueryvLLM(url=args.url, model_name=args.model_name)
    outputs = nq.query_llm(
        prompts=[args.prompt],
        max_tokens=args.max_tokens,
        min_tokens=args.min_tokens,
        n_log_probs=args.n_log_probs,
        n_prompt_log_probs=args.n_prompt_log_probs,
        seed=args.seed,
        top_k=args.top_k,
        top_p=args.top_p,
        temperature=args.temperature,
        init_timeout=args.init_timeout,
    )

    print(outputs)


if __name__ == "__main__":
    query(sys.argv[1:])
