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


import argparse
import json
import time

from nemo_deploy.llm.megatronllm_deployable import MegatronLLMDeployable

try:
    from nemo_deploy import DeployPyTriton
    from nemo_deploy.llm import NemoQueryLLMPyTorch
except Exception:  # noqa: BLE001
    pass


def get_accuracy_with_lambada(model, nq, lora_uids, test_data_path=None):
    # lambada dataset based accuracy test, which includes more than 5000 sentences.
    # Use generated last token with original text's last token for accuracy comparison.
    # It generates a CSV file for text comparison detail.

    if test_data_path is None:
        raise Exception("test_data_path cannot be None.")

    correct = 0
    deployed_correct = 0
    correct_relaxed = 0
    deployed_correct_relaxed = 0
    all_expected_outputs = []
    all_outputs = []

    with open(test_data_path, "r") as file:
        records = json.load(file)

        eval_start = time.perf_counter()
        for record in records:
            prompt = record["text_before_last_word"]
            expected_output = record["last_word"].strip().lower()
            output = model.forward(
                input_texts=[prompt],
                max_output_len=1,
                top_k=1,
                top_p=0,
                temperature=0.1,
                lora_uids=lora_uids,
            )
            output = output[0][0].strip().lower()

            all_expected_outputs.append(expected_output)
            all_outputs.append(output)

            if expected_output == output:
                correct += 1

            if expected_output == output or output.startswith(expected_output) or expected_output.startswith(output):
                if len(output) == 1 and len(expected_output) > 1:
                    continue
                correct_relaxed += 1

            if nq is not None:
                deployed_output = nq.query_llm(
                    prompts=[prompt],
                    max_output_len=1,
                    top_k=1,
                    top_p=0,
                    temperature=0.1,
                )
                deployed_output = deployed_output[0][0].strip().lower()

                if expected_output == deployed_output:
                    deployed_correct += 1

                if (
                    expected_output == deployed_output
                    or deployed_output.startswith(expected_output)
                    or expected_output.startswith(deployed_output)
                ):
                    if len(deployed_output) == 1 and len(expected_output) > 1:
                        continue
                    deployed_correct_relaxed += 1
        eval_end = time.perf_counter()

    accuracy = correct / len(all_expected_outputs)
    accuracy_relaxed = correct_relaxed / len(all_expected_outputs)
    deployed_accuracy = deployed_correct / len(all_expected_outputs)
    deployed_accuracy_relaxed = deployed_correct_relaxed / len(all_expected_outputs)
    evaluation_time = eval_end - eval_start

    return (
        accuracy,
        accuracy_relaxed,
        deployed_accuracy,
        deployed_accuracy_relaxed,
        evaluation_time,
    )


def run_in_framework_inference(
    model_name,
    prompt,
    checkpoint_path,
    n_gpu=1,
    max_batch_size=None,
    max_input_len=None,
    max_output_len=None,
):
    model = MegatronLLMDeployable(
        megatron_checkpoint_filepath=checkpoint_path,
        num_devices=n_gpu,
        num_nodes=1,
    )
    nm = DeployPyTriton(
        model=model,
        triton_model_name=model_name,
        http_port=8000,
    )
    nm.deploy()
    nm.run()
    nq = NemoQueryLLMPyTorch(url="localhost:8000", model_name=model_name)

    output_deployed = nq.query_llm(
        prompts=prompt,
    )

    print("Output: ", output_deployed)

    nm.stop()

    return None, None, None, None, None


def get_args():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Deploy nemo models to Triton and benchmark the models",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
    )
    parser.add_argument(
        "--model_type",
        type=str,
        required=False,
    )
    parser.add_argument(
        "--min_gpus",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--max_gpus",
        type=int,
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="/tmp/nemo_checkpoint/",
        required=False,
    )
    parser.add_argument(
        "--max_batch_size",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--max_input_len",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--max_output_len",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--max_num_tokens",
        type=int,
    )
    parser.add_argument(
        "--lora_checkpoint",
        type=str,
    )
    parser.add_argument(
        "--lora",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--tp_size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--pp_size",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=1,
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--run_accuracy",
        type=str,
        default="False",
    )
    parser.add_argument(
        "--accuracy_threshold",
        type=float,
        default=0.5,
    )
    parser.add_argument(
        "--test_deployment",
        type=str,
        default="False",
    )
    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
    )
    parser.add_argument(
        "--test_data_path",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--save_engine",
        type=str,
        default="False",
    )

    return parser.parse_args()


def run_inference_tests(args):
    if args.test_deployment == "True":
        args.test_deployment = True
    else:
        args.test_deployment = False

    if args.save_engine == "True":
        args.save_engine = True
    else:
        args.save_engine = False

    if args.run_accuracy == "True":
        args.run_accuracy = True
    else:
        args.run_accuracy = False

    if args.run_accuracy and args.test_data_path is None:
        raise Exception("test_data_path param cannot be None.")

    result_dic = {}

    prompt_template = ["The capital of France is", "Largest animal in the sea is"]
    n_gpus = args.min_gpus
    if args.max_gpus is None:
        args.max_gpus = args.min_gpus

    while n_gpus <= args.max_gpus:
        result_dic[n_gpus] = run_in_framework_inference(
            model_name=args.model_name,
            prompt=prompt_template,
            checkpoint_path=args.checkpoint_dir,
            n_gpu=n_gpus,
            max_batch_size=args.max_batch_size,
            max_input_len=args.max_input_len,
            max_output_len=args.max_output_len,
        )

        n_gpus = n_gpus * 2

    test_result = "PASS"
    print_separator = False
    print("============= Test Summary ============")
    for i, results in result_dic.items():
        if not results[0] is None and not results[1] is None:
            if print_separator:
                print("---------------------------------------")
            print(
                "Number of GPUS:                  {}\n"
                "Model Accuracy:                  {:.4f}\n"
                "Relaxed Model Accuracy:          {:.4f}\n"
                "Deployed Model Accuracy:         {:.4f}\n"
                "Deployed Relaxed Model Accuracy: {:.4f}\n"
                "Evaluation Time [s]:             {:.2f}".format(i, *results)
            )
            print_separator = True
            if results[1] < args.accuracy_threshold:
                test_result = "FAIL"

    print("=======================================")
    print("TEST: " + test_result)
    if test_result == "FAIL":
        raise Exception(f"Model accuracy is below {args.accuracy_threshold}")


if __name__ == "__main__":
    args = get_args()
    run_inference_tests(args)
