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
import logging

import requests

LOGGER = logging.getLogger("NeMo")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Query a deployed HuggingFace model using Ray"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host address of the Ray Serve server",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=1024,
        help="Port number of the Ray Serve server",
    )
    parser.add_argument(
        "--model_id",
        type=str,
        default="nemo-model",
        help="Identifier for the model in the API responses",
    )
    return parser.parse_args()


def test_completions_endpoint(base_url: str, model_id: str) -> None:
    """Test the completions endpoint."""
    url = f"{base_url}/v1/completions/"
    payload = {
        "model": model_id,
        "prompt": r"Question: Jen and Tyler are gymnasts practicing flips. Jen is practicing the triple-flip while Tyler is practicing the double-flip. Jen did sixteen triple-flips during practice. Tyler flipped in the air half the number of times Jen did. How many double-flips did Tyler do?\nAnswer: Jen did 16 triple-flips, so she did 16 * 3 = <<16*3=48>>48 flips.\nTyler did half the number of flips, so he did 48 / 2 = <<48/2=24>>24 flips.\nA double flip has two flips, so Tyler did 24 / 2 = <<24/2=12>>12 double-flips.\n#### 12\n\nQuestion: Four people in a law firm are planning a party. Mary will buy a platter of pasta for $20 and a loaf of bread for $2. Elle and Andrea will split the cost for buying 4 cans of soda which cost $1.50 each, and chicken wings for $10. Joe will buy a cake that costs $5. How much more will Mary spend than the rest of the firm put together?\nAnswer: Mary will spend $20 + $2 = $<<20+2=22>>22.\nElle and Andrea will spend $1.5 x 4 = $<<1.5*4=6>>6 for the soda.\nElle and Andrea will spend $6 + $10 = $<<6+10=16>>16 for the soda and chicken wings.\nElle, Andrea, and Joe together will spend $16 + $5 = $<<16+5=21>>21.\nSo, Mary will spend $22 - $21 = $<<22-21=1>>1 more than all of them combined.\n#### 1\n\nQuestion: A charcoal grill burns fifteen coals to ash every twenty minutes of grilling. The grill ran for long enough to burn three bags of coals. Each bag of coal contains 60 coals. How long did the grill run?\nAnswer: The grill burned 3 * 60 = <<3*60=180>>180 coals.\nIt takes 20 minutes to burn 15 coals, so the grill ran for 180 / 15 * 20 = <<180/15*20=240>>240 minutes.\n#### 240\n\nQuestion: A bear is preparing to hibernate for the winter and needs to gain 1000 pounds. At the end of summer, the bear feasts on berries and small woodland animals. During autumn, it devours acorns and salmon. It gained a fifth of the weight it needed from berries during summer, and during autumn, it gained twice that amount from acorns. Salmon made up half of the remaining weight it had needed to gain. How many pounds did it gain eating small animals?\nAnswer: The bear gained 1 / 5 * 1000 = <<1/5*1000=200>>200 pounds from berries.\nIt gained 2 * 200 = <<2*200=400>>400 pounds from acorns.\nIt still needed 1000 - 200 - 400 = <<1000-200-400=400>>400 pounds.\nThus, it gained 400 / 2 = <<400/2=200>>200 pounds from salmon.\nTherefore, the bear gained 400 - 200 = <<400-200=200>>200 pounds from small animals.\n#### 200\n\nQuestion: Brendan can cut 8 yards of grass per day, he bought a lawnmower and it helped him to cut more yards by Fifty percent per day. How many yards will Brendan be able to cut after a week?\nAnswer: The additional yard Brendan can cut after buying the lawnmower is 8 x 0.50 = <<8*0.50=4>>4 yards.\nSo, the total yards he can cut with the lawnmower is 8 + 4 = <<8+4=12>>12.\nTherefore, the total number of yards he can cut in a week is 12 x 7 = <<12*7=84>>84 yards.\n#### 84\n\nQuestion: Janet’s ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?\nAnswer:",
        "max_tokens": 250,
        "temperature": 0.999,
        "logprobs": 1,
    }

    LOGGER.info(f"Testing completions endpoint at {url}")
    response = requests.post(url, json=payload)
    LOGGER.info(f"Response status code: {response.status_code}")
    if response.status_code == 200:
        LOGGER.info(f"Response: {json.dumps(response.json(), indent=2)}")
    else:
        LOGGER.error(f"Error: {response.text}")


def test_chat_completions_endpoint(base_url: str, model_id: str) -> None:
    """Test the chat completions endpoint."""
    url = f"{base_url}/v1/chat/completions/"
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": "Hello, how are you doing?"}],
        "max_tokens": 50,
        "temperature": 0.7,
        "logprobs": 1,
    }

    LOGGER.info(f"Testing chat completions endpoint at {url}")
    response = requests.post(url, json=payload)
    LOGGER.info(f"Response status code: {response.status_code}")
    if response.status_code == 200:
        LOGGER.info(f"Response: {json.dumps(response.json(), indent=2)}")
    else:
        LOGGER.error(f"Error: {response.text}")


def test_models_endpoint(base_url: str) -> None:
    """Test the models endpoint."""
    url = f"{base_url}/v1/models"

    LOGGER.info(f"Testing models endpoint at {url}")
    response = requests.get(url)
    LOGGER.info(f"Response status code: {response.status_code}")
    if response.status_code == 200:
        LOGGER.info(f"Response: {json.dumps(response.json(), indent=2)}")
    else:
        LOGGER.error(f"Error: {response.text}")


def test_health_endpoint(base_url: str) -> None:
    """Test the health endpoint."""
    url = f"{base_url}/v1/health"

    LOGGER.info(f"Testing health endpoint at {url}")
    response = requests.get(url)
    LOGGER.info(f"Response status code: {response.status_code}")
    if response.status_code == 200:
        LOGGER.info(f"Response: {json.dumps(response.json(), indent=2)}")
    else:
        LOGGER.error(f"Error: {response.text}")


def main():
    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    args = parse_args()
    base_url = f"http://{args.host}:{args.port}"

    LOGGER.info(f"Testing endpoints for model {args.model_id} at {base_url}")

    # Test all endpoints
    test_completions_endpoint(base_url, args.model_id)
    test_health_endpoint(base_url)
    test_models_endpoint(base_url)


if __name__ == "__main__":
    main()
