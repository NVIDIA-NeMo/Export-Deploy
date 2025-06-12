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

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from nemo_deploy.nlp.query_llm import NemoQueryLLM, NemoQueryLLMBase, NemoQueryLLMHF, NemoQueryLLMPyTorch


class TestNemoQueryLLMBase:
    def test_base_initialization(self):
        url = "localhost:8000"
        model_name = "test-model"
        query = NemoQueryLLMBase(url=url, model_name=model_name)
        assert query.url == url
        assert query.model_name == model_name


class TestNemoQueryLLMPyTorch:
    @pytest.fixture
    def query(self):
        return NemoQueryLLMPyTorch(url="localhost:8000", model_name="test-model")

    def test_initialization(self, query):
        assert isinstance(query, NemoQueryLLMBase)
        assert query.url == "localhost:8000"
        assert query.model_name == "test-model"

    @patch("nemo_deploy.nlp.query_llm.ModelClient")
    def test_query_llm_basic(self, mock_client, query):
        # Setup mock
        mock_instance = MagicMock()
        mock_client.return_value.__enter__.return_value = mock_instance
        mock_instance.infer_batch.return_value = {"sentences": np.array([b"test response"])}
        mock_instance.model_config.outputs = [MagicMock(dtype=np.bytes_)]

        # Test basic query
        response = query.query_llm(prompts=["test prompt"], max_length=100, temperature=0.7, top_k=1, top_p=0.9)

        assert isinstance(response, dict)
        assert "choices" in response
        assert response["choices"][0]["text"] == "test response"

    @patch("nemo_deploy.nlp.query_llm.ModelClient")
    def test_query_llm_with_logprobs(self, mock_client, query):
        # Setup mock
        mock_instance = MagicMock()
        mock_client.return_value.__enter__.return_value = mock_instance
        mock_instance.infer_batch.return_value = {
            "sentences": np.array([b"test response"]),
            "log_probs": np.array([0.1, 0.2, 0.3]),
        }
        mock_instance.model_config.outputs = [MagicMock(dtype=np.bytes_)]

        # Test query with logprobs
        response = query.query_llm(prompts=["test prompt"], max_length=100, compute_logprob=True)

        assert "logprobs" in response["choices"][0]
        assert "token_logprobs" in response["choices"][0]["logprobs"]

    @patch("nemo_deploy.nlp.query_llm.ModelClient")
    def test_query_llm_unknown_output_keyword(self, mock_client, query):
        # Setup mock for unknown output keyword case
        mock_instance = MagicMock()
        mock_client.return_value.__enter__.return_value = mock_instance
        mock_instance.infer_batch.return_value = {"unknown_key": np.array([b"test response"])}
        mock_instance.model_config.outputs = [MagicMock(dtype=np.bytes_)]

        # Test query with unknown output keyword
        response = query.query_llm(prompts=["test prompt"])

        assert response == "Unknown output keyword."

    @patch("nemo_deploy.nlp.query_llm.ModelClient")
    def test_query_llm_non_bytes_output(self, mock_client, query):
        # Setup mock for non-bytes output type
        mock_instance = MagicMock()
        mock_client.return_value.__enter__.return_value = mock_instance
        mock_instance.infer_batch.return_value = {"sentences": np.array(["test response"])}
        mock_instance.model_config.outputs = [MagicMock(dtype=np.float32)]

        # Test query with non-bytes output
        response = query.query_llm(prompts=["test prompt"])

        assert response == np.array(["test response"])

    @patch("nemo_deploy.nlp.query_llm.ModelClient")
    def test_query_llm_with_top_logprobs(self, mock_client, query):
        # Setup mock for top logprobs
        mock_instance = MagicMock()
        mock_client.return_value.__enter__.return_value = mock_instance
        mock_instance.infer_batch.return_value = {
            "sentences": np.array([b"test response"]),
            "log_probs": np.array([0.1, 0.2, 0.3]),
            "top_logprobs": np.array([['[{"token": "test", "prob": 0.9}]']]),
        }
        mock_instance.model_config.outputs = [MagicMock(dtype=np.bytes_)]

        # Test query with top logprobs
        response = query.query_llm(prompts=["test prompt"], compute_logprob=True, n_top_logprobs=5)

        assert "logprobs" in response["choices"][0]
        assert "top_logprobs" in response["choices"][0]["logprobs"]

    @patch("nemo_deploy.nlp.query_llm.ModelClient")
    def test_query_llm_all_parameters(self, mock_client, query):
        # Setup mock
        mock_instance = MagicMock()
        mock_client.return_value.__enter__.return_value = mock_instance
        mock_instance.infer_batch.return_value = {"sentences": np.array([b"test response"])}
        mock_instance.model_config.outputs = [MagicMock(dtype=np.bytes_)]

        # Test query with all parameters
        response = query.query_llm(
            prompts=["test prompt"],
            use_greedy=True,
            temperature=0.8,
            top_k=5,
            top_p=0.95,
            repetition_penalty=1.1,
            add_BOS=True,
            all_probs=True,
            compute_logprob=True,
            end_strings=["</s>"],
            min_length=10,
            max_length=100,
            apply_chat_template=True,
            n_top_logprobs=3,
            init_timeout=30.0,
            echo=True,
        )

        assert isinstance(response, dict)
        assert "choices" in response


class TestNemoQueryLLMHF:
    @pytest.fixture
    def query(self):
        return NemoQueryLLMHF(url="localhost:8000", model_name="test-model")

    def test_initialization(self, query):
        assert isinstance(query, NemoQueryLLMBase)
        assert query.url == "localhost:8000"
        assert query.model_name == "test-model"

    @patch("nemo_deploy.nlp.query_llm.ModelClient")
    def test_query_llm_basic(self, mock_client, query):
        # Setup mock
        mock_instance = MagicMock()
        mock_client.return_value.__enter__.return_value = mock_instance
        mock_instance.infer_batch.return_value = {"sentences": np.array([b"test response"])}
        mock_instance.model_config.outputs = [MagicMock(dtype=np.bytes_)]

        # Test basic query
        response = query.query_llm(prompts=["test prompt"], max_length=100, temperature=0.7, top_k=1, top_p=0.9)

        assert isinstance(response, dict)
        assert "choices" in response
        assert response["choices"][0]["text"] == "test response"

    @patch("nemo_deploy.nlp.query_llm.ModelClient")
    def test_query_llm_with_logits(self, mock_client, query):
        # Setup mock
        mock_instance = MagicMock()
        mock_client.return_value.__enter__.return_value = mock_instance
        mock_instance.infer_batch.return_value = {
            "sentences": np.array([b"test response"]),
            "logits": np.array([[0.1, 0.2, 0.3]]),
        }
        mock_instance.model_config.outputs = [MagicMock(dtype=np.bytes_)]

        # Test query with logits
        response = query.query_llm(prompts=["test prompt"], max_length=100, output_logits=True)

        assert "logits" in response

    @patch("nemo_deploy.nlp.query_llm.ModelClient")
    def test_query_llm_unknown_output_keyword(self, mock_client, query):
        # Setup mock for unknown output keyword case
        mock_instance = MagicMock()
        mock_client.return_value.__enter__.return_value = mock_instance
        mock_instance.infer_batch.return_value = {"unknown_key": np.array([b"test response"])}
        mock_instance.model_config.outputs = [MagicMock(dtype=np.bytes_)]

        # Test query with unknown output keyword
        response = query.query_llm(prompts=["test prompt"])

        assert response == "Unknown output keyword."

    @patch("nemo_deploy.nlp.query_llm.ModelClient")
    def test_query_llm_non_bytes_output(self, mock_client, query):
        # Setup mock for non-bytes output type
        mock_instance = MagicMock()
        mock_client.return_value.__enter__.return_value = mock_instance
        mock_instance.infer_batch.return_value = {"sentences": np.array(["test response"])}
        mock_instance.model_config.outputs = [MagicMock(dtype=np.float32)]

        # Test query with non-bytes output
        response = query.query_llm(prompts=["test prompt"])

        assert response == np.array(["test response"])

    @patch("nemo_deploy.nlp.query_llm.ModelClient")
    def test_query_llm_with_scores(self, mock_client, query):
        # Setup mock for output_scores
        mock_instance = MagicMock()
        mock_client.return_value.__enter__.return_value = mock_instance
        mock_instance.infer_batch.return_value = {
            "sentences": np.array([b"test response"]),
            "scores": np.array([[0.5, 0.3, 0.2]]),
        }
        mock_instance.model_config.outputs = [MagicMock(dtype=np.bytes_)]

        # Test query with scores
        response = query.query_llm(prompts=["test prompt"], max_length=100, output_scores=True)

        assert "scores" in response

    @patch("nemo_deploy.nlp.query_llm.ModelClient")
    def test_query_llm_all_parameters(self, mock_client, query):
        # Setup mock
        mock_instance = MagicMock()
        mock_client.return_value.__enter__.return_value = mock_instance
        mock_instance.infer_batch.return_value = {"sentences": np.array([b"test response"])}
        mock_instance.model_config.outputs = [MagicMock(dtype=np.bytes_)]

        # Test query with all parameters
        response = query.query_llm(
            prompts=["test prompt"],
            use_greedy=True,
            temperature=0.8,
            top_k=5,
            top_p=0.95,
            repetition_penalty=1.1,
            add_BOS=True,
            all_probs=True,
            output_logits=True,
            output_scores=True,
            end_strings=["</s>"],
            min_length=10,
            max_length=100,
            init_timeout=30.0,
        )

        assert isinstance(response, dict)
        assert "choices" in response


class TestNemoQueryLLM:
    @pytest.fixture
    def query(self):
        return NemoQueryLLM(url="localhost:8000", model_name="test-model")

    def test_initialization(self, query):
        assert isinstance(query, NemoQueryLLMBase)
        assert query.url == "localhost:8000"
        assert query.model_name == "test-model"

    @patch("nemo_deploy.nlp.query_llm.ModelClient")
    def test_query_llm_basic(self, mock_client, query):
        # Setup mock
        mock_instance = MagicMock()
        mock_client.return_value.__enter__.return_value = mock_instance
        mock_instance.infer_batch.return_value = {"outputs": np.array([b"test response"])}
        mock_instance.model_config.outputs = [MagicMock(dtype=np.bytes_)]

        # Test basic query
        response = query.query_llm(prompts=["test prompt"], max_output_len=100, temperature=0.7, top_k=1, top_p=0.9)

        assert isinstance(response[0], str)
        assert response[0] == "test response"

    @patch("nemo_deploy.nlp.query_llm.ModelClient")
    def test_query_llm_openai_format(self, mock_client, query):
        # Setup mock
        mock_instance = MagicMock()
        mock_client.return_value.__enter__.return_value = mock_instance
        mock_instance.infer_batch.return_value = {"outputs": np.array([b"test response"])}
        mock_instance.model_config.outputs = [MagicMock(dtype=np.bytes_)]

        # Test query with OpenAI format
        response = query.query_llm(prompts=["test prompt"], max_output_len=100, openai_format_response=True)

        assert isinstance(response, dict)
        assert "choices" in response
        assert response["choices"][0]["text"] == "test response"

    @patch("nemo_deploy.nlp.query_llm.DecoupledModelClient")
    def test_query_llm_streaming(self, mock_client, query):
        # Setup mock
        mock_instance = MagicMock()
        mock_client.return_value.__enter__.return_value = mock_instance
        mock_instance.infer_batch.return_value = [
            {"outputs": np.array([b"test"])},
            {"outputs": np.array([b" response"])},
        ]
        mock_instance.model_config.outputs = [MagicMock(dtype=np.bytes_)]

        # Test streaming query
        responses = list(query.query_llm_streaming(prompts=["test prompt"], max_output_len=100))

        assert len(responses) == 2
        assert responses[0] == "test"
        assert responses[1] == " response"

    @patch("nemo_deploy.nlp.query_llm.ModelClient")
    def test_query_llm_with_stop_words(self, mock_client, query):
        # Setup mock
        mock_instance = MagicMock()
        mock_client.return_value.__enter__.return_value = mock_instance
        mock_instance.infer_batch.return_value = {"outputs": np.array([b"test response"])}
        mock_instance.model_config.outputs = [MagicMock(dtype=np.bytes_)]

        # Test query with stop words
        response = query.query_llm(prompts=["test prompt"], max_output_len=100, stop_words_list=["stop"])

        assert isinstance(response[0], str)
        assert response[0] == "test response"

    @patch("nemo_deploy.nlp.query_llm.ModelClient")
    def test_query_llm_with_bad_words(self, mock_client, query):
        # Setup mock
        mock_instance = MagicMock()
        mock_client.return_value.__enter__.return_value = mock_instance
        mock_instance.infer_batch.return_value = {"outputs": np.array([b"test response"])}
        mock_instance.model_config.outputs = [MagicMock(dtype=np.bytes_)]

        # Test query with bad words
        response = query.query_llm(prompts=["test prompt"], max_output_len=100, bad_words_list=["bad"])

        assert isinstance(response[0], str)
        assert response[0] == "test response"

    @patch("nemo_deploy.nlp.query_llm.ModelClient")
    def test_query_llm_unknown_output_keyword(self, mock_client, query):
        # Setup mock for unknown output keyword case
        mock_instance = MagicMock()
        mock_client.return_value.__enter__.return_value = mock_instance
        mock_instance.infer_batch.return_value = {"unknown_key": np.array([b"test response"])}
        mock_instance.model_config.outputs = [MagicMock(dtype=np.bytes_)]

        # Test query with unknown output keyword
        response = query.query_llm(prompts=["test prompt"])

        assert response == "Unknown output keyword."

    @patch("nemo_deploy.nlp.query_llm.ModelClient")
    def test_query_llm_non_bytes_output(self, mock_client, query):
        # Setup mock for non-bytes output type
        mock_instance = MagicMock()
        mock_client.return_value.__enter__.return_value = mock_instance
        mock_instance.infer_batch.return_value = {"outputs": np.array(["test response"])}
        mock_instance.model_config.outputs = [MagicMock(dtype=np.float32)]

        # Test query with non-bytes output
        response = query.query_llm(prompts=["test prompt"])

        assert response == np.array(["test response"])

    @patch("nemo_deploy.nlp.query_llm.ModelClient")
    def test_query_llm_sentences_fallback(self, mock_client, query):
        # Setup mock for sentences fallback case
        mock_instance = MagicMock()
        mock_client.return_value.__enter__.return_value = mock_instance
        mock_instance.infer_batch.return_value = {"sentences": np.array([b"test response"])}
        mock_instance.model_config.outputs = [MagicMock(dtype=np.bytes_)]

        # Test query that falls back to sentences key
        response = query.query_llm(prompts=["test prompt"])

        assert isinstance(response[0], str)
        assert response[0] == "test response"

    @patch("nemo_deploy.nlp.query_llm.ModelClient")
    def test_query_llm_with_lora_uids(self, mock_client, query):
        # Setup mock
        mock_instance = MagicMock()
        mock_client.return_value.__enter__.return_value = mock_instance
        mock_instance.infer_batch.return_value = {"outputs": np.array([b"test response"])}
        mock_instance.model_config.outputs = [MagicMock(dtype=np.bytes_)]

        # Test query with LoRA UIDs
        response = query.query_llm(prompts=["test prompt"], max_output_len=100, lora_uids=["lora1", "lora2"])

        assert isinstance(response[0], str)
        assert response[0] == "test response"

    @patch("nemo_deploy.nlp.query_llm.ModelClient")
    def test_query_llm_with_min_output_len(self, mock_client, query):
        # Setup mock
        mock_instance = MagicMock()
        mock_client.return_value.__enter__.return_value = mock_instance
        mock_instance.infer_batch.return_value = {"outputs": np.array([b"test response"])}
        mock_instance.model_config.outputs = [MagicMock(dtype=np.bytes_)]

        # Test query with min_output_len
        response = query.query_llm(prompts=["test prompt"], min_output_len=10, max_output_len=100)

        assert isinstance(response[0], str)
        assert response[0] == "test response"

    @patch("nemo_deploy.nlp.query_llm.ModelClient")
    def test_query_llm_with_logits_output(self, mock_client, query):
        # Setup mock for generation and context logits
        mock_instance = MagicMock()
        mock_client.return_value.__enter__.return_value = mock_instance
        mock_instance.infer_batch.return_value = {
            "outputs": np.array([b"test response"]),
            "generation_logits": np.array([[0.1, 0.2, 0.3]]),
            "context_logits": np.array([[0.4, 0.5, 0.6]]),
        }
        mock_instance.model_config.outputs = [MagicMock(dtype=np.bytes_)]

        # Test query with generation and context logits
        response = query.query_llm(
            prompts=["test prompt"],
            openai_format_response=True,
            output_generation_logits=True,
            output_context_logits=True,
        )

        assert isinstance(response, dict)
        assert "choices" in response
        assert "generation_logits" in response["choices"][0]
        assert "context_logits" in response["choices"][0]

    @patch("nemo_deploy.nlp.query_llm.ModelClient")
    def test_query_llm_all_parameters(self, mock_client, query):
        # Setup mock
        mock_instance = MagicMock()
        mock_client.return_value.__enter__.return_value = mock_instance
        mock_instance.infer_batch.return_value = {"outputs": np.array([b"test response"])}
        mock_instance.model_config.outputs = [MagicMock(dtype=np.bytes_)]

        # Test query with all parameters
        response = query.query_llm(
            prompts=["test prompt"],
            stop_words_list=["stop"],
            bad_words_list=["bad"],
            no_repeat_ngram_size=3,
            min_output_len=10,
            max_output_len=100,
            top_k=5,
            top_p=0.9,
            temperature=0.8,
            random_seed=42,
            lora_uids=["lora1"],
            use_greedy=True,
            repetition_penalty=1.1,
            add_BOS=True,
            all_probs=True,
            compute_logprob=True,
            end_strings=["</s>"],
            init_timeout=30.0,
            openai_format_response=False,
            output_context_logits=True,
            output_generation_logits=True,
        )

        assert isinstance(response[0], str)
        assert response[0] == "test response"

    @patch("nemo_deploy.nlp.query_llm.DecoupledModelClient")
    def test_query_llm_streaming_non_bytes_output(self, mock_client, query):
        # Setup mock for streaming with non-bytes output
        mock_instance = MagicMock()
        mock_client.return_value.__enter__.return_value = mock_instance
        mock_instance.infer_batch.return_value = [{"outputs": np.array(["test"])}, {"outputs": np.array([" response"])}]
        mock_instance.model_config.outputs = [MagicMock(dtype=np.float32)]

        # Test streaming query with non-bytes output
        responses = list(query.query_llm_streaming(prompts=["test prompt"], max_output_len=100))

        assert len(responses) == 2
        assert responses[0] == np.array(["test"])
        assert responses[1] == np.array([" response"])

    @patch("nemo_deploy.nlp.query_llm.DecoupledModelClient")
    def test_query_llm_streaming_all_parameters(self, mock_client, query):
        # Setup mock
        mock_instance = MagicMock()
        mock_client.return_value.__enter__.return_value = mock_instance
        mock_instance.infer_batch.return_value = [{"outputs": np.array([b"test response"])}]
        mock_instance.model_config.outputs = [MagicMock(dtype=np.bytes_)]

        # Test streaming query with all parameters
        responses = list(
            query.query_llm_streaming(
                prompts=["test prompt"],
                stop_words_list=["stop"],
                bad_words_list=["bad"],
                no_repeat_ngram_size=3,
                max_output_len=100,
                top_k=5,
                top_p=0.9,
                temperature=0.8,
                random_seed=42,
                lora_uids=["lora1"],
                init_timeout=30.0,
            )
        )

        assert len(responses) == 1
        assert responses[0] == "test response"
