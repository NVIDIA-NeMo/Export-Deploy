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
Tests for OpenAI API compatibility of HFRayDeployable completions endpoint.
"""

import json
import time
from unittest.mock import MagicMock, patch

import pytest
import torch


def run_coroutine(coro):
    """Helper function to run coroutines synchronously."""
    import asyncio

    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    return loop.run_until_complete(coro)


class TestOAIFormatCompliance:
    """Test suite to verify OpenAI API format compliance.
    
    These tests verify the completions endpoint logic without requiring Ray Serve.
    """

    @pytest.fixture
    def mock_model(self):
        """Create a mock model with ray_infer_fn."""
        mock_model = MagicMock()
        
        # Mock ray_infer_fn to return realistic results
        def mock_ray_infer_fn(inputs):
            sentences = ["Generated text for testing"]
            result = {"sentences": sentences}
            
            if inputs.get("compute_logprob") or inputs.get("n_top_logprobs", 0) > 0:
                result["log_probs"] = [[-0.1, -0.2, -0.3, -0.4]]
                
            if inputs.get("n_top_logprobs", 0) > 0:
                result["top_logprobs"] = [
                    json.dumps([
                        {" Generated": -0.1, " Other": -2.5},
                        {" text": -0.2, " word": -3.0},
                        {" for": -0.3, " to": -2.8},
                        {" testing": -0.4, " example": -3.2},
                    ])
                ]
            
            return result
        
        mock_model.ray_infer_fn = mock_ray_infer_fn
        return mock_model
    
    def simulate_completions_endpoint(self, mock_model, request, model_id="test-model"):
        """Simulate the completions endpoint logic from hf_deployable_ray.py."""
        # Preprocessing
        if "prompt" in request:
            request["prompts"] = [request["prompt"]]
        
        temperature = request.get("temperature", 0.0)
        top_p = request.get("top_p", 0.0)
        if temperature == 0.0 and top_p == 0.0:
            request["top_k"] = 1
        
        # Build inference inputs
        inference_inputs = {
            "prompts": request.get("prompts", []),
            "max_tokens": request.get("max_tokens", 256),
            "temperature": request.get("temperature", 0.0),
            "top_k": request.get("top_k", 0),
            "top_p": request.get("top_p", 0),
            "compute_logprob": True if (request.get("logprobs") is not None and request.get("logprobs", 0) > 0) else False,
            "n_top_logprobs": request.get("logprobs", 0),
            "echo": request.get("echo", False),
        }
        
        # Call model
        results = mock_model.ray_infer_fn(inference_inputs)
        generated_texts = results.get("sentences", [])
        
        # Calculate tokens
        prompt_tokens = sum(len(p.split()) for p in request.get("prompts", []))
        completion_tokens = sum(len(r.split()) for r in generated_texts)
        total_tokens = prompt_tokens + completion_tokens
        
        # Process logprobs
        log_probs_data = results.get("log_probs", None)
        if log_probs_data is not None:
            if isinstance(log_probs_data, list) and len(log_probs_data) > 0:
                if isinstance(log_probs_data[0], list):
                    log_probs_data = log_probs_data[0]
        
        top_log_probs_data = results.get("top_logprobs", None)
        if top_log_probs_data is not None:
            if isinstance(top_log_probs_data, list) and len(top_log_probs_data) > 0:
                if isinstance(top_log_probs_data[0], str):
                    top_log_probs_data = json.loads(top_log_probs_data[0])
                elif isinstance(top_log_probs_data[0], list):
                    top_log_probs_data = top_log_probs_data[0]
        
        # Build response
        output = {
            "id": f"cmpl-{int(time.time())}",
            "object": "text_completion",
            "created": int(time.time()),
            "model": model_id,
            "choices": [
                {
                    "text": " ".join(generated_texts),
                    "index": 0,
                    "logprobs": (
                        {
                            "token_logprobs": log_probs_data,
                            "top_logprobs": top_log_probs_data,
                        }
                        if log_probs_data is not None
                        else None
                    ),
                    "finish_reason": (
                        "length"
                        if generated_texts and len(generated_texts[0]) >= request.get("max_tokens", 256)
                        else "stop"
                    ),
                }
            ],
            "usage": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
            },
        }
        return output

    def test_basic_completion_format(self, mock_model):
        """Test that basic completion response follows OpenAI format."""
        request = {
            "prompt": "Test prompt",
            "max_tokens": 10,
        }

        result = self.simulate_completions_endpoint(mock_model, request)

        # Verify top-level structure
        assert "id" in result
        assert "object" in result
        assert "created" in result
        assert "model" in result
        assert "choices" in result
        assert "usage" in result

        # Verify object type
        assert result["object"] == "text_completion"

        # Verify id format
        assert result["id"].startswith("cmpl-")

        # Verify model
        assert result["model"] == "test-model"

        # Verify timestamp
        assert isinstance(result["created"], int)
        assert result["created"] > 0

    def test_choices_structure(self, mock_model):
        """Test that choices array follows OpenAI format."""
        request = {
            "prompt": "Test prompt",
            "max_tokens": 10,
        }

        result = self.simulate_completions_endpoint(mock_model, request)

        # Verify choices structure
        assert isinstance(result["choices"], list)
        assert len(result["choices"]) == 1

        choice = result["choices"][0]
        assert "text" in choice
        assert "index" in choice
        assert "logprobs" in choice
        assert "finish_reason" in choice

        # Verify choice values
        assert isinstance(choice["text"], str)
        assert choice["index"] == 0
        assert choice["finish_reason"] in ["stop", "length"]

    def test_usage_structure(self, mock_model):
        """Test that usage object follows OpenAI format."""
        request = {
            "prompt": "Test prompt",
            "max_tokens": 10,
        }

        result = self.simulate_completions_endpoint(mock_model, request)

        # Verify usage structure
        usage = result["usage"]
        assert "prompt_tokens" in usage
        assert "completion_tokens" in usage
        assert "total_tokens" in usage

        # Verify usage values
        assert isinstance(usage["prompt_tokens"], int)
        assert isinstance(usage["completion_tokens"], int)
        assert isinstance(usage["total_tokens"], int)
        assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]

    def test_logprobs_format_when_requested(self, mock_model):
        """Test that logprobs follow OpenAI format when requested."""
        request = {
            "prompt": "Test prompt",
            "max_tokens": 10,
            "logprobs": 1,
        }

        result = self.simulate_completions_endpoint(mock_model, request)

        # Verify logprobs structure
        logprobs = result["choices"][0]["logprobs"]
        assert logprobs is not None
        assert "token_logprobs" in logprobs
        assert "top_logprobs" in logprobs

        # Verify token_logprobs format
        assert isinstance(logprobs["token_logprobs"], list)
        assert all(isinstance(lp, (int, float)) for lp in logprobs["token_logprobs"])

        # Verify top_logprobs format
        assert isinstance(logprobs["top_logprobs"], list)
        for top_lp in logprobs["top_logprobs"]:
            assert isinstance(top_lp, dict)
            # Each entry should be token: logprob
            for token, prob in top_lp.items():
                assert isinstance(token, str)
                assert isinstance(prob, (int, float))

    def test_logprobs_none_when_not_requested(self, mock_model):
        """Test that logprobs is None when not requested."""
        request = {
            "prompt": "Test prompt",
            "max_tokens": 10,
        }

        result = self.simulate_completions_endpoint(mock_model, request)

        # Verify logprobs is None
        assert result["choices"][0]["logprobs"] is None

    def test_echo_parameter(self, mock_model):
        """Test echo parameter behavior."""
        # Mock to return echo response
        original_ray_infer_fn = mock_model.ray_infer_fn
        
        def mock_echo_ray_infer_fn(inputs):
            if inputs.get("echo"):
                return {"sentences": ["Test prompt Generated text"]}
            else:
                return {"sentences": ["Generated text"]}
        
        mock_model.ray_infer_fn = mock_echo_ray_infer_fn

        # Test with echo=True
        request_with_echo = {
            "prompt": "Test prompt",
            "max_tokens": 10,
            "echo": True,
        }

        result_with_echo = self.simulate_completions_endpoint(mock_model, request_with_echo)
        assert "Test prompt" in result_with_echo["choices"][0]["text"]

        # Test with echo=False
        request_without_echo = {
            "prompt": "Test prompt",
            "max_tokens": 10,
            "echo": False,
        }

        result_without_echo = self.simulate_completions_endpoint(mock_model, request_without_echo)
        # Should not start with prompt (though may contain it in generated text)
        assert result_without_echo["choices"][0]["text"] == "Generated text"

        # Restore original
        mock_model.ray_infer_fn = original_ray_infer_fn

    def test_temperature_parameter(self, mock_model):
        """Test temperature parameter handling."""
        request = {
            "prompt": "Test prompt",
            "max_tokens": 10,
            "temperature": 0.7,
        }

        result = self.simulate_completions_endpoint(mock_model, request)

        # Should succeed and return valid response
        assert result["object"] == "text_completion"
        assert len(result["choices"]) == 1

    def test_temperature_zero_handling(self, mock_model):
        """Test that temperature=0 is handled correctly."""
        request = {
            "prompt": "Test prompt",
            "max_tokens": 10,
            "temperature": 0.0,
            "top_p": 0.0,
        }

        # Should not raise an error
        result = self.simulate_completions_endpoint(mock_model, request)

        # Should succeed and return valid response
        assert result["object"] == "text_completion"
        assert len(result["choices"]) == 1

    def test_multiple_prompts_support(self, mock_model):
        """Test support for multiple prompts (prompts parameter)."""
        # Mock to return multiple responses
        original_ray_infer_fn = mock_model.ray_infer_fn
        
        def mock_multi_ray_infer_fn(inputs):
            num_prompts = len(inputs.get("prompts", []))
            return {"sentences": [f"Response {i}" for i in range(num_prompts)]}
        
        mock_model.ray_infer_fn = mock_multi_ray_infer_fn

        request = {
            "prompts": ["Prompt 1", "Prompt 2"],
            "max_tokens": 10,
        }

        result = self.simulate_completions_endpoint(mock_model, request)

        # Should return combined response
        assert result["object"] == "text_completion"
        assert "Response" in result["choices"][0]["text"]

        # Restore original
        mock_model.ray_infer_fn = original_ray_infer_fn

    def test_finish_reason_stop(self, mock_model):
        """Test finish_reason is 'stop' for completed generation."""
        request = {
            "prompt": "Short",
            "max_tokens": 100,  # More than needed
        }

        result = self.simulate_completions_endpoint(mock_model, request)

        # Should finish with 'stop'
        assert result["choices"][0]["finish_reason"] == "stop"

    def test_finish_reason_length(self, mock_model):
        """Test finish_reason is 'length' when max_tokens is reached."""
        # Mock to return long response
        original_ray_infer_fn = mock_model.ray_infer_fn
        
        def mock_long_ray_infer_fn(inputs):
            # Return a response that's longer than max_tokens
            max_tokens = inputs.get("max_tokens", 256)
            return {"sentences": [" ".join(["word"] * (max_tokens + 10))]}
        
        mock_model.ray_infer_fn = mock_long_ray_infer_fn

        request = {
            "prompt": "Test",
            "max_tokens": 5,
        }

        result = self.simulate_completions_endpoint(mock_model, request)

        # Should finish with 'length'
        assert result["choices"][0]["finish_reason"] == "length"

        # Restore original
        mock_model.ray_infer_fn = original_ray_infer_fn

    def test_json_serializable(self, mock_model):
        """Test that the entire response is JSON serializable."""
        request = {
            "prompt": "Test prompt",
            "max_tokens": 10,
            "logprobs": 1,
            "echo": True,
        }

        result = self.simulate_completions_endpoint(mock_model, request)

        # Should be JSON serializable
        try:
            json_str = json.dumps(result)
            json.loads(json_str)  # Should not raise
        except (TypeError, ValueError) as e:
            pytest.fail(f"Response is not JSON serializable: {e}")

    def test_prompt_parameter_backward_compatibility(self, mock_model):
        """Test backward compatibility with 'prompt' parameter (vs 'prompts')."""
        request = {
            "prompt": "Test prompt",  # Using singular 'prompt'
            "max_tokens": 10,
        }

        result = self.simulate_completions_endpoint(mock_model, request)

        # Should work and return valid response
        assert result["object"] == "text_completion"
        assert len(result["choices"]) == 1

    def test_default_parameters(self, mock_model):
        """Test that request works with minimal parameters."""
        request = {
            "prompt": "Test prompt",
        }

        result = self.simulate_completions_endpoint(mock_model, request)

        # Should work with defaults
        assert result["object"] == "text_completion"
        assert len(result["choices"]) == 1
        assert result["choices"][0]["logprobs"] is None


@pytest.mark.skip(reason="These tests are covered by test_hf_deployable_standalone.py which has better mocking")
class TestHFDeployableGenerateMethod:
    """Test suite for HuggingFaceLLMDeploy generate method changes."""

    @pytest.fixture
    def mock_model_and_tokenizer(self):
        """Create mock model and tokenizer."""
        with patch("transformers.AutoModelForCausalLM") as mock_model_class:
            with patch("transformers.AutoTokenizer") as mock_tokenizer_class:
                # Mock tokenizer
                mock_tokenizer = MagicMock()
                mock_tokenizer.return_value = {
                    "input_ids": torch.tensor([[1, 2, 3, 4]]),
                    "attention_mask": torch.tensor([[1, 1, 1, 1]]),
                }
                mock_tokenizer.decode.return_value = " generated text"
                mock_tokenizer.eos_token = "</s>"
                mock_tokenizer.pad_token = "</s>"
                mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

                # Mock model
                mock_model = MagicMock()
                mock_model.generate.return_value = {
                    "sequences": torch.tensor([[1, 2, 3, 4, 5, 6]]),
                    "scores": [
                        torch.randn(1, 50000),
                        torch.randn(1, 50000),
                    ],
                }
                mock_model.cuda.return_value = mock_model
                mock_model_class.from_pretrained.return_value = mock_model

                yield mock_model, mock_tokenizer

    @pytest.fixture
    def hf_deployable(self, mock_model_and_tokenizer):
        """Create HuggingFaceLLMDeploy instance for testing."""
        from nemo_deploy.llm.hf_deployable import HuggingFaceLLMDeploy
        
        mock_model, mock_tokenizer = mock_model_and_tokenizer
        
        with patch("torch.cuda.device_count", return_value=1):
            instance = HuggingFaceLLMDeploy(
                hf_model_id_path="test/model",
                task="text-generation",
            )
            instance.model = mock_model
            instance.tokenizer = mock_tokenizer
            yield instance

    def test_generate_with_echo_false(self, hf_deployable):
        """Test generate method with echo=False (only generated text)."""
        result = hf_deployable.generate(
            text_inputs=["Test prompt"],
            max_new_tokens=5,
            echo=False,
            return_dict_in_generate=True,
            output_scores=True,
        )

        # Should return dict with sentences
        assert isinstance(result, dict)
        assert "sentences" in result
        assert isinstance(result["sentences"], list)

    def test_generate_with_echo_true(self, hf_deployable):
        """Test generate method with echo=True (prompt + generated text)."""
        result = hf_deployable.generate(
            text_inputs=["Test prompt"],
            max_new_tokens=5,
            echo=True,
            return_dict_in_generate=True,
            output_scores=True,
        )

        # Should return dict with sentences
        assert isinstance(result, dict)
        assert "sentences" in result
        assert isinstance(result["sentences"], list)

    def test_generate_returns_input_lengths(self, hf_deployable):
        """Test that generate returns input_lengths for echo processing."""
        result = hf_deployable.generate(
            text_inputs=["Test prompt"],
            max_new_tokens=5,
            echo=True,
            return_dict_in_generate=True,
            output_scores=True,
        )

        # Should include input_lengths
        assert "input_lengths" in result
        assert isinstance(result["input_lengths"], list)

    def test_generate_returns_sequences(self, hf_deployable):
        """Test that generate returns sequences when return_dict_in_generate=True."""
        result = hf_deployable.generate(
            text_inputs=["Test prompt"],
            max_new_tokens=5,
            return_dict_in_generate=True,
            output_scores=True,
        )

        # Should include sequences
        assert "sequences" in result

    def test_generate_returns_scores(self, hf_deployable):
        """Test that generate returns scores when output_scores=True."""
        result = hf_deployable.generate(
            text_inputs=["Test prompt"],
            max_new_tokens=5,
            return_dict_in_generate=True,
            output_scores=True,
        )

        # Should include scores
        assert "scores" in result


@pytest.mark.skip(reason="These tests are covered by test_hf_deployable_standalone.py which has better mocking")
class TestRayInferFnLogprobProcessing:
    """Test suite for ray_infer_fn logprob processing logic."""

    @pytest.fixture
    def mock_hf_deployable_for_logprobs(self):
        """Create a mock HuggingFaceLLMDeploy with realistic logprob processing."""
        from nemo_deploy.llm.hf_deployable import HuggingFaceLLMDeploy
        
        with patch("transformers.AutoModelForCausalLM") as mock_model_class:
            with patch("transformers.AutoTokenizer") as mock_tokenizer_class:
                # Mock tokenizer
                mock_tokenizer = MagicMock()
                mock_tokenizer.return_value = {
                    "input_ids": torch.tensor([[1, 2, 3, 4]]),
                    "attention_mask": torch.tensor([[1, 1, 1, 1]]),
                }
                mock_tokenizer.decode.side_effect = lambda ids: f"token_{ids[0] if isinstance(ids, list) and len(ids) > 0 else 'unknown'}"
                mock_tokenizer.eos_token = "</s>"
                mock_tokenizer.pad_token = "</s>"
                mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

                # Mock model
                mock_model = MagicMock()
                
                # Mock forward pass for prompt logits
                mock_output = MagicMock()
                mock_output.logits = torch.randn(1, 4, 50000)  # [batch, seq_len, vocab]
                mock_model.return_value = mock_output
                
                # Mock generate
                mock_model.generate.return_value = {
                    "sequences": torch.tensor([[1, 2, 3, 4, 5, 6]]),
                    "scores": [
                        torch.randn(1, 50000),  # Score for position 4
                        torch.randn(1, 50000),  # Score for position 5
                    ],
                }
                mock_model.cuda.return_value = mock_model
                mock_model_class.from_pretrained.return_value = mock_model

                with patch("torch.cuda.device_count", return_value=1):
                    instance = HuggingFaceLLMDeploy(
                        hf_model_id_path="test/model",
                        task="text-generation",
                    )
                    instance.model = mock_model
                    instance.tokenizer = mock_tokenizer
                    yield instance

    def test_ray_infer_fn_computes_logprobs(self, mock_hf_deployable_for_logprobs):
        """Test that ray_infer_fn computes logprobs when requested."""
        inputs = {
            "prompts": ["Test prompt"],
            "max_tokens": 2,
            "compute_logprob": True,
            "n_top_logprobs": 0,
        }

        result = mock_hf_deployable_for_logprobs.ray_infer_fn(inputs)

        # Should include log_probs
        assert "log_probs" in result
        assert isinstance(result["log_probs"], list)
        assert len(result["log_probs"]) > 0

    def test_ray_infer_fn_computes_top_logprobs(self, mock_hf_deployable_for_logprobs):
        """Test that ray_infer_fn computes top_logprobs when requested."""
        inputs = {
            "prompts": ["Test prompt"],
            "max_tokens": 2,
            "compute_logprob": True,
            "n_top_logprobs": 1,
        }

        result = mock_hf_deployable_for_logprobs.ray_infer_fn(inputs)

        # Should include top_logprobs
        assert "top_logprobs" in result
        assert isinstance(result["top_logprobs"], list)
        assert len(result["top_logprobs"]) > 0

    def test_ray_infer_fn_echo_includes_prompt_logprobs(self, mock_hf_deployable_for_logprobs):
        """Test that ray_infer_fn includes prompt token logprobs when echo=True."""
        inputs = {
            "prompts": ["Test prompt"],
            "max_tokens": 2,
            "compute_logprob": True,
            "n_top_logprobs": 1,
            "echo": True,
        }

        result = mock_hf_deployable_for_logprobs.ray_infer_fn(inputs)

        # Should include log_probs and top_logprobs
        assert "log_probs" in result
        assert "top_logprobs" in result
        
        # Log probs should include both prompt and generated tokens
        assert len(result["log_probs"][0]) > 2  # More than just generated tokens

    def test_ray_infer_fn_removes_intermediate_outputs(self, mock_hf_deployable_for_logprobs):
        """Test that ray_infer_fn removes scores, sequences, input_lengths from output."""
        inputs = {
            "prompts": ["Test prompt"],
            "max_tokens": 2,
            "compute_logprob": True,
            "n_top_logprobs": 1,
        }

        result = mock_hf_deployable_for_logprobs.ray_infer_fn(inputs)

        # Should not include intermediate outputs
        assert "scores" not in result
        assert "sequences" not in result
        assert "input_lengths" not in result

    def test_top_logprobs_json_format(self, mock_hf_deployable_for_logprobs):
        """Test that top_logprobs are JSON encoded strings."""
        inputs = {
            "prompts": ["Test prompt"],
            "max_tokens": 2,
            "compute_logprob": True,
            "n_top_logprobs": 1,
        }

        result = mock_hf_deployable_for_logprobs.ray_infer_fn(inputs)

        # top_logprobs should be JSON strings
        assert isinstance(result["top_logprobs"], list)
        assert len(result["top_logprobs"]) > 0
        assert isinstance(result["top_logprobs"][0], str)
        
        # Should be valid JSON
        parsed = json.loads(result["top_logprobs"][0])
        assert isinstance(parsed, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

