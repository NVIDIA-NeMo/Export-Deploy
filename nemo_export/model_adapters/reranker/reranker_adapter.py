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

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer


class SequenceClassificationModelAdapterWithoutTypeIds(torch.nn.Module):
    """Adapter for sequence classification models that don't use token type IDs.

    This adapter wraps a HuggingFace AutoModelForSequenceClassification model
    and provides a simplified forward method that only takes input_ids and
    attention_mask as inputs, excluding token_type_ids.

    Args:
        model: A HuggingFace AutoModelForSequenceClassification model to wrap.

    Attributes:
        config: The configuration object from the wrapped model.
    """

    def __init__(self, model: AutoModelForSequenceClassification) -> None:
        super().__init__()
        self._model = model
        self.config = model.config

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Forward pass through the sequence classification model.

        Args:
            input_ids: Token IDs for the input sequences. Shape: (batch_size, sequence_length)
            attention_mask: Attention mask indicating which tokens should be attended to.
                           Shape: (batch_size, sequence_length)

        Returns:
            Logits from the classification head. Shape: (batch_size, num_labels)
        """
        return self._model(input_ids=input_ids, attention_mask=attention_mask).logits


class SequenceClassificationModelAdapterWithTypeIds(torch.nn.Module):
    """Adapter for sequence classification models that use token type IDs.

    This adapter wraps a HuggingFace AutoModelForSequenceClassification model
    and provides a forward method that includes token_type_ids for models that
    require this input (e.g., BERT-based models).

    Args:
        model: A HuggingFace AutoModelForSequenceClassification model to wrap.

    Attributes:
        config: The configuration object from the wrapped model.
    """

    def __init__(self, model: AutoModelForSequenceClassification) -> None:
        super().__init__()
        self._model = model
        self.config = model.config

    def forward(
        self,
        input_ids: torch.Tensor,
        token_type_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through the sequence classification model.

        Args:
            input_ids: Token IDs for the input sequences. Shape: (batch_size, sequence_length)
            token_type_ids: Token type IDs to distinguish between different parts of the input.
                           Shape: (batch_size, sequence_length)
            attention_mask: Attention mask indicating which tokens should be attended to.
                           Shape: (batch_size, sequence_length)

        Returns:
            Logits from the classification head. Shape: (batch_size, num_labels)
        """
        return self._model(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask,
        ).logits


def get_llama_reranker_hf_model(
    model_name_or_path: str | os.PathLike[str],
    trust_remote_code: bool = False,
    attn_implementation: str | None = None,
):
    """Load and adapt a HuggingFace reranker model for export.

    This function loads a sequence classification model from HuggingFace and wraps
    it with an appropriate adapter based on whether the model uses token_type_ids.
    It also handles specific configuration adjustments for certain model types.

    Args:
        model_name_or_path: Path to the model directory or HuggingFace model identifier.
        trust_remote_code: Whether to trust and execute remote code from the model repository.
                          Defaults to False.
        attn_implementation: Specific attention implementation to use. If provided,
                           the model's attention implementation will be set to this value.
                           Defaults to None.

    Returns:
        tuple: A tuple containing:
            - model: The wrapped sequence classification model (either with or without token type IDs).
            - tokenizer: The corresponding tokenizer for the model.

    Note:
        The function automatically determines whether to use the adapter with or without
        token_type_ids based on the tokenizer's model_input_names attribute.

        For models with attn_implementation specified, the config is reset after
        initialization to handle cases where the config is mutated during init.
    """
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
        attn_implementation=attn_implementation,
    ).eval()

    if attn_implementation:
        # reset config to handle case where config is mutated after init
        # TODO: remove when we're no longer using Llama 3.1 model with `_attn_implementation` set in __init__ method.
        model.config._attn_implementation = attn_implementation

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=trust_remote_code,
    )
    if "token_type_ids" in tokenizer.model_input_names:
        model = SequenceClassificationModelAdapterWithTypeIds(model)
    else:
        model = SequenceClassificationModelAdapterWithoutTypeIds(model)

    return model, tokenizer
