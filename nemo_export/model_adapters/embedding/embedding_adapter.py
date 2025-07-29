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
from typing import Literal, Optional, Union

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


class LlamaBidirectionalHFAdapter(torch.nn.Module):
    """
    Wraps a text embedding model with pooling and normalization for bidirectional encoding.

    This adapter combines a transformer model with configurable pooling strategies and optional
    L2 normalization to produce fixed-size embeddings from variable-length text sequences.
    It supports dimension reduction and various pooling methods including average, CLS token,
    and last token pooling.

    Args:
        model: The underlying transformer model (e.g., AutoModel from HuggingFace).
        normalize: Whether to apply L2 normalization to the output embeddings.
        pooling_module: The pooling module to use for aggregating token embeddings.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        normalize: bool,
        pooling_module: torch.nn.Module,
    ) -> None:
        """
        Initialize the LlamaBidirectionalHFAdapter.

        Args:
            model: The transformer model to wrap.
            normalize: If True, applies L2 normalization to output embeddings.
            pooling_module: Module that handles pooling of token embeddings.
        """
        super().__init__()
        self.model = model
        self.normalize = normalize
        self.pooling_module = pooling_module

    @property
    def device(self) -> torch.device:
        """
        Returns the device of the underlying model.

        Returns:
            torch.device: The device where the model parameters are located.
        """
        return self.model.device

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        dimensions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass through the adapted model to generate embeddings.

        Args:
            input_ids: Token IDs of shape (batch_size, sequence_length).
            attention_mask: Attention mask of shape (batch_size, sequence_length).
            token_type_ids: Optional token type IDs for models that use them.
            dimensions: Optional tensor specifying the desired output dimensions
                       for each sample in the batch. If provided, embeddings will
                       be truncated/masked to these dimensions.

        Returns:
            torch.Tensor: Pooled and optionally normalized embeddings of shape
                         (batch_size, embedding_dim) or (batch_size, max_dimensions)
                         if dimensions parameter is used.

        Raises:
            ValueError: If dimensions contain non-positive values.
        """
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if token_type_ids is not None:
            inputs["token_type_ids"] = token_type_ids
        outputs = self.model(**inputs)
        hidden_states = outputs["last_hidden_state"].to(torch.float32)
        embeddings = self.pooling_module(hidden_states, inputs["attention_mask"])

        if dimensions is not None:
            if not torch.all(dimensions > 0):
                raise ValueError("Dimensions must be positive")

            fill_value = torch.tensor(float("-inf"), dtype=embeddings.dtype, device=embeddings.device)

            clipped_dimensions = torch.clamp(dimensions, max=int(embeddings.shape[1]))

            embeddings = embeddings.masked_fill(
                torch.arange(embeddings.shape[1], device=embeddings.device) >= clipped_dimensions.unsqueeze(-1),
                fill_value,
            )[:, : dimensions.max()]

        if self.normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)

        return embeddings


class Pooling(torch.nn.Module):
    """
    Pooling layer that aggregates token-level embeddings into sequence-level embeddings.

    Supports multiple pooling strategies:
    - 'avg': Average pooling over non-padded tokens
    - 'cls': Uses the first token (CLS token) with right padding
    - 'cls__left': Uses the first non-padded token with left padding
    - 'last': Uses the last token with left padding
    - 'last__right': Uses the last non-padded token with right padding

    Args:
        pooling_mode: The pooling strategy to use.
    """

    def __init__(self, pooling_mode: str):
        """
        Initialize the Pooling layer.

        Args:
            pooling_mode: The pooling strategy. Must be one of:
                         'avg', 'cls', 'cls__left', 'last', 'last__right'.
        """
        super().__init__()
        self.pooling_mode = pooling_mode

    def forward(self, last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Apply pooling to the hidden states.

        Args:
            last_hidden_states: Hidden states from the transformer model of shape
                               (batch_size, sequence_length, hidden_size).
            attention_mask: Attention mask of shape (batch_size, sequence_length)
                           where 1 indicates real tokens and 0 indicates padding.

        Returns:
            torch.Tensor: Pooled embeddings of shape (batch_size, hidden_size).

        Raises:
            ValueError: If the pooling_mode is not supported.
        """
        last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)

        pool_type = self.pooling_mode
        if pool_type == "avg":
            epsilon = 1e-9  # A small value to avoid division by zero
            emb = last_hidden.sum(dim=1) / (attention_mask.sum(dim=1)[..., None] + epsilon)
        elif pool_type == "cls":  # tokenizer padding right
            emb = last_hidden[:, 0]
        elif pool_type == "cls__left":  # tokenizer padding left
            seq_idxs = (1 - attention_mask).sum(dim=1).to(dtype=torch.long)
            batch_size = last_hidden.shape[0]
            batch_idxs = torch.arange(batch_size, device=last_hidden.device)
            emb = last_hidden[batch_idxs, seq_idxs]
        elif pool_type == "last":  # tokenizer padding left
            emb = last_hidden[:, -1]
        elif pool_type == "last__right":  # tokenizer padding right
            sequence_lengths = (attention_mask.sum(dim=1) - 1).to(dtype=torch.long)
            batch_size = last_hidden.shape[0]
            emb = last_hidden[torch.arange(batch_size, device=last_hidden.device), sequence_lengths]
        else:
            raise ValueError(f"pool_type {pool_type} not supported")

        return emb


def get_llama_bidirectional_hf_model(
    model_name_or_path: Union[str, os.PathLike[str]],
    normalize: bool,
    pooling_mode: Optional[Literal["avg", "cls", "last"]] = None,
    torch_dtype: Optional[Union[torch.dtype, str]] = None,
    trust_remote_code: bool = False,
):
    """
    Factory function to create a LlamaBidirectionalHFAdapter with proper configuration.

    This function loads a HuggingFace transformer model and tokenizer, configures the
    appropriate pooling strategy based on the tokenizer's padding side, and wraps
    everything in a LlamaBidirectionalHFAdapter.

    Special handling is provided for NVEmbedModel which has separate embedding and
    latent attention components.

    Args:
        model_name_or_path: Path to the model directory or HuggingFace model identifier.
        normalize: Whether to apply L2 normalization to the output embeddings.
        pooling_mode: The pooling strategy to use. If None, defaults to "avg".
                     Will be automatically adjusted based on tokenizer padding side:
                     - "last" becomes "last__right" for right-padding tokenizers
                     - "cls" becomes "cls__left" for left-padding tokenizers
        torch_dtype: The torch data type to use for the model. If None, uses model default.
        trust_remote_code: Whether to trust remote code when loading the model.

    Returns:
        tuple: A tuple containing:
            - LlamaBidirectionalHFAdapter: The configured adapter model
            - AutoTokenizer: The tokenizer for the model

    Example:
        >>> model, tokenizer = get_llama_bidirectional_hf_model(
        ...     "sentence-transformers/all-MiniLM-L6-v2",
        ...     normalize=True,
        ...     pooling_mode="avg"
        ... )
        >>> # Use model and tokenizer for embedding generation
    """
    # check that the tokenizer matches the requirements of the pooling mode
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=trust_remote_code)
    pooling_mode = pooling_mode or "avg"
    if pooling_mode == "last" and tokenizer.padding_side == "right":
        pooling_mode = "last__right"  # type: ignore
    if pooling_mode == "cls" and tokenizer.padding_side == "left":
        pooling_mode = "cls__left"  # type: ignore

    # load the model
    model = AutoModel.from_pretrained(
        model_name_or_path, torch_dtype=torch_dtype, trust_remote_code=trust_remote_code
    ).eval()

    # configure pooling
    pooling_module = Pooling(pooling_mode=pooling_mode)

    # NV-Embed-v1 model has seperate embedding model and a built-in pooling module
    if (
        model.__class__.__name__ == "NVEmbedModel"
        and hasattr(model, "latent_attention_model")
        and hasattr(model, "embedding_model")
    ):
        pooling_module = model.latent_attention_model
        model = model.embedding_model

    adapted_model = LlamaBidirectionalHFAdapter(model=model, normalize=normalize, pooling_module=pooling_module)
    return adapted_model, tokenizer
