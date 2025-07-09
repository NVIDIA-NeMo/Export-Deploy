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

from unittest.mock import (
    MagicMock,
)

import numpy as np
import pytest


@pytest.mark.run_only_on("GPU")
@pytest.mark.unit
def test_to_word_list_format_basic():
    """Test basic functionality of to_word_list_format function."""
    try:
        import tensorrt_llm  # noqa: F401
    except ImportError:
        pytest.skip("Could not import TRTLLM helpers. tensorrt_llm is likely not installed")
        return

    from nemo_export.trt_llm.tensorrt_llm_run import to_word_list_format

    # Create a mock tokenizer
    mock_tokenizer = MagicMock()
    mock_tokenizer.encode.side_effect = lambda x: {
        "<extra_id_1>": [100],
        "<extra_id_1>hello": [100, 200],
        "<extra_id_1>world": [100, 300],
        "hello": [200],
        "world": [300],
    }.get(x, [])

    # Test basic functionality
    word_dict = [["hello,world"]]
    result = to_word_list_format(word_dict, tokenizer=mock_tokenizer)

    # Check result shape and format
    assert result.shape[0] == 1  # batch_size
    assert result.shape[1] == 2  # flat_ids and offsets
    assert result.dtype == np.int32

    # Check that the function processed the CSV format correctly
    flat_ids = result[0, 0]

    # Should have tokens for "hello" and "world"
    assert 200 in flat_ids  # token for "hello"
    assert 300 in flat_ids  # token for "world"


@pytest.mark.run_only_on("GPU")
@pytest.mark.unit
def test_to_word_list_format_multiple_batches():
    """Test to_word_list_format with multiple batches."""
    try:
        import tensorrt_llm  # noqa: F401
    except ImportError:
        pytest.skip("Could not import TRTLLM helpers. tensorrt_llm is likely not installed")
        return

    from nemo_export.trt_llm.tensorrt_llm_run import to_word_list_format

    # Create a mock tokenizer
    mock_tokenizer = MagicMock()
    mock_tokenizer.encode.side_effect = lambda x: {
        "<extra_id_1>": [100],
        "<extra_id_1>hello": [100, 200],
        "<extra_id_1>world": [100, 300],
        "<extra_id_1>foo": [100, 400],
        "<extra_id_1>bar": [100, 500],
        "hello": [200],
        "world": [300],
        "foo": [400],
        "bar": [500],
    }.get(x, [])

    # Test with multiple batches
    word_dict = [["hello,world"], ["foo,bar"]]
    result = to_word_list_format(word_dict, tokenizer=mock_tokenizer)

    # Check result shape
    assert result.shape[0] == 2  # batch_size = 2
    assert result.shape[1] == 2  # flat_ids and offsets
    assert result.dtype == np.int32

    # Check first batch
    flat_ids_0 = result[0, 0]
    assert 200 in flat_ids_0  # token for "hello"
    assert 300 in flat_ids_0  # token for "world"

    # Check second batch
    flat_ids_1 = result[1, 0]
    assert 400 in flat_ids_1  # token for "foo"
    assert 500 in flat_ids_1  # token for "bar"


@pytest.mark.run_only_on("GPU")
@pytest.mark.unit
def test_to_word_list_format_bytes_input():
    """Test to_word_list_format with bytes input."""
    try:
        import tensorrt_llm  # noqa: F401
    except ImportError:
        pytest.skip("Could not import TRTLLM helpers. tensorrt_llm is likely not installed")
        return

    from nemo_export.trt_llm.tensorrt_llm_run import to_word_list_format

    # Create a mock tokenizer
    mock_tokenizer = MagicMock()
    mock_tokenizer.encode.side_effect = lambda x: {
        "<extra_id_1>": [100],
        "<extra_id_1>hello": [100, 200],
        "hello": [200],
    }.get(x, [])

    # Test with bytes input
    word_dict = [[b"hello"]]
    result = to_word_list_format(word_dict, tokenizer=mock_tokenizer)

    # Check that bytes were properly decoded and processed
    assert result.shape[0] == 1  # batch_size
    assert result.shape[1] == 2  # flat_ids and offsets
    assert result.dtype == np.int32

    flat_ids = result[0, 0]
    assert 200 in flat_ids  # token for "hello"


@pytest.mark.run_only_on("GPU")
@pytest.mark.unit
def test_to_word_list_format_empty_words():
    """Test to_word_list_format with empty words."""
    try:
        import tensorrt_llm  # noqa: F401
    except ImportError:
        pytest.skip("Could not import TRTLLM helpers. tensorrt_llm is likely not installed")
        return

    from nemo_export.trt_llm.tensorrt_llm_run import to_word_list_format

    # Create a mock tokenizer that returns empty list for empty string
    mock_tokenizer = MagicMock()
    mock_tokenizer.encode.side_effect = lambda x: {
        "<extra_id_1>": [100],
        "<extra_id_1>": [100],  # Empty word after prefix
        "": [],  # Empty string
    }.get(x, [])

    # Test with empty words
    word_dict = [["hello,"]]  # This will create "hello" and empty string
    result = to_word_list_format(word_dict, tokenizer=mock_tokenizer)

    # Should still work and handle empty words gracefully
    assert result.shape[0] == 1  # batch_size
    assert result.shape[1] == 2  # flat_ids and offsets
    assert result.dtype == np.int32


@pytest.mark.run_only_on("GPU")
@pytest.mark.unit
def test_to_word_list_format_custom_ref_string():
    """Test to_word_list_format with custom reference string."""
    try:
        import tensorrt_llm  # noqa: F401
    except ImportError:
        pytest.skip("Could not import TRTLLM helpers. tensorrt_llm is likely not installed")
        return

    from nemo_export.trt_llm.tensorrt_llm_run import to_word_list_format

    # Create a mock tokenizer
    mock_tokenizer = MagicMock()
    mock_tokenizer.encode.side_effect = lambda x: {
        "<custom_ref>": [999],
        "<custom_ref>hello": [999, 200],
        "hello": [200],
    }.get(x, [])

    # Test with custom reference string
    word_dict = [["hello"]]
    result = to_word_list_format(word_dict, tokenizer=mock_tokenizer, ref_str="<custom_ref>")

    # Check that custom ref string was used
    assert result.shape[0] == 1  # batch_size
    assert result.shape[1] == 2  # flat_ids and offsets
    assert result.dtype == np.int32

    flat_ids = result[0, 0]
    assert 200 in flat_ids  # token for "hello"


@pytest.mark.run_only_on("GPU")
@pytest.mark.unit
def test_to_word_list_format_prefix_merge_fallback():
    """Test to_word_list_format fallback when prefix merges with word."""
    try:
        import tensorrt_llm  # noqa: F401
    except ImportError:
        pytest.skip("Could not import TRTLLM helpers. tensorrt_llm is likely not installed")
        return

    from nemo_export.trt_llm.tensorrt_llm_run import to_word_list_format

    # Create a mock tokenizer that simulates prefix merging
    mock_tokenizer = MagicMock()
    mock_tokenizer.encode.side_effect = lambda x: {
        "<extra_id_1>": [100],
        "<extra_id_1>hello": [888],  # Merged token, different from [100, 200]
        "hello": [200],  # Fallback encoding
    }.get(x, [])

    # Test with prefix merge scenario
    word_dict = [["hello"]]
    result = to_word_list_format(word_dict, tokenizer=mock_tokenizer)

    # Should use fallback encoding when prefix merges
    assert result.shape[0] == 1  # batch_size
    assert result.shape[1] == 2  # flat_ids and offsets
    assert result.dtype == np.int32

    flat_ids = result[0, 0]
    assert 200 in flat_ids  # Should use fallback token for "hello"


@pytest.mark.run_only_on("GPU")
@pytest.mark.unit
def test_to_word_list_format_no_tokenizer():
    """Test to_word_list_format raises error when no tokenizer is provided."""
    try:
        import tensorrt_llm  # noqa: F401
    except ImportError:
        pytest.skip("Could not import TRTLLM helpers. tensorrt_llm is likely not installed")
        return

    from nemo_export.trt_llm.tensorrt_llm_run import to_word_list_format

    # Test that function raises assertion error when no tokenizer is provided
    word_dict = [["hello"]]
    with pytest.raises(AssertionError, match="need to set tokenizer"):
        to_word_list_format(word_dict, tokenizer=None)


@pytest.mark.run_only_on("GPU")
@pytest.mark.unit
def test_to_word_list_format_padding():
    """Test to_word_list_format padding behavior."""
    try:
        import tensorrt_llm  # noqa: F401
    except ImportError:
        pytest.skip("Could not import TRTLLM helpers. tensorrt_llm is likely not installed")
        return

    from nemo_export.trt_llm.tensorrt_llm_run import to_word_list_format

    # Create a mock tokenizer with different length tokens
    mock_tokenizer = MagicMock()
    mock_tokenizer.encode.side_effect = lambda x: {
        "<extra_id_1>": [100],
        "<extra_id_1>short": [100, 200],
        "<extra_id_1>verylongword": [100, 300, 301, 302, 303],
        "short": [200],
        "verylongword": [300, 301, 302, 303],
    }.get(x, [])

    # Test with words of different lengths
    word_dict = [["short"], ["verylongword"]]
    result = to_word_list_format(word_dict, tokenizer=mock_tokenizer)

    # Check that padding was applied correctly
    assert result.shape[0] == 2  # batch_size
    assert result.shape[1] == 2  # flat_ids and offsets
    assert result.shape[2] == 4  # Should be padded to max length (4 tokens for "verylongword")
    assert result.dtype == np.int32

    # Check that shorter sequences are padded with zeros
    flat_ids_0 = result[0, 0]
    assert 200 in flat_ids_0  # token for "short"
    assert 0 in flat_ids_0  # Should have padding zeros

    # Check that offsets are padded with -1
    offsets_0 = result[0, 1]
    assert -1 in offsets_0  # Should have padding -1s
