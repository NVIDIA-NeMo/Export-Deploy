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

import os
import re
import torch
import shutil
import inspect
import logging
import numpy as np

from typing import (
    Callable,
    Optional,
    Generic,
    TypeVar,
    Union,
    Tuple,
    Dict,
    List,
    Any,
    overload,
)

from pathlib import Path, WindowsPath, PosixPath
from filelock import FileLock, Timeout
from nemo.utils import logging
from dataclasses import dataclass
from torch import nn


if os.name == 'nt':
    BasePath = WindowsPath
else:
    BasePath = PosixPath

DEFAULT_NEMO_CACHE_HOME = Path.home() / ".cache" / "nemo"
NEMO_CACHE_HOME = Path(os.getenv("NEMO_HOME", DEFAULT_NEMO_CACHE_HOME))
DEFAULT_NEMO_DATASETS_CACHE = NEMO_CACHE_HOME / "datasets"
NEMO_DATASETS_CACHE = Path(os.getenv("NEMO_DATASETS_CACHE", DEFAULT_NEMO_DATASETS_CACHE))
DEFAULT_NEMO_MODELS_CACHE = NEMO_CACHE_HOME / "models"
NEMO_MODELS_CACHE = Path(os.getenv("NEMO_MODELS_CACHE", DEFAULT_NEMO_MODELS_CACHE))


SourceT = TypeVar("SourceT")
TargetT = TypeVar("TargetT")
SourceModuleT = TypeVar("SourceModuleT", bound=nn.Module)
TargetModuleT = TypeVar("TargetModuleT", bound=nn.Module)
F = TypeVar("F", bound=Callable[..., Any])


def extract_dtypes(ckpt):
    """
    Extracts dtype from the input iterator
    ckpt can be module.named_parameters or module.state_dict().items()
    """
    dtypes = {}
    for key, val in ckpt:
        if hasattr(val, 'dtype'):
            dtypes[key] = val.dtype
        elif hasattr(val, 'data') and hasattr(val.data, 'dtype'):
            # if it's ShardedTensor populated with data.
            dtypes[key] = val.data.dtype
    return dtypes


@dataclass
class TransformCTX:
    """Transform Data class Definition."""

    source: nn.Module
    source_state: dict
    target: nn.Module
    target_state: dict


class _ModelState:
    """
    Helper class for used for to modify state dict of a source model during model conversion.
    """

    def __init__(self, state_dict, config=None):
        self._state_dict = state_dict
        self.config = config

    def state_dict(self):
        # pylint: disable=C0115,C0116
        return self._state_dict

    def to(self, dtype):
        # pylint: disable=C0115,C0116
        for k, v in self._state_dict.items():
            if v.dtype != dtype:
                logging.warning(f"Converting {k} from {v.dtype} (source model) to {dtype} (target model)")
            self._state_dict[k] = v.to(dtype)


@torch.no_grad
def apply_transforms(
    source: Union[nn.Module, _ModelState],
    target: TargetModuleT,
    mapping: Dict[str, str],
    transforms: Optional[List[Callable[[TransformCTX], TransformCTX]]] = [],
    state_dict_ignored_entries: List = [],
) -> TargetModuleT:
    """
    Applies a series of transformations to adapt the state dictionary of a source module to
    match the structure of a target module's state dictionary.

    This function renames keys according to a provided mapping and modifies values using a list
    of transformation functions. Each transformation function typically is decorated
    with `io.state_transform`.

    Args:
        source (nn.Module): The source module from which parameters and buffers are taken.
        target (TargetModuleT): The target module to which parameters and buffers are adapted.
        mapping (Dict[str, str]): Key-value pairs where each key from the source state dictionary
            is mapped to a corresponding key in the target state dictionary.
        transforms (Optional[List[Callable[[TransformCTX], TransformCTX]]]): A list of functions
            that modify the `TransformCTX` object. If None, no transformations beyond key renaming
            are applied. Defaults to None.
        state_dict_ignored_entries: List of entries to ignore in _target.state_dict(). There are cases
            where multiple entries in model's state_dict point to one entry in model's named_parameter.
            E.g., model has multiple pointers pointing to one shared parameters (`encoder.embed_tokens.weight`,
            `decoder.embed_tokens.weight` and `shared.weight` all points to `shared.weight
            in T5 Huggingface implementation.). In these cases, ignore redundant entries.

    Returns
    -------
        TargetModuleT: The modified target module with its state dictionary adjusted according to
        the specified mappings and transformations.

    Raises
    ------
        ValueError: If there's a mismatch in shape between corresponding source and target parameters
            or buffers.
        RuntimeError: If the target state dictionary contains keys that are not present in the source
            state dictionary after all transformations.

    Examples
    --------
        >>> source_module = nn.Linear(10, 5)
        >>> target_module = nn.Linear(10, 5)
        >>> mapping = {'weight': 'weights', 'bias': 'biases'}
        @io.state_transform(
            source_key="weight",
            target_key="weights"
        )
        def scale_weights(ctx):
            ctx.target_state['weights'] = ctx.source_state['weight'] * 2
            return ctx
        >>> transformed_target = apply_transforms(
        ...     source_module, target_module, mapping, [scale_weights]
        ... )
        >>> print(transformed_target.state_dict()['weights'])

    See Also
    --------
        - `TransformCTX`: For more details on the context object used in transformations.
        - `StateDictTransform`: For creating complex transformations.

    Note:
        This function is particularly useful when adapting models from different frameworks or
        when consolidating models with different architectural changes.
    """
    from megatron.core.transformer.module import MegatronModule

    # TODO: How can we improve this?
    _source = source
    if hasattr(source, "module") and isinstance(source.module, MegatronModule):
        _source = source.module
    _target = target
    if hasattr(target, "module") and isinstance(target.module, MegatronModule):
        _target = target.module

    # Track dtypes to make sure they weren't modified during conversion.
    target_orig_dtypes = extract_dtypes(_target.named_parameters())

    target_state = _target.state_dict()
    ctx = TransformCTX(
        source=_source,
        source_state=_source.state_dict(),
        target=_target,
        target_state=target_state,
    )

    for key, val in mapping.items():
        logging.debug(f"Mapping {key} -> {val}")
        ctx = StateDictTransform(key, val)(ctx)

    for transform in transforms:
        logging.debug(f"Transforming {transform.source_key} -> {transform.target_key}")
        ctx = transform(ctx)

    _params: Dict[str, nn.Parameter] = {}
    for name, param in _target.named_parameters():
        if name in target_state:
            target_param = target_state[name]
            if param.data.shape != target_param.shape:
                raise ValueError(
                    f"Shape mismatch for parameter {name}: target shape {param.shape} vs "
                    f"converted source shape {target_param.shape}"
                )

            _params[name] = nn.Parameter(target_param, requires_grad=param.requires_grad)
            target_state.pop(name)
        else:
            print(f"Unexpected key: {name} not in checkpoint but in model.")

    for key, val in _params.items():
        _module, _key = _target, key
        if "." in key:
            for part in key.split(".")[:-1]:
                _module = getattr(_module, part)
            _key = key.split(".")[-1]

        _module.register_parameter(_key, val)

    _buffers = {}
    for name, buffer in _target.named_buffers():
        if name in target_state:
            if buffer.shape != target_state[name].shape:
                raise ValueError(f"Shape mismatch for buffer {name}: {buffer.shape} vs {target_state[name].shape}")

            _buffers[name] = nn.Parameter(target_state[name], requires_grad=False)
            target_state.pop(name)

    for key, val in _buffers.items():
        _module, _key = _target, key
        if "." in key:
            for part in key.split(".")[:-1]:
                _module = getattr(_module, part)
            _key = key.split(".")[-1]

        _module.register_buffer(_key, val)

    keys = list(filter(lambda x: x is not None and not x.endswith("_extra_state"), target_state.keys()))
    keys = [key for key in keys if key not in state_dict_ignored_entries]
    if len(keys) != 0:
        raise RuntimeError(f"Additional keys: {keys} in checkpoint but not in model.")

    # TODO: Is this correct?
    # for key in target.state_dict():
    #     if key.endswith("_extra_state"):
    #         del target.state_dict()[key]

    """finally:
        cls._set_model_restore_state(is_being_restored=False)"""

    meta_tensor_keys = []
    for name, param in target.named_parameters():
        if param.is_meta:
            meta_tensor_keys.append(name)

    assert not meta_tensor_keys, (
        f"{meta_tensor_keys}\nThere are meta tensors in the model after conversion."
        f"Did you forget to include these parameters in the mapping or transforms in `convert_state`?"
    )

    assert target_orig_dtypes == extract_dtypes(_target.named_parameters()), (
        f"dtype mismatch between source and target state dicts. "
        f"Left side is { {k: v for k, v in target_orig_dtypes.items() if v!=torch.bfloat16} }, "
        f"Right side is { {k: v for k, v in extract_dtypes(_target.named_parameters()).items() if v!=torch.bfloat16} }"
    )
    if hasattr(target, "module") and isinstance(target.module, MegatronModule):
        target.module = _target

        return target

    return _target


def _default_transform(inp):
    return inp


class StateDictTransform(Generic[F]):
    """
    A transformation class for state dictionaries, allowing for flexible key matching and
    transformation of values between source and target state dictionaries.

    Attributes
    ----------
        source_key: A string, tuple of strings, or a dictionary specifying the keys in the source
            state dictionary to match. Wildcards (*) are supported.
        target_key: A string or tuple of strings specifying the keys in the target state dictionary
            to match. Wildcards (*) are supported.
        transform: A callable that performs the transformation on matched keys' values.

    Examples
    --------
        >>> def example_transform(ctx, *args):
        ...     return sum(args)
        >>> transform = StateDictTransform(
        ...     source_key="model.layers.*.self_attn.*_proj.weight",
        ...     target_key="decoder.layers.*.self_attention.linear_qkv.weight",
        ...     transform=example_transform
        ... )
    """

    def __init__(
        self,
        source_key: Union[str, Tuple[str, ...], Dict[str, str]],
        target_key: Union[str, Tuple[str, ...]],
        transform: F = _default_transform,
    ):
        self.source_key = source_key
        self.target_key = target_key
        self.transform = transform

    def __call__(self, ctx: TransformCTX) -> TransformCTX:
        source_key = self.source_key
        target_key = self.target_key
        source_dict, target_dict = ctx.source_state, ctx.target_state

        fn_params = dict(inspect.signature(self.transform).parameters)
        fn_params.pop("ctx", None)

        if isinstance(source_key, (dict, tuple)):
            if isinstance(source_key, tuple):
                source_key_dict = {param: source_key[i] for i, param in enumerate(fn_params)}
            else:
                source_key_dict = source_key
            source_matches_dict = {k: _match_keys(list(source_dict.keys()), v) for k, v in source_key_dict.items()}
            target_matches = _match_keys(list(target_dict.keys()), target_key)
            param_names = list(filter(lambda x: x in source_matches_dict, fn_params))
            source_matches = [
                source_matches_dict[v] if source_matches_dict[v].ndim > 0 else [source_matches_dict[v].item()]
                for v in param_names
            ]
            target_matches = [target_matches if target_matches.ndim > 0 else [target_matches.item()]]
            for layer_names_group in zip(*(source_matches + target_matches)):
                # Wrap in a list if it's a single layer (ie non-expert)
                if isinstance(layer_names_group[0], str):
                    layer_names_group = [[x] for x in layer_names_group]
                for layer_names in zip(*layer_names_group):
                    target_dict[layer_names[-1]] = self.call_transform(
                        ctx, **dict(zip(param_names, [source_dict[x] for x in layer_names[:-1]]))
                    )
                logging.debug(f"Matched (transform)! {layer_names_group=}")
        else:
            source_keys = list(source_dict.keys())
            target_keys = list(target_dict.keys())

            source_matches = _match_keys(source_keys, source_key)
            if source_matches.size == 1 and source_matches == np.array(None):
                raise ValueError(f"No matches found for source key: {source_key}")

            if isinstance(target_key, str):
                target_matches = _match_keys(target_keys, target_key)
                if target_matches.size == 1 and target_matches == np.array(None):
                    raise ValueError(f"No matches found for target key: {target_key}")
            else:
                if isinstance(target_key, dict):
                    raise ValueError("Target key must be a string or a tuple of strings.")
                _matches = [_match_keys(target_keys, key) for key in target_key]
                target_matches = np.stack(_matches, axis=-1)

            # Determine if we are dealing with multiple source matches or multiple target matches
            multiple_sources = source_matches.ndim >= target_matches.ndim
            accepts_var_args = any(
                param.kind == param.VAR_POSITIONAL for param in inspect.signature(self.transform).parameters.values()
            )

            if multiple_sources:
                for target_index, target_match in np.ndenumerate(target_matches):
                    try:
                        source_match = source_matches[target_index]
                    except IndexError as e:
                        logging.error(f"Enountered IndexError during transform.\n{source_matches=}\n{target_matches=}")
                        raise e
                    if accepts_var_args:
                        source_values = [source_dict[k] for k in source_match]
                        target_dict[target_match] = self.call_transform(ctx, *source_values)
                    else:
                        _source_match_list = [source_match] if isinstance(source_match, str) else list(source_match)
                        if len(fn_params) != len(_source_match_list):
                            raise ValueError(
                                f"Mismatch between source and target keys: {source_match} vs {target_match}"
                            )

                        kwargs = {param: source_dict[k] for param, k in zip(fn_params, _source_match_list)}
                        target_dict[target_match] = self.call_transform(ctx, **kwargs)
                    logging.debug(f"Matched (multi source)! {target_match=} {source_match=}")
            else:
                for source_index, source_match in np.ndenumerate(source_matches):
                    target_match = target_matches[source_index]
                    source_values = (
                        [source_dict[source_match]]
                        if np.isscalar(source_match)
                        else [source_dict[k] for k in source_match]
                    )
                    if accepts_var_args:
                        outputs = self.call_transform(ctx, *source_values)
                    else:
                        kwargs = {param: val for param, val in zip(fn_params, source_values)}
                        outputs = self.call_transform(ctx, **kwargs)

                    if isinstance(target_match, str):
                        target_dict[target_match] = outputs
                    else:
                        for i, t in enumerate(outputs):
                            target_dict[target_match[i]] = t
                    logging.debug(f"Matched (single source)! {target_match=} {source_match=}")

        return ctx

    def call_transform(self, ctx: TransformCTX, *args, **kwargs):
        """Perform transform and check if the given args valid."""
        func_params = inspect.signature(self.transform).parameters
        expected_num_args = len([p for p in func_params if p not in ['self', 'ctx']])
        provided_num_args = len(args) + len(kwargs)
        accepts_var_args = any(param.kind == param.VAR_POSITIONAL for param in func_params.values())

        if not accepts_var_args and provided_num_args != expected_num_args:
            raise ValueError(
                f"Expected {expected_num_args} arguments for the transformation function, but got {provided_num_args}."
            )

        if 'ctx' in func_params:
            return self.transform(ctx, *args, **kwargs)

        return self.transform(*args, **kwargs)


def _match_keys(keys: List[str], pattern: str) -> np.ndarray:
    escaped_pattern = ''
    i = 0
    wildcard_positions = []
    while i < len(pattern):
        if pattern[i : i + 2] == '**':
            escaped_pattern += r'(.+)'  # Match any characters including dots
            wildcard_positions.append('**')
            i += 2
        elif pattern[i] == '*':
            escaped_pattern += r'([^.]+)'  # Match any characters except dots
            wildcard_positions.append('*')
            i += 1
        else:
            if pattern[i] == '.':
                escaped_pattern += r'\.'  # Escape the dot
            else:
                escaped_pattern += pattern[i]
            i += 1

    regex_pattern = re.compile("^" + escaped_pattern + "$")
    num_wildcards = len(wildcard_positions)
    wildcard_matches = [[] for _ in range(num_wildcards)]

    for key in filter(lambda x: x is not None, keys):
        match = regex_pattern.match(key)
        if match:
            for i, group in enumerate(match.groups()):
                if group not in wildcard_matches[i]:
                    wildcard_matches[i].append(group)

    # Sort the wildcard matches to maintain consistent ordering
    for i in range(len(wildcard_matches)):
        wildcard_matches[i].sort(key=lambda x: int(x) if x.isdigit() else x)

    # Determine the shape of the output array based on the unique matches for each wildcard
    shape = [len(matches) for matches in wildcard_matches]

    if len(wildcard_matches) == 0:
        # If there is no wildcard matches, assuming it is a single match
        shape = [1]
    # Initialize an empty array with the determined shape
    output_array = np.empty(shape, dtype=object)

    # Populate the array with the keys, now that we have the correct shape and ordering
    for key in filter(lambda x: x is not None, keys):
        match = regex_pattern.match(key)
        if match:
            # Convert match groups to indices based on their position in wildcard_matches
            indices = [wildcard_matches[i].index(group) for i, group in enumerate(match.groups())]
            output_array[tuple(indices)] = key  # Place the key in the array based on the indices

    return output_array


@overload
def state_transform(
    source_key: Union[str, Tuple[str, ...], Dict[str, str]],
    target_key: Union[str, Tuple[str, ...]],
) -> Callable[[F], StateDictTransform[F]]: ...


@overload
def state_transform(
    source_key: Union[str, Tuple[str, ...], Dict[str, str]], target_key: Union[str, Tuple[str, ...]], fn: F
) -> StateDictTransform[F]: ...


def state_transform(
    source_key: Union[str, Tuple[str, ...], Dict[str, str]],
    target_key: Union[str, Tuple[str, ...]],
    fn: Optional[F] = None,
):
    """
    A decorator for creating StateDictTransform instances with specified source and target keys,
    and a transformation function. This allows for concise definition of state dictionary
    transformations.

    Args:
        source_key: A string, tuple of strings, or a dictionary specifying the keys in the source
            state dictionary to match. Wildcards (*) are supported.
        target_key: A string or tuple of strings specifying the keys in the target state dictionary
            to match. Wildcards (*) are supported.
        fn: An optional callable that performs the transformation on matched keys' values. If not
            provided, the decorator can be used to wrap a function definition.

    Returns
    -------
        A StateDictTransform instance if `fn` is provided, otherwise returns a decorator that
        takes a function and returns a StateDictTransform instance.

    Examples
    --------
        >>> @state_transform(
        ...     source_key="model.layers.*.self_attn.*_proj.weight",
        ...     target_key="decoder.layers.*.self_attention.linear_qkv.weight"
        ... )
        ... def sum_transform(ctx, *args):
        ...     return sum(args)
    """

    def wrapper(fn) -> StateDictTransform:
        return StateDictTransform(source_key, target_key, fn)

    if fn is None:
        return wrapper

    return wrapper(fn)


class TransformFns:
    """
    A collection of common functions used in state dict transformation.
    """

    @staticmethod
    def split_qkv(ctx: TransformCTX, linear_qkv: torch.Tensor):
        """
        Split interleave-concatenated qkv to q, k, v

        Example: export layer linear_qkv to HF {q|k|v}_proj
        """
        megatron_config = ctx.source.config

        head_num = megatron_config.num_attention_heads
        num_query_groups = megatron_config.num_query_groups
        heads_per_group = head_num // num_query_groups
        # hidden_size = megatron_config.hidden_size
        head_size = megatron_config.kv_channels
        qkv_total_dim = head_num + 2 * num_query_groups

        linear_qkv = linear_qkv.reshape([qkv_total_dim, head_size, -1])
        # when converting base model (linear_qkv), hidden size = megatron_config.hidden_size
        # when converting lora (linear_qkv.adapter.linear_out), hidden size = lora_r
        hidden_size = linear_qkv.size(-1)
        q_slice = torch.cat(
            [
                torch.arange((heads_per_group + 2) * i, (heads_per_group + 2) * i + heads_per_group)
                for i in range(num_query_groups)
            ]
        )
        k_slice = torch.arange(heads_per_group, qkv_total_dim, (heads_per_group + 2))
        v_slice = torch.arange(heads_per_group + 1, qkv_total_dim, (heads_per_group + 2))

        q_proj = linear_qkv[q_slice].reshape(-1, hidden_size).cpu()
        k_proj = linear_qkv[k_slice].reshape(-1, hidden_size).cpu()
        v_proj = linear_qkv[v_slice].reshape(-1, hidden_size).cpu()

        return q_proj, k_proj, v_proj

    @staticmethod
    def split_fc1(linear_fc1: torch.Tensor):
        """
        Split concatenated fc1 to gate and up proj

        Example: export layer linear_fc1 to HF {gate|up}_proj
        """
        gate_proj, up_proj = torch.chunk(linear_fc1, 2, dim=0)
        return gate_proj, up_proj


class Connector(BasePath, Generic[SourceT, TargetT]):
    """
    A generic connector class that provides a framework for transforming a source type (SourceT)
    to a target type (TargetT) while handling file paths based on the operating system.

    Attributes
    ----------
        default_path (Optional[Path]): A default path used when no path is explicitly provided.

    Methods
    -------
        init() -> TargetT:
            Should be implemented to initialize the target type from the source type.

        apply(output_path: Path) -> Path:
            Should be implemented to apply the transformation and save the result at the output path.

        __new__(cls, *args, **kwargs) -> 'Connector':
            Creates a new instance of the connector, using default_path if no path is provided.

        __call__(output_path: Optional[Path] = None, overwrite: bool = False) -> Path:
            Processes the transformation and handles file operations like overwriting.

        local_path(base_path: Optional[Path] = None) -> Path:
            Computes the local path for storage based on a base path or a default cache home.

        is_in_cache(base_path: Optional[Path] = None) -> bool:
            Checks if the transformed data is already cached at the specified base path.
    """

    default_path = None
    LOCK_TIMEOUT = 1200

    def init(self) -> TargetT:
        """Should be implemented to initialize the target type from the source type."""
        raise NotImplementedError()

    def apply(self, output_path: Path, **kwargs) -> Path:
        """Should be implemented to apply the transformation and save the result at the output path."""
        raise NotImplementedError()

    def __new__(cls, *args, **kwargs):
        if cls.default_path is not None and not args and 'path' not in kwargs:
            # If default_path is set and no arguments are provided, use default_path as the argument
            return super().__new__(cls, cls.default_path)

        return super().__new__(cls, *args, **kwargs)

    def __call__(self, output_path: Optional[Path] = None, overwrite: bool = False, **kwargs) -> Path:
        _output_path = output_path or self.local_path()
        lock_path = _output_path.with_suffix(_output_path.suffix + '.lock')
        lock = FileLock(lock_path)

        # Check if the lock file exists and set overwrite to False if it does
        if lock_path.exists():
            overwrite = False

        try:
            with lock.acquire(timeout=self.LOCK_TIMEOUT):
                if overwrite and _output_path.exists():
                    shutil.rmtree(_output_path)

                if not _output_path.exists():
                    to_return = self.apply(_output_path, **kwargs)
                    _output_path = to_return or _output_path

        except Timeout:
            logging.error(f"Timeout occurred while trying to acquire the lock for {_output_path}")
            raise

        except Exception as e:
            logging.error(f"An error occurred: {e}")
            raise

        finally:
            # Delete the lock file if it exists
            if lock_path.exists():
                try:
                    os.remove(lock_path)
                except OSError as e:
                    logging.warning(f"Failed to remove lock file {lock_path}: {e}")

        return _output_path

    def local_path(self, base_path: Optional[Path] = None) -> Path:
        """Computes the local path for storage based on a base path or a default cache home."""
        if base_path:
            _base = base_path
        else:
            _base = Path(NEMO_CACHE_HOME)

        return _base / str(self).replace("://", "/")

    def is_in_cache(self, base_path: Optional[Path] = None) -> bool:
        """Checks if the transformed data is already cached at the specified base path."""
        return self.local_path(base_path=base_path).exists()


class ModelConnector(Connector, Generic[SourceT, TargetT]):
    """
    A specialized connector that extends the generic Connector to handle model-specific operations
    such as setup, save, and load using the Lightning framework.

    Methods
    -------
        nemo_setup(model: pl.LightningModule, trainer: Optional[pl.Trainer] = None) -> pl.Trainer:
            Sets up the model and trainer using a specified strategy, preparing it for training or inference.

        nemo_save(output_path: Path, trainer: pl.Trainer):
            Saves the model's state to the specified path using the trainer's current strategy.

        nemo_load(path: Path, trainer: Optional[pl.Trainer] = None, cpu: bool = True) -> Tuple[Any, pl.Trainer]:
            Loads a model from the specified path, optionally using a CPU-focused strategy, and returns the model and
            trainer.
    """

    def local_path(self, base_path: Optional[Path] = None) -> Path:
        if base_path:
            _base = Path(base_path)
        else:
            _base = Path(NEMO_MODELS_CACHE)

        # If the user supplied `hf:///path/to/downloaded/my-model/`
        # then extract the last dir-name (i.e. my-model) and append it to _base
        if str(self).startswith('/'):
            if self.suffix in ['.pt', '.pth']:
                return _base / self.parent.name
            return _base / self.name
        return _base / str(self).replace("://", "/")

    def on_import_ckpt(self, model):
        """Called after checkpoint is imported"""
        if hasattr(self, "tokenizer"):
            model.tokenizer = self.tokenizer
            if hasattr(model, "__io__") and hasattr(self.tokenizer, '__io__'):
                model.__io__.tokenizer = self.tokenizer.__io__

    def save_hf_tokenizer_assets(self, tokenizer_name_or_path, save_path="/tmp/nemo_tokenizer"):
        """Save HF tokenizer to the imported NeMo model"""
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(tokenizer_name_or_path, trust_remote_code=True)
        # Save tokenizer assets to save_path.
        tok.save_pretrained(save_path)
        return save_path


def export_ckpt(
    path: Path,
    target: str,
    load_connector: Callable[[Path, str], ModelConnector],
    output_path: Optional[Path] = None,
    overwrite: bool = False,
    **kwargs,
) -> Path:
    """
    Exports a checkpoint from a model using the model's associated exporter, typically for
    the purpose of sharing a model that has been fine-tuned or customized within NeMo.
    This function leverages the ConnectorMixin interface to seamlessly integrate
    the model's state into an external checkpoint format.

    The exporter component of the model reads the model's state from the specified path and
    exports it into the format specified by the 'target' identifier. This is particularly
    useful for adapting models that have been developed or fine-tuned within the current system
    to be compatible with other environments or frameworks. The function allows for specifying
    an output path for the exported checkpoint; if not provided, the exporter's default path
    will be used. The 'overwrite' parameter enables the replacement of existing data at the
    output path, which is useful when updating models with new data and discarding old checkpoint
    files.

    Args:
        path (Path): The path to the model's checkpoint file from which data will be exported.
        target (str): The identifier for the exporter that defines the format of the export.
        output_path (Optional[Path]): The path where the exported checkpoint will be saved.
            If not specified, the exporter's default path is used.
        overwrite (bool): If set to True, existing files at the output path will be overwritten.
            This is useful for model updates where retaining old checkpoint files is not required.
        load_connector (Callable[[Path, str], ModelConnector]): A function to load the appropriate
            exporter based on the model and target format.

    Returns
    -------
        Path: The path where the checkpoint has been saved after export. This path is determined
            by the exporter, based on the provided output_path and its internal logic.

    Raises
    ------
        ValueError: If the model does not implement ConnectorMixin, indicating a lack of
            necessary exporter functionality.

    Example:
        nemo_ckpt_path = Path("/path/to/model.ckpt")
        export_path = export_ckpt(nemo_ckpt_path, "hf")
    """
    exporter: ModelConnector = load_connector(path, target)
    _output_path = output_path or Path(path) / target

    return exporter(overwrite=overwrite, output_path=_output_path, **kwargs)
