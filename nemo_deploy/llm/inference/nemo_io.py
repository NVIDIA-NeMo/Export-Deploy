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
"""IO utilities for loading NeMo 2.0 checkpoints without a direct nemo import.

Copied from the NeMo project (https://github.com/NVIDIA/NeMo).  Static
``from nemo import …`` statements are removed; the logic is otherwise
identical to the upstream sources.  When a NeMo checkpoint is actually
loaded at runtime, NeMo classes are imported transitively through
``pydoc.locate`` — NeMo must therefore still be installed to read NeMo
checkpoints.

Sources
  - IOProtocol         : nemo/lightning/io/capture.py
  - IO helpers, load   : nemo/lightning/io/mixin.py
  - load_context       : nemo/lightning/io/api.py
  - Torch-dtype fiddle
    registration       : nemo/lightning/io/fdl_torch.py
"""

from __future__ import annotations

import dataclasses
import functools
import inspect
import json
import logging
import threading
import uuid
from pathlib import Path
from pydoc import locate
from typing import Any, Dict, Generic, List, Optional, Protocol, TypeVar, runtime_checkable

import fiddle as fdl
import fiddle._src.experimental.dataclasses as fdl_dc
import torch
from cloudpickle import dump
from cloudpickle import load as pickle_load
from fiddle._src.experimental import serialization

_logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thread-local storage (mirrors nemo.lightning.io.mixin._thread_local)
# ---------------------------------------------------------------------------

_thread_local = threading.local()


def _set_thread_local_output_dir(path: Path) -> None:
    """Set output_dir in our thread-local and in NeMo's (if already imported).

    NeMo classes registered before our first load call will use NeMo's
    _io_unflatten_object, which reads from NeMo's own _thread_local.  We
    mirror the value there so that pickle-based artifacts still resolve
    correctly even in that edge case.
    """
    _thread_local.output_dir = path
    try:
        import sys

        nemo_mixin = sys.modules.get("nemo.lightning.io.mixin")
        if nemo_mixin is not None and hasattr(nemo_mixin, "_thread_local"):
            nemo_mixin._thread_local.output_dir = path
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Register torch dtypes as fiddle constants
# (from nemo.lightning.io.fdl_torch.enable — only register_constant calls
# are needed for deserialization; libcst / codegen parts are omitted)
# ---------------------------------------------------------------------------

_TORCH_DTYPE_NAMES = [
    "bool",
    "uint8",
    "int8",
    "int16",
    "int32",
    "int64",
    "float16",
    "bfloat16",
    "float32",
    "float64",
    "complex64",
    "complex128",
]


def _register_torch_dtypes() -> None:
    for name in _TORCH_DTYPE_NAMES:
        if hasattr(torch, name):
            serialization.register_constant("torch", name, compare_by_identity=True)


_register_torch_dtypes()

# ---------------------------------------------------------------------------
# IOProtocol  (from nemo.lightning.io.capture)
# ---------------------------------------------------------------------------

SelfT = TypeVar("SelfT", covariant=True)


@runtime_checkable
class IOProtocol(Protocol, Generic[SelfT]):
    @property
    def __io__(self) -> fdl.Config[SelfT]: ...  # noqa: E704


# ---------------------------------------------------------------------------
# IO helper functions  (from nemo.lightning.io.mixin)
# ---------------------------------------------------------------------------


def _io_transform_args(self, init_fn, *args, **kwargs) -> Dict[str, Any]:
    """Capture __init__ arguments as a plain dict for fdl.Config creation."""
    sig = inspect.signature(init_fn)
    bound_args = sig.bind_partial(self, *args, **kwargs)
    config_kwargs = {k: v for k, v in bound_args.arguments.items() if k != "self"}

    to_del: List[str] = []
    for key in config_kwargs:
        if isinstance(config_kwargs[key], IOProtocol):
            config_kwargs[key] = config_kwargs[key].__io__
        if dataclasses.is_dataclass(config_kwargs[key]):
            config_kwargs[key] = fdl_dc.convert_dataclasses_to_configs(config_kwargs[key], allow_post_init=True)
        if config_kwargs[key].__class__.__name__ == "_HAS_DEFAULT_FACTORY_CLASS":
            to_del.append(key)

    for key in to_del:
        del config_kwargs[key]

    return config_kwargs


def _io_init(self, **kwargs) -> fdl.Config:
    """Create an fdl.Config for *self* from captured init kwargs."""
    try:
        return fdl.Config(type(self), **kwargs)
    except Exception as e:
        raise RuntimeError(
            f"Error creating fdl.Config for {type(self).__name__}: {e}\nArguments that caused the error: {kwargs}"
        ) from e


def _io_wrap_init(cls):
    """Wrap cls.__init__ to populate __io__ on every instance."""
    original_init = cls.__init__

    if getattr(cls, "__wrapped_init__", False):
        return cls

    @functools.wraps(original_init)
    def wrapped_init(self, *args, **kwargs):
        if hasattr(self, "io_transform_args"):
            cfg_kwargs = self.io_transform_args(original_init, *args, **kwargs)
        else:
            cfg_kwargs = _io_transform_args(self, original_init, *args, **kwargs)
        if hasattr(self, "io_init"):
            self.__io__ = self.io_init(**cfg_kwargs)
        else:
            self.__io__ = _io_init(self, **cfg_kwargs)
        original_init(self, *args, **kwargs)

    cls.__init__ = wrapped_init
    cls.__wrapped_init__ = True
    return cls


def _io_flatten_object(instance):
    """Flatten an IOMixin object to a form fiddle can serialize."""
    try:
        serialization.dump_json(instance.__io__)
    except (serialization.UnserializableValueError, AttributeError) as exc:
        if not hasattr(_thread_local, "local_artifacts_dir") or not hasattr(_thread_local, "output_path"):
            raise exc

        local_artifact_path = Path(_thread_local.local_artifacts_dir) / f"{uuid.uuid4()}"
        output_path = _thread_local.output_path
        artifact_path = output_path / local_artifact_path
        with open(artifact_path, "wb") as f:
            dump(getattr(instance, "__io__", instance), f)
        return (str(local_artifact_path),), None

    return instance.__io__.__flatten__()


def _io_unflatten_object(values, metadata):
    """Unflatten an IOMixin object; load from pickle if it was saved that way."""
    if not hasattr(_thread_local, "output_dir"):
        return fdl.Config.__unflatten__(values, metadata)

    output_dir = _thread_local.output_dir
    if len(values) == 1:
        pickle_path = values[0]
        with open(Path(output_dir) / pickle_path, "rb") as f:
            return pickle_load(f)

    return fdl.Config.__unflatten__(values, metadata)


def _io_path_elements_fn(x):
    """Return the path elements for fiddle graph traversal."""
    try:
        serialization.dump_json(x.__io__)
    except (serialization.UnserializableValueError, AttributeError):
        return (serialization.IdentityElement(),)

    return x.__io__.__path_elements__()


def _io_register_serialization(cls) -> None:
    """Register fiddle traversal functions for *cls* using our _thread_local."""
    serialization.register_node_traverser(
        cls,
        flatten_fn=_io_flatten_object,
        unflatten_fn=_io_unflatten_object,
        path_elements_fn=_io_path_elements_fn,
    )


def track_io(target, artifacts=None):
    """Add fiddle IO functionality to a class or all eligible classes in a module.

    Copied from ``nemo.lightning.io.mixin.track_io``.
    """
    import types as _types

    def _add_io_to_class(cls):
        if inspect.isclass(cls) and hasattr(cls, "__init__") and not hasattr(cls, "__io__"):
            if cls in [str, int, float, tuple, list, dict, bool, type(None)]:
                return cls
            cls = _io_wrap_init(cls)
            _io_register_serialization(cls)
            cls.__io_artifacts__ = artifacts or []
        return cls

    def _is_in_module(obj, module):
        return obj.__module__ == module.__name__ or obj.__module__.startswith(f"{module.__name__}.")

    def _process_module(module):
        for name, obj in inspect.getmembers(module):
            if inspect.isclass(obj) and _is_in_module(obj, module):
                setattr(module, name, _add_io_to_class(obj))
        return module

    if isinstance(target, _types.ModuleType):
        return _process_module(target)
    elif inspect.isclass(target):
        return _add_io_to_class(target)
    else:
        raise TypeError("Target must be a module or a class")


def drop_unexpected_params(config: fdl.Config) -> bool:
    """Remove deprecated / unexpected parameters from a fiddle Config tree.

    Copied from ``nemo.lightning.io.mixin.drop_unexpected_params``.
    """
    updated = False

    def analyze(cfg, prefix: str):
        nonlocal updated
        if not isinstance(cfg, fdl.Config):
            return
        signature = inspect.signature(cfg.__fn_or_cls__)
        accept_kwargs = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in signature.parameters.values())
        if not accept_kwargs:
            to_drop = [p for p in cfg.__arguments__ if p not in signature.parameters]
            if to_drop:
                updated = True
                _logger.warning("Deprecated parameters to drop from %s: %s", prefix, to_drop)
                for p in to_drop:
                    del cfg.__arguments__[p]
        for key, value in cfg.__arguments__.items():
            analyze(value, f"{prefix}.{key}")

    analyze(config, "<root>")
    return updated


def _artifact_transform_load(cfg: fdl.Config, path: Path) -> None:
    """Rewrite artifact paths stored in a config to absolute paths.

    Copied from ``nemo.lightning.io.mixin._artifact_transform_load``.
    """
    for artifact in getattr(cfg.__fn_or_cls__, "__io_artifacts__", []):
        current_val = getattr(cfg, artifact.attr)
        if isinstance(current_val, fdl.Config):
            setattr(cfg, artifact.attr, fdl.build(current_val).attr)
            continue
        if artifact.skip:
            continue
        current_val = getattr(cfg, artifact.attr)
        if current_val is None:
            continue
        new_val = str(Path(path) / current_val)
        setattr(cfg, artifact.attr, new_val)

    for attr in dir(cfg):
        try:
            child = getattr(cfg, attr)
            if isinstance(child, fdl.Config):
                _artifact_transform_load(child, path=path)
        except (ValueError, AttributeError):
            pass


# ---------------------------------------------------------------------------
# load  (from nemo.lightning.io.mixin)
# ---------------------------------------------------------------------------


def load(
    path: Path,
    output_type: Any = None,
    subpath: Optional[str] = None,
    build: bool = True,
) -> Any:
    """Load a fiddle-serialised NeMo checkpoint context from an ``io.json`` file.

    Copied from ``nemo.lightning.io.mixin.load``.
    """
    _path = Path(path)
    _set_thread_local_output_dir(_path)

    if _path.is_dir():
        _path = _path / "io.json"

    if not _path.is_file():
        raise FileNotFoundError(f"No such file: '{_path}'")

    if subpath:
        subpath = "<root>." + subpath

    # Register / re-register fiddle traversal for every class in the JSON.
    # We always re-register (not just when missing) so that our _thread_local
    # is used by _io_unflatten_object for classes that NeMo may have already
    # registered before this call.
    with open(_path) as f:
        j = json.load(f)

    for obj, val in j.get("objects", {}).items():
        clss = ".".join([val["type"]["module"], val["type"]["name"]])
        if subpath and "paths" in val:
            if all(subpath not in p for p in val["paths"]):
                continue
        cls_obj = locate(clss)
        if cls_obj is None:
            continue
        if not serialization.find_node_traverser(cls_obj):
            track_io(cls_obj)
        else:
            # Re-register with our traversal so our _thread_local is active.
            _io_register_serialization(cls_obj)

    with open(_path, "rb") as f:
        json_config = json.loads(f.read())

    root_key = None
    for obj, val in json_config.get("objects", {}).items():
        if "paths" in val and subpath in val["paths"]:
            root_key = obj
            break

    if subpath and not root_key:
        _logger.warning("Could not find %s for %s in %s", subpath, output_type, _path)

    if root_key:
        json_config["root"]["key"] = root_key

    config = serialization.Deserialization(json_config).result
    _artifact_transform_load(config, path)
    drop_unexpected_params(config)

    if not build:
        return config

    return fdl.build(config)


# ---------------------------------------------------------------------------
# load_context  (from nemo.lightning.io.api)
# ---------------------------------------------------------------------------


def load_context(path: Path, subpath: Optional[str] = None, build: bool = True) -> Any:
    """Load a NeMo TrainerContext (or a subpath of it) from a checkpoint directory.

    Copied from ``nemo.lightning.io.api.load_context``.
    """
    if not isinstance(path, Path):
        path = Path(path)

    try:
        return load(path, subpath=subpath, build=build)
    except FileNotFoundError:
        # Backwards compatibility: checkpoints without a ``/context`` sub-dir.
        if path.parts[-1] == "context":
            path = path.parent
        else:
            path = path / "context"
        return load(path, subpath=subpath, build=build)
