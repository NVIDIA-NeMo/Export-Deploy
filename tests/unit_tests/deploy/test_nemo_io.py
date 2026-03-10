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

import json
import sys
import types
from pathlib import Path
from unittest.mock import MagicMock, patch

import fiddle as fdl
import pytest

from nemo_deploy.llm.inference.nemo_io import (
    IOProtocol,
    _artifact_transform_load,
    _io_flatten_object,
    _io_init,
    _io_path_elements_fn,
    _io_register_serialization,
    _io_transform_args,
    _io_unflatten_object,
    _io_wrap_init,
    _set_thread_local_output_dir,
    _thread_local,
    drop_unexpected_params,
    load,
    load_context,
    track_io,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _SimpleClass:
    """A minimal class used across multiple tests."""

    def __init__(self, x=1, y=2):
        self.x = x
        self.y = y


class _ClassWithIO:
    """A class that implements IOProtocol (has __io__ attribute)."""

    def __init__(self):
        self.__io__ = MagicMock()


# ---------------------------------------------------------------------------
# _set_thread_local_output_dir
# ---------------------------------------------------------------------------


class TestSetThreadLocalOutputDir:
    def test_sets_output_dir_on_thread_local(self):
        p = Path("/some/path")
        _set_thread_local_output_dir(p)
        assert _thread_local.output_dir == p

    def test_mirrors_to_nemo_module_when_present(self):
        """When nemo.lightning.io.mixin is already in sys.modules the value should be mirrored."""
        fake_mixin = types.SimpleNamespace(_thread_local=types.SimpleNamespace())
        p = Path("/mirror/path")
        with patch.dict(sys.modules, {"nemo.lightning.io.mixin": fake_mixin}):
            _set_thread_local_output_dir(p)
        assert fake_mixin._thread_local.output_dir == p

    def test_does_not_raise_when_nemo_module_missing(self):
        """Should not raise even when nemo is not imported."""
        # Remove only the nemo mixin key if present; don't wipe all of sys.modules
        with patch.dict(sys.modules, {"nemo.lightning.io.mixin": None}):
            _set_thread_local_output_dir(Path("/no/nemo"))

    def test_suppresses_exception_from_nemo_mirror(self):
        """An exception thrown while mirroring must be swallowed."""

        class _BrokenMixin:
            @property
            def _thread_local(self):
                raise RuntimeError("boom")

        broken_mixin = _BrokenMixin()
        with patch.dict(sys.modules, {"nemo.lightning.io.mixin": broken_mixin}):
            # Should not raise
            _set_thread_local_output_dir(Path("/ok"))


# ---------------------------------------------------------------------------
# IOProtocol
# ---------------------------------------------------------------------------


class TestIOProtocol:
    def test_class_with_io_attr_is_instance(self):
        obj = _ClassWithIO()
        assert isinstance(obj, IOProtocol)

    def test_class_without_io_attr_is_not_instance(self):
        assert not isinstance(_SimpleClass(), IOProtocol)


# ---------------------------------------------------------------------------
# _io_transform_args
# ---------------------------------------------------------------------------


class TestIoTransformArgs:
    def test_captures_kwargs(self):
        instance = object.__new__(_SimpleClass)
        result = _io_transform_args(instance, _SimpleClass.__init__, x=10, y=20)
        assert result == {"x": 10, "y": 20}

    def test_captures_positional_args(self):
        instance = object.__new__(_SimpleClass)
        result = _io_transform_args(instance, _SimpleClass.__init__, 5, 6)
        assert result == {"x": 5, "y": 6}

    def test_replaces_io_protocol_with_io_attr(self):
        instance = object.__new__(_SimpleClass)
        io_obj = _ClassWithIO()
        mock_io = MagicMock(spec=fdl.Config)
        io_obj.__io__ = mock_io
        result = _io_transform_args(instance, _SimpleClass.__init__, x=io_obj, y=2)
        assert result["x"] is mock_io

    def test_removes_default_factory_entries(self):
        """Arguments whose value has class name _HAS_DEFAULT_FACTORY_CLASS are dropped."""

        class _HAS_DEFAULT_FACTORY_CLASS:
            pass

        factory_sentinel = _HAS_DEFAULT_FACTORY_CLASS()
        instance = object.__new__(_SimpleClass)
        result = _io_transform_args(instance, _SimpleClass.__init__, x=factory_sentinel, y=3)
        assert "x" not in result
        assert result["y"] == 3


# ---------------------------------------------------------------------------
# _io_init
# ---------------------------------------------------------------------------


class TestIoInit:
    def test_creates_fdl_config_for_class(self):
        instance = _SimpleClass(x=7, y=8)
        cfg = _io_init(instance, x=7, y=8)
        assert isinstance(cfg, fdl.Config)
        assert cfg.__fn_or_cls__ is _SimpleClass

    def test_raises_runtime_error_on_invalid_kwarg(self):
        instance = _SimpleClass()
        with pytest.raises(RuntimeError, match="Error creating fdl.Config"):
            _io_init(instance, not_a_valid_param=99)


# ---------------------------------------------------------------------------
# _io_wrap_init
# ---------------------------------------------------------------------------


class TestIoWrapInit:
    def test_wraps_init_and_sets_io(self):
        class _Target:
            def __init__(self, val=42):
                self.val = val

        _io_wrap_init(_Target)
        obj = _Target(val=10)
        assert hasattr(obj, "__io__")
        assert isinstance(obj.__io__, fdl.Config)
        assert obj.val == 10

    def test_idempotent_double_wrap(self):
        class _Target2:
            def __init__(self, val=1):
                self.val = val

        _io_wrap_init(_Target2)
        wrapped_once = _Target2.__init__
        _io_wrap_init(_Target2)
        # Second call must be a no-op (guard via __wrapped_init__)
        assert _Target2.__init__ is wrapped_once

    def test_uses_custom_io_transform_args_when_available(self):
        """If the class defines io_transform_args, it should be called instead of the global fn."""

        class _CustomTransform:
            custom_called = False

            def io_transform_args(self, init_fn, *args, **kwargs):
                _CustomTransform.custom_called = True
                return {}

            def __init__(self):
                pass

        _io_wrap_init(_CustomTransform)
        _CustomTransform()
        assert _CustomTransform.custom_called

    def test_uses_custom_io_init_when_available(self):
        """If the class defines io_init, it should be called instead of the global fn."""
        custom_cfg = MagicMock()

        class _CustomInit:
            def io_init(self, **kwargs):
                return custom_cfg

            def __init__(self):
                pass

        _io_wrap_init(_CustomInit)
        obj = _CustomInit()
        assert obj.__io__ is custom_cfg


# ---------------------------------------------------------------------------
# _io_flatten_object
# ---------------------------------------------------------------------------


class TestIoFlattenObject:
    def test_returns_flatten_result_on_success(self):
        """When dump_json succeeds, __flatten__ is called on __io__."""
        mock_io = MagicMock()
        mock_io.__flatten__ = MagicMock(return_value=("values", "metadata"))

        instance = MagicMock()
        instance.__io__ = mock_io

        with patch("nemo_deploy.llm.inference.nemo_io.serialization.dump_json", return_value=None):
            result = _io_flatten_object(instance)

        mock_io.__flatten__.assert_called_once()
        assert result == ("values", "metadata")

    def test_falls_back_to_pickle_when_unserializable(self, tmp_path):
        """When dump_json raises UnserializableValueError, object is pickled to disk."""
        from fiddle._src.experimental import serialization as _ser

        instance = MagicMock()
        instance.__io__ = MagicMock()

        import nemo_deploy.llm.inference.nemo_io as _nemo_io

        _nemo_io._thread_local.local_artifacts_dir = "artifacts"
        _nemo_io._thread_local.output_path = tmp_path

        (tmp_path / "artifacts").mkdir(parents=True, exist_ok=True)

        with patch(
            "nemo_deploy.llm.inference.nemo_io.serialization.dump_json",
            side_effect=_ser.UnserializableValueError("bad"),
        ):
            result = _io_flatten_object(instance)

        # result should be ((str_path,), None) tuple
        assert isinstance(result, tuple)
        assert result[1] is None
        assert isinstance(result[0], tuple)
        assert len(result[0]) == 1

        # cleanup
        del _nemo_io._thread_local.local_artifacts_dir
        del _nemo_io._thread_local.output_path

    def test_raises_when_pickle_path_unavailable(self):
        """When no thread-local paths are set and dump_json fails, the exception propagates."""
        from fiddle._src.experimental import serialization as _ser

        instance = MagicMock()
        instance.__io__ = MagicMock()

        import nemo_deploy.llm.inference.nemo_io as _nemo_io

        # Make sure thread-local does NOT have paths
        for attr in ("local_artifacts_dir", "output_path"):
            try:
                delattr(_nemo_io._thread_local, attr)
            except AttributeError:
                pass

        with patch(
            "nemo_deploy.llm.inference.nemo_io.serialization.dump_json",
            side_effect=_ser.UnserializableValueError("bad"),
        ):
            with pytest.raises(_ser.UnserializableValueError):
                _io_flatten_object(instance)


# ---------------------------------------------------------------------------
# _io_unflatten_object
# ---------------------------------------------------------------------------


class TestIoUnflattenObject:
    def test_calls_fdl_config_unflatten_when_no_output_dir(self):
        import nemo_deploy.llm.inference.nemo_io as _nemo_io

        try:
            delattr(_nemo_io._thread_local, "output_dir")
        except AttributeError:
            pass

        mock_result = MagicMock()
        with patch.object(fdl.Config, "__unflatten__", return_value=mock_result) as mock_unflat:
            result = _io_unflatten_object(("v1",), "meta")

        mock_unflat.assert_called_once_with(("v1",), "meta")
        assert result is mock_result

    def test_loads_pickle_when_single_value_and_output_dir_set(self, tmp_path):
        import nemo_deploy.llm.inference.nemo_io as _nemo_io

        _nemo_io._thread_local.output_dir = tmp_path

        # Write a simple pickle
        import pickle

        payload = {"key": "value"}
        pickle_file = tmp_path / "artifact.pkl"
        with open(pickle_file, "wb") as f:
            pickle.dump(payload, f)

        result = _io_unflatten_object(("artifact.pkl",), "meta")
        assert result == payload

        del _nemo_io._thread_local.output_dir

    def test_calls_fdl_unflatten_when_multiple_values(self):
        import nemo_deploy.llm.inference.nemo_io as _nemo_io

        _nemo_io._thread_local.output_dir = Path("/some/dir")

        mock_result = MagicMock()
        with patch.object(fdl.Config, "__unflatten__", return_value=mock_result) as mock_unflat:
            result = _io_unflatten_object(("v1", "v2"), "meta")

        mock_unflat.assert_called_once_with(("v1", "v2"), "meta")
        assert result is mock_result

        del _nemo_io._thread_local.output_dir


# ---------------------------------------------------------------------------
# _io_path_elements_fn
# ---------------------------------------------------------------------------


class TestIoPathElementsFn:
    def test_returns_identity_element_when_unserializable(self):
        from fiddle._src.experimental import serialization as _ser

        instance = MagicMock()
        instance.__io__ = MagicMock()

        with patch(
            "nemo_deploy.llm.inference.nemo_io.serialization.dump_json",
            side_effect=_ser.UnserializableValueError("bad"),
        ):
            result = _io_path_elements_fn(instance)

        assert len(result) == 1
        assert isinstance(result[0], _ser.IdentityElement)

    def test_returns_path_elements_on_success(self):
        mock_io = MagicMock()
        expected = ("elem1", "elem2")
        mock_io.__path_elements__ = MagicMock(return_value=expected)

        instance = MagicMock()
        instance.__io__ = mock_io

        with patch("nemo_deploy.llm.inference.nemo_io.serialization.dump_json", return_value=None):
            result = _io_path_elements_fn(instance)

        assert result == expected

    def test_returns_identity_element_when_attribute_error(self):
        from fiddle._src.experimental import serialization as _ser

        instance = MagicMock()
        instance.__io__ = MagicMock()

        with patch(
            "nemo_deploy.llm.inference.nemo_io.serialization.dump_json",
            side_effect=AttributeError("no attr"),
        ):
            result = _io_path_elements_fn(instance)

        assert isinstance(result[0], _ser.IdentityElement)


# ---------------------------------------------------------------------------
# _io_register_serialization
# ---------------------------------------------------------------------------


class TestIoRegisterSerialization:
    def test_registers_node_traverser(self):
        class _Reg:
            def __init__(self):
                pass

        with patch("nemo_deploy.llm.inference.nemo_io.serialization.register_node_traverser") as mock_reg:
            _io_register_serialization(_Reg)
            mock_reg.assert_called_once()
            args, kwargs = mock_reg.call_args
            assert args[0] is _Reg


# ---------------------------------------------------------------------------
# track_io
# ---------------------------------------------------------------------------


class TestTrackIo:
    def test_wraps_class(self):
        class _Trackable:
            def __init__(self, a=1):
                self.a = a

        result = track_io(_Trackable)
        assert result is _Trackable
        assert getattr(_Trackable, "__wrapped_init__", False)

    def test_module_type_processes_members(self):
        """track_io on a module should add IO to eligible classes."""

        class _InModule:
            def __init__(self):
                pass

        fake_module = types.ModuleType("fake_module")
        fake_module.__name__ = "fake_module"
        _InModule.__module__ = "fake_module"
        fake_module._InModule = _InModule

        result = track_io(fake_module)
        assert result is fake_module

    def test_raises_type_error_for_non_class_non_module(self):
        with pytest.raises(TypeError, match="module or a class"):
            track_io("not_a_class_or_module")

    def test_skips_builtin_types(self):
        """Built-in types like str, int should be returned unchanged."""

        class _BuiltinHolder:
            pass

        _BuiltinHolder.__init__ = str.__init__
        # Just verify built-ins are in the exclusion list by calling on str
        result = track_io(str)
        assert result is str

    def test_sets_io_artifacts(self):
        class _WithArtifacts:
            def __init__(self):
                pass

        artifacts = ["artifact1"]
        track_io(_WithArtifacts, artifacts=artifacts)
        assert _WithArtifacts.__io_artifacts__ == artifacts

    def test_skips_class_already_having_io(self):
        """Classes that already have an __io__ attribute should not be re-wrapped."""

        class _AlreadyWrapped:
            __io__ = MagicMock()

            def __init__(self):
                pass

        original_init = _AlreadyWrapped.__init__
        track_io(_AlreadyWrapped)
        # __init__ should be unchanged since the class already had __io__
        assert _AlreadyWrapped.__init__ is original_init


# ---------------------------------------------------------------------------
# drop_unexpected_params
# ---------------------------------------------------------------------------


class TestDropUnexpectedParams:
    def test_no_change_when_all_params_valid(self):
        cfg = fdl.Config(_SimpleClass, x=1, y=2)
        updated = drop_unexpected_params(cfg)
        assert not updated

    def test_removes_extra_params(self):
        cfg = fdl.Config(_SimpleClass, x=1, y=2)
        # Inject a bogus parameter directly
        cfg.__arguments__["bogus_param"] = 99
        updated = drop_unexpected_params(cfg)
        assert updated
        assert "bogus_param" not in cfg.__arguments__

    def test_no_update_when_class_accepts_kwargs(self):
        """Classes accepting **kwargs should never be considered for dropping."""

        class _AcceptsKwargs:
            def __init__(self, **kwargs):
                pass

        cfg = fdl.Config(_AcceptsKwargs)
        cfg.__arguments__["extra"] = "value"
        updated = drop_unexpected_params(cfg)
        assert not updated

    def test_returns_false_on_non_config_input(self):
        """Passing a non-fdl.Config root should short-circuit without raising."""
        updated = drop_unexpected_params(MagicMock(spec=[]))  # not an fdl.Config
        assert not updated

    def test_recurses_into_nested_configs(self):
        """Nested fdl.Config values should also be cleaned up."""

        class _Outer:
            def __init__(self, inner=None):
                self.inner = inner

        cfg_outer = fdl.Config(_Outer, inner=fdl.Config(_SimpleClass, x=1, y=2))
        cfg_outer.__arguments__["inner"].__arguments__["bogus"] = 42
        updated = drop_unexpected_params(cfg_outer)
        assert updated


# ---------------------------------------------------------------------------
# _artifact_transform_load
# ---------------------------------------------------------------------------


class TestArtifactTransformLoad:
    def test_rewrites_string_artifact_path(self):
        """String artifact values should be rewritten to absolute paths."""
        import dataclasses as _dc

        @_dc.dataclass
        class _Artifact:
            attr: str
            skip: bool = False

        class _FnCls:
            __io_artifacts__ = [_Artifact(attr="model_path")]

            def __init__(self, model_path=None):
                self.model_path = model_path

        cfg = fdl.Config(_FnCls)
        cfg.model_path = "relative/path"
        _artifact_transform_load(cfg, Path("/base"))
        assert cfg.model_path == str(Path("/base") / "relative/path")

    def test_skips_none_artifact(self):
        import dataclasses as _dc

        @_dc.dataclass
        class _Artifact:
            attr: str
            skip: bool = False

        class _FnCls:
            __io_artifacts__ = [_Artifact(attr="model_path")]

            def __init__(self, model_path=None):
                self.model_path = model_path

        cfg = fdl.Config(_FnCls)
        cfg.model_path = None
        _artifact_transform_load(cfg, Path("/base"))
        assert cfg.model_path is None

    def test_skips_artifact_with_skip_flag(self):
        import dataclasses as _dc

        @_dc.dataclass
        class _Artifact:
            attr: str
            skip: bool = True

        class _FnCls:
            __io_artifacts__ = [_Artifact(attr="model_path", skip=True)]

            def __init__(self, model_path=None):
                self.model_path = model_path

        cfg = fdl.Config(_FnCls)
        cfg.model_path = "should/not/be/changed"
        _artifact_transform_load(cfg, Path("/base"))
        # skip=True means after reading current_val the function hits `continue`
        # before rewriting, so model_path remains as-is.
        assert cfg.model_path == "should/not/be/changed"

    def test_handles_fdl_config_artifact_value(self):
        """When artifact value is itself a fdl.Config, it calls fdl.build on it."""
        import dataclasses as _dc

        @_dc.dataclass
        class _Artifact:
            attr: str
            skip: bool = False

        class _Inner:
            def __init__(self):
                self.attr = "built_value"

        class _FnCls:
            __io_artifacts__ = [_Artifact(attr="nested")]

            def __init__(self, nested=None):
                self.nested = nested

        inner_cfg = fdl.Config(_Inner)
        cfg = fdl.Config(_FnCls)
        cfg.nested = inner_cfg

        with patch("nemo_deploy.llm.inference.nemo_io.fdl.build") as mock_build:
            mock_built = MagicMock()
            mock_built.attr = "built_value"
            mock_build.return_value = mock_built
            _artifact_transform_load(cfg, Path("/base"))

        mock_build.assert_called_once_with(inner_cfg)


# ---------------------------------------------------------------------------
# load
# ---------------------------------------------------------------------------


class TestLoad:
    def _make_minimal_json(self):
        """Return a minimal io.json payload that load() will parse."""
        return {
            "objects": {},
            "root": {"key": None, "type": {"module": "builtins", "name": "dict"}, "args": [], "kwargs": {}},
        }

    def test_raises_file_not_found_for_missing_path(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load(tmp_path / "nonexistent.json")

    def test_raises_file_not_found_for_missing_io_json_in_dir(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load(tmp_path)

    def test_load_returns_config_when_build_false(self, tmp_path):
        io_json = self._make_minimal_json()
        io_path = tmp_path / "io.json"
        io_path.write_text(json.dumps(io_json))

        mock_result = MagicMock(spec=fdl.Config)
        with patch("nemo_deploy.llm.inference.nemo_io.serialization.Deserialization") as mock_deser:
            mock_deser.return_value.result = mock_result
            with patch("nemo_deploy.llm.inference.nemo_io._artifact_transform_load"):
                with patch("nemo_deploy.llm.inference.nemo_io.drop_unexpected_params"):
                    result = load(io_path, build=False)

        assert result is mock_result

    def test_load_calls_fdl_build_when_build_true(self, tmp_path):
        io_json = self._make_minimal_json()
        io_path = tmp_path / "io.json"
        io_path.write_text(json.dumps(io_json))

        mock_config = MagicMock(spec=fdl.Config)
        mock_built = MagicMock()
        with patch("nemo_deploy.llm.inference.nemo_io.serialization.Deserialization") as mock_deser:
            mock_deser.return_value.result = mock_config
            with patch("nemo_deploy.llm.inference.nemo_io._artifact_transform_load"):
                with patch("nemo_deploy.llm.inference.nemo_io.drop_unexpected_params"):
                    with patch("nemo_deploy.llm.inference.nemo_io.fdl.build", return_value=mock_built) as mock_build:
                        result = load(io_path, build=True)

        mock_build.assert_called_once_with(mock_config)
        assert result is mock_built

    def test_load_uses_dir_io_json(self, tmp_path):
        """When path is a directory, load should look for io.json inside it."""
        io_json = self._make_minimal_json()
        (tmp_path / "io.json").write_text(json.dumps(io_json))

        mock_result = MagicMock(spec=fdl.Config)
        with patch("nemo_deploy.llm.inference.nemo_io.serialization.Deserialization") as mock_deser:
            mock_deser.return_value.result = mock_result
            with patch("nemo_deploy.llm.inference.nemo_io._artifact_transform_load"):
                with patch("nemo_deploy.llm.inference.nemo_io.drop_unexpected_params"):
                    result = load(tmp_path, build=False)

        assert result is mock_result

    def test_load_with_objects_and_subpath_filtering(self, tmp_path):
        """When subpath is specified only matching objects get re-registered."""
        io_json = {
            "objects": {
                "obj1": {
                    "type": {"module": "builtins", "name": "int"},
                    "paths": ["<root>.model"],
                },
                "obj2": {
                    "type": {"module": "builtins", "name": "str"},
                    "paths": ["<root>.other"],
                },
            },
            "root": {
                "key": None,
                "type": {"module": "builtins", "name": "dict"},
                "args": [],
                "kwargs": {},
            },
        }
        io_path = tmp_path / "io.json"
        io_path.write_text(json.dumps(io_json))

        mock_result = MagicMock(spec=fdl.Config)
        with patch("nemo_deploy.llm.inference.nemo_io.serialization.Deserialization") as mock_deser:
            mock_deser.return_value.result = mock_result
            with patch("nemo_deploy.llm.inference.nemo_io._artifact_transform_load"):
                with patch("nemo_deploy.llm.inference.nemo_io.drop_unexpected_params"):
                    with patch("nemo_deploy.llm.inference.nemo_io.locate", return_value=None):
                        result = load(io_path, subpath="model", build=False)

        assert result is mock_result

    def test_load_track_io_called_when_traverser_missing(self, tmp_path):
        """When locate returns a class and no traverser is registered, track_io is called."""
        fake_cls = type("FakeCls", (), {"__init__": lambda self: None})
        io_json = {
            "objects": {
                "obj1": {
                    "type": {"module": "some.module", "name": "FakeCls"},
                    "paths": ["<root>"],
                },
            },
            "root": {"key": None, "type": {"module": "builtins", "name": "dict"}, "args": [], "kwargs": {}},
        }
        io_path = tmp_path / "io.json"
        io_path.write_text(json.dumps(io_json))

        mock_result = MagicMock(spec=fdl.Config)
        with patch("nemo_deploy.llm.inference.nemo_io.locate", return_value=fake_cls):
            with patch(
                "nemo_deploy.llm.inference.nemo_io.serialization.find_node_traverser", return_value=None
            ):
                with patch("nemo_deploy.llm.inference.nemo_io.track_io") as mock_track:
                    with patch("nemo_deploy.llm.inference.nemo_io.serialization.Deserialization") as mock_deser:
                        mock_deser.return_value.result = mock_result
                        with patch("nemo_deploy.llm.inference.nemo_io._artifact_transform_load"):
                            with patch("nemo_deploy.llm.inference.nemo_io.drop_unexpected_params"):
                                load(io_path, build=False)
                    mock_track.assert_called_once_with(fake_cls)

    def test_load_reregisters_when_traverser_already_exists(self, tmp_path):
        """When locate returns a class and a traverser exists, _io_register_serialization is called."""
        fake_cls = type("FakeCls2", (), {"__init__": lambda self: None})
        io_json = {
            "objects": {
                "obj1": {
                    "type": {"module": "some.module", "name": "FakeCls2"},
                    "paths": ["<root>"],
                },
            },
            "root": {"key": None, "type": {"module": "builtins", "name": "dict"}, "args": [], "kwargs": {}},
        }
        io_path = tmp_path / "io.json"
        io_path.write_text(json.dumps(io_json))

        mock_result = MagicMock(spec=fdl.Config)
        with patch("nemo_deploy.llm.inference.nemo_io.locate", return_value=fake_cls):
            with patch(
                "nemo_deploy.llm.inference.nemo_io.serialization.find_node_traverser",
                return_value=MagicMock(),
            ):
                with patch("nemo_deploy.llm.inference.nemo_io._io_register_serialization") as mock_rereg:
                    with patch("nemo_deploy.llm.inference.nemo_io.serialization.Deserialization") as mock_deser:
                        mock_deser.return_value.result = mock_result
                        with patch("nemo_deploy.llm.inference.nemo_io._artifact_transform_load"):
                            with patch("nemo_deploy.llm.inference.nemo_io.drop_unexpected_params"):
                                load(io_path, build=False)
                    mock_rereg.assert_called_once_with(fake_cls)


# ---------------------------------------------------------------------------
# load_context
# ---------------------------------------------------------------------------


class TestLoadContext:
    def test_load_context_delegates_to_load(self, tmp_path):
        mock_result = MagicMock()
        with patch("nemo_deploy.llm.inference.nemo_io.load", return_value=mock_result) as mock_load:
            result = load_context(tmp_path, subpath="model")
        mock_load.assert_called_once_with(tmp_path, subpath="model", build=True)
        assert result is mock_result

    def test_load_context_accepts_string_path(self, tmp_path):
        mock_result = MagicMock()
        with patch("nemo_deploy.llm.inference.nemo_io.load", return_value=mock_result):
            result = load_context(str(tmp_path))
        assert result is mock_result

    def test_load_context_falls_back_to_context_subdir(self, tmp_path):
        """When load raises FileNotFoundError, try appending 'context'."""
        call_count = {"n": 0}

        def _mock_load(path, subpath=None, build=True):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise FileNotFoundError("not found")
            return "loaded_from_context"

        with patch("nemo_deploy.llm.inference.nemo_io.load", side_effect=_mock_load):
            result = load_context(tmp_path)

        assert result == "loaded_from_context"
        assert call_count["n"] == 2

    def test_load_context_strips_context_suffix_on_retry(self, tmp_path):
        """When path ends with 'context' and load fails, retry with the parent."""
        context_path = tmp_path / "context"
        context_path.mkdir()

        call_paths = []

        def _mock_load(path, subpath=None, build=True):
            call_paths.append(path)
            if len(call_paths) == 1:
                raise FileNotFoundError("not found")
            return "loaded"

        with patch("nemo_deploy.llm.inference.nemo_io.load", side_effect=_mock_load):
            result = load_context(context_path)

        assert result == "loaded"
        # Second call should use parent (tmp_path), not tmp_path / "context" / "context"
        assert call_paths[1] == tmp_path

    def test_load_context_passes_build_flag(self, tmp_path):
        mock_result = MagicMock()
        with patch("nemo_deploy.llm.inference.nemo_io.load", return_value=mock_result) as mock_load:
            load_context(tmp_path, build=False)
        mock_load.assert_called_once_with(tmp_path, subpath=None, build=False)
