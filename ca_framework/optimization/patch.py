# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Immutable path-based scene parameter patches."""

from __future__ import annotations

from copy import deepcopy
import json
from typing import Any, Mapping

from ca_framework.scene.model import Scene

from .model import ParameterSpec


def apply_parameter_patch(
    base_scene: Scene,
    values: Mapping[str, Any],
    specs: Mapping[str, ParameterSpec] | None = None,
) -> Scene:
    """Return a deep-copied scene with validated path updates applied.

    Args:
        base_scene: Scene used as the immutable trial baseline.
        values: Mapping from serialized scene paths to candidate values.
        specs: Optional specs keyed by path. When supplied, every patch path
            must have an enabled spec and its value is range/type checked.

    Returns:
        A new scene with no mutable state shared with :paramref:`base_scene`.
    """
    candidate = json.loads(json.dumps(base_scene.to_dict()))
    for path, value in values.items():
        if specs is not None:
            try:
                spec = specs[path]
            except KeyError as error:
                raise ValueError(f"No parameter spec registered for path {path!r}") from error
            if not spec.enabled:
                raise ValueError(f"Parameter {spec.name!r} is disabled")
            spec.validate_value(value)
        _set_path(candidate, path, deepcopy(value))
    return Scene.from_dict(candidate)


def _set_path(root: dict[str, Any], path: str, value: Any) -> None:
    parts = path.split(".")
    if not path or any(not part for part in parts):
        raise ValueError(f"Invalid parameter path: {path!r}")
    current: Any = root
    for part in parts[:-1]:
        current = _path_child(current, part, path)
    leaf = parts[-1]
    if isinstance(current, dict):
        if leaf not in current:
            raise ValueError(f"Unknown parameter path: {path!r}")
        existing = current[leaf]
    elif isinstance(current, list):
        index = _list_index(leaf, path)
        if index >= len(current):
            raise ValueError(f"Unknown parameter path: {path!r}")
        existing = current[index]
    else:
        raise ValueError(f"Unknown parameter path: {path!r}")
    if type(existing) is bool:
        valid_type = type(value) is bool
    elif type(existing) is int:
        valid_type = type(value) is int
    elif type(existing) is float:
        valid_type = type(value) in {int, float} and type(value) is not bool
    else:
        valid_type = isinstance(value, type(existing))
    if not valid_type:
        raise ValueError(
            f"Parameter path {path!r} expects {type(existing).__name__}, got {type(value).__name__}"
        )
    if isinstance(current, dict):
        current[leaf] = value
    else:
        current[_list_index(leaf, path)] = value


def _path_child(current: Any, part: str, path: str) -> Any:
    if isinstance(current, dict):
        if part not in current:
            raise ValueError(f"Unknown parameter path: {path!r}")
        return current[part]
    if isinstance(current, list):
        index = _list_index(part, path)
        if index >= len(current):
            raise ValueError(f"Unknown parameter path: {path!r}")
        return current[index]
    raise ValueError(f"Unknown parameter path: {path!r}")


def _list_index(part: str, path: str) -> int:
    if not part.isdecimal():
        raise ValueError(f"List path component must be a zero-based index in {path!r}")
    return int(part)
