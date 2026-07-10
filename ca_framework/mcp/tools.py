# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Protocol-independent tools exposed by the MCP server."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Protocol

from ca_framework.scene import Scene, SceneStore
from ca_framework.scene.model import _constraint_from_dict, _field_from_dict, _object_from_dict


class SceneExecutor(Protocol):
    """Backend contract for scene simulation, rendering, and export."""

    def simulate(self, scene: Scene, *, frames: int | None = None) -> dict[str, Any]: ...

    def render(self, scene: Scene, *, output: Path) -> dict[str, Any]: ...

    def export(self, scene: Scene, *, output: Path, format: str) -> dict[str, Any]: ...


class SceneTools:
    """Safe scene editing operations shared by MCP and local callers."""

    def __init__(self, store: SceneStore, executor: SceneExecutor | None = None):
        self.store = store
        self.executor = executor

    def create_scene(self, name: str, overwrite: bool = False) -> dict[str, Any]:
        """Create an empty scene."""
        return self.store.create(name, overwrite=overwrite).to_dict()

    def list_scenes(self) -> list[str]:
        """List saved scene names."""
        return self.store.list()

    def get_scene(self, name: str) -> dict[str, Any]:
        """Return a complete scene description."""
        return self.store.load(name).to_dict()

    def add_object(self, scene_name: str, spec: dict[str, Any]) -> dict[str, Any]:
        """Add a rigid, cloth, or fluid object."""
        scene = self.store.load(scene_name)
        item = _object_from_dict(spec)
        _ensure_new_id(item.id, scene.objects, "object")
        scene.objects[item.id] = item
        self.store.save(scene)
        return item_to_dict(item)

    def add_constraint(self, scene_name: str, spec: dict[str, Any]) -> dict[str, Any]:
        """Add a fixed-point or distance constraint."""
        scene = self.store.load(scene_name)
        item = _constraint_from_dict(spec)
        _ensure_new_id(item.id, scene.constraints, "constraint")
        for object_id in _constraint_object_ids(item_to_dict(item)):
            _ensure_object_exists(scene, object_id)
        scene.constraints[item.id] = item
        self.store.save(scene)
        return item_to_dict(item)

    def add_field(self, scene_name: str, spec: dict[str, Any]) -> dict[str, Any]:
        """Add a uniform or radial field."""
        scene = self.store.load(scene_name)
        item = _field_from_dict(spec)
        _ensure_new_id(item.id, scene.fields, "field")
        for object_id in item.object_ids or []:
            _ensure_object_exists(scene, object_id)
        scene.fields[item.id] = item
        self.store.save(scene)
        return item_to_dict(item)

    def update_item(self, scene_name: str, collection: str, item_id: str, patch: dict[str, Any]) -> dict[str, Any]:
        """Apply a shallow patch to an existing object, constraint, or field."""
        scene = self.store.load(scene_name)
        items = _collection(scene, collection)
        if item_id not in items:
            raise KeyError(f"Unknown {collection} item: {item_id}")
        merged = item_to_dict(items[item_id]) | patch
        if merged.get("id") != item_id:
            raise ValueError("Item ids cannot be changed")
        if merged.get("kind") != item_to_dict(items[item_id]).get("kind"):
            raise ValueError("Item kinds cannot be changed")
        parser = {"objects": _object_from_dict, "constraints": _constraint_from_dict, "fields": _field_from_dict}[
            collection
        ]
        updated = parser(merged)
        if collection == "constraints":
            for object_id in _constraint_object_ids(item_to_dict(updated)):
                _ensure_object_exists(scene, object_id)
        elif collection == "fields":
            for object_id in updated.object_ids or []:
                _ensure_object_exists(scene, object_id)
        items[item_id] = updated
        self.store.save(scene)
        return item_to_dict(items[item_id])

    def remove_item(self, scene_name: str, collection: str, item_id: str) -> None:
        """Remove an item, rejecting dangling constraint and field references."""
        scene = self.store.load(scene_name)
        items = _collection(scene, collection)
        if collection == "objects":
            references = _find_object_references(scene, item_id)
            if references:
                raise ValueError(f"Object {item_id!r} is referenced by: {', '.join(references)}")
        del items[item_id]
        self.store.save(scene)

    def simulate_scene(self, scene_name: str, frames: int | None = None) -> dict[str, Any]:
        """Simulate a scene using the configured backend."""
        return self._executor().simulate(self.store.load(scene_name), frames=frames)

    def render_scene(self, scene_name: str, output: str) -> dict[str, Any]:
        """Render a scene using the configured backend."""
        return self._executor().render(self.store.load(scene_name), output=Path(output))

    def export_scene(self, scene_name: str, output: str, format: str = "json") -> dict[str, Any]:
        """Export a scene using the configured backend."""
        return self._executor().export(self.store.load(scene_name), output=Path(output), format=format)

    def _executor(self) -> SceneExecutor:
        if self.executor is None:
            raise RuntimeError("No scene executor is configured")
        return self.executor


def item_to_dict(item: object) -> dict[str, Any]:
    """Serialize a scene item without exposing dataclass internals."""
    return asdict(item)


def _collection(scene: Scene, name: str) -> dict[str, Any]:
    if name not in {"objects", "constraints", "fields"}:
        raise ValueError("collection must be 'objects', 'constraints', or 'fields'")
    return getattr(scene, name)


def _ensure_new_id(item_id: str, items: dict[str, Any], kind: str) -> None:
    if not item_id:
        raise ValueError(f"{kind.capitalize()} id cannot be empty")
    if item_id in items:
        raise ValueError(f"Duplicate {kind} id: {item_id}")


def _ensure_object_exists(scene: Scene, object_id: str) -> None:
    if object_id not in scene.objects:
        raise ValueError(f"Unknown object id: {object_id}")


def _constraint_object_ids(item: dict[str, Any]) -> list[str]:
    return [value for key, value in item.items() if key in {"object_id", "object_a", "object_b"}]


def _find_object_references(scene: Scene, object_id: str) -> list[str]:
    references = [
        f"constraint:{item.id}"
        for item in scene.constraints.values()
        if object_id in _constraint_object_ids(item_to_dict(item))
    ]
    references.extend(
        f"field:{item.id}" for item in scene.fields.values() if item.object_ids and object_id in item.object_ids
    )
    return references
