# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Protocol-independent tools exposed by the MCP server."""

from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from threading import Event, Lock
from typing import Any, Protocol
from uuid import uuid4

from ca_framework.scene import Scene, SceneStore
from ca_framework.scene.model import _action_from_dict, _constraint_from_dict, _field_from_dict, _object_from_dict
from ca_framework.scene.validation import validate_scene


class SceneExecutor(Protocol):
    """Backend contract for scene simulation, rendering, and export."""

    def simulate(self, scene: Scene, *, frames: int | None = None) -> dict[str, Any]: ...

    def render(self, scene: Scene, *, output: Path) -> dict[str, Any]: ...

    def export(self, scene: Scene, *, output: Path, format: str) -> dict[str, Any]: ...

    def preview(self, scene: Scene, *, frames: int | None = None) -> dict[str, Any]: ...

    def run(
        self,
        scene: Scene,
        *,
        output_dir: Path,
        cancel_event: Event | None = None,
        progress: dict[str, Any] | None = None,
    ) -> dict[str, Any]: ...


class SceneTools:
    """Safe scene editing operations shared by MCP and local callers."""

    def __init__(self, store: SceneStore, executor: SceneExecutor | None = None):
        self.store = store
        self.executor = executor
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ca-scene")
        self._jobs: dict[str, tuple[Future[dict[str, Any]], str, Event, dict[str, Any]]] = {}
        self._jobs_lock = Lock()

    def create_scene(self, name: str, overwrite: bool = False) -> dict[str, Any]:
        """Create an empty scene."""
        return self.store.create(name, overwrite=overwrite).to_dict()

    def list_scenes(self) -> list[str]:
        """List saved scene names."""
        return self.store.list()

    def get_scene(self, name: str) -> dict[str, Any]:
        """Return a complete scene description."""
        return self.store.load(name).to_dict()

    def get_capabilities(self) -> dict[str, Any]:
        """Return supported IR features and solver routes."""
        return {
            "schema_version": 2,
            "objects": ["rigid", "cloth", "fluid", "container"],
            "rigid_shapes": ["box", "sphere", "capsule", "compound"],
            "fluid_phases": ["smoke", "liquid"],
            "solvers": {"rigid_and_cloth": "xpbd", "smoke": "smoke", "liquid": "apic"},
            "actions": ["transform", "impulse", "force", "emit"],
            "outputs": ["mp4", "scene-json", "python", "metrics", "diagnostics"],
        }

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

    def add_action(self, scene_name: str, spec: dict[str, Any]) -> dict[str, Any]:
        """Add a timeline action."""
        scene = self.store.load(scene_name)
        item = _action_from_dict(spec)
        _ensure_new_id(item.id, scene.actions, "action")
        _ensure_object_exists(scene, item.object_id)
        scene.actions[item.id] = item
        self.store.save(scene)
        return item_to_dict(item)

    def apply_scene_patch(self, scene_name: str, patch: dict[str, Any]) -> dict[str, Any]:
        """Apply a recursive merge transaction, saving only a valid parse."""
        original = self.store.load(scene_name).to_dict()
        merged = _merge_patch(deepcopy(original), patch)
        merged["name"] = original["name"]
        candidate = Scene.from_dict(merged)
        report = validate_scene(candidate)
        if not report["valid"]:
            raise ValueError(f"Scene patch failed validation: {report['diagnostics']}")
        self.store.save(candidate)
        return candidate.to_dict()

    def update_object(self, scene_name: str, item_id: str, patch: dict[str, Any]) -> dict[str, Any]:
        """Update a rigid, cloth, fluid, or container object."""
        return self._update_collection_entry(scene_name, "objects", item_id, patch)

    def update_constraint(self, scene_name: str, item_id: str, patch: dict[str, Any]) -> dict[str, Any]:
        """Update a fixed-point or distance constraint."""
        return self._update_collection_entry(scene_name, "constraints", item_id, patch)

    def update_field(self, scene_name: str, item_id: str, patch: dict[str, Any]) -> dict[str, Any]:
        """Update a uniform or radial field."""
        return self._update_collection_entry(scene_name, "fields", item_id, patch)

    def update_action(self, scene_name: str, item_id: str, patch: dict[str, Any]) -> dict[str, Any]:
        """Update a transform, impulse, force, or emission action."""
        return self._update_collection_entry(scene_name, "actions", item_id, patch)

    def _update_collection_entry(
        self, scene_name: str, collection: str, item_id: str, patch: dict[str, Any]
    ) -> dict[str, Any]:
        """Apply a validated shallow patch to one typed collection."""
        scene = self.store.load(scene_name)
        items = _collection(scene, collection)
        if item_id not in items:
            raise KeyError(f"Unknown {collection} item: {item_id}")
        merged = item_to_dict(items[item_id]) | patch
        if merged.get("id") != item_id:
            raise ValueError("Item ids cannot be changed")
        if merged.get("kind") != item_to_dict(items[item_id]).get("kind"):
            raise ValueError("Item kinds cannot be changed")
        parser = {
            "objects": _object_from_dict,
            "constraints": _constraint_from_dict,
            "fields": _field_from_dict,
            "actions": _action_from_dict,
        }[collection]
        updated = parser(merged)
        if collection == "constraints":
            for object_id in _constraint_object_ids(item_to_dict(updated)):
                _ensure_object_exists(scene, object_id)
        elif collection == "fields":
            for object_id in updated.object_ids or []:
                _ensure_object_exists(scene, object_id)
        elif collection == "actions":
            _ensure_object_exists(scene, updated.object_id)
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

    def validate_scene(self, scene_name: str) -> dict[str, Any]:
        """Return structured validation and resource estimates."""
        return validate_scene(self.store.load(scene_name))

    def preview_scene(self, scene_name: str, frames: int | None = None) -> dict[str, Any]:
        """Run a short preview and return keyframes and metrics."""
        return self._executor().preview(self.store.load(scene_name), frames=frames)

    def run_scene(self, scene_name: str, output_dir: str) -> dict[str, Any]:
        """Asynchronously create a durable simulation and rendering bundle."""
        scene = self.store.load(scene_name)
        job_id = uuid4().hex
        cancel_event = Event()
        progress = {"status": "queued", "stage": "queued", "frame": 0, "scene": scene_name}
        future = self._pool.submit(
            self._executor().run,
            scene,
            output_dir=Path(output_dir),
            cancel_event=cancel_event,
            progress=progress,
        )
        with self._jobs_lock:
            self._jobs[job_id] = (future, scene_name, cancel_event, progress)
        return {"job_id": job_id, **progress, "output_dir": str(Path(output_dir).resolve())}

    def get_job(self, job_id: str) -> dict[str, Any]:
        """Return asynchronous job state or result."""
        with self._jobs_lock:
            future, scene_name, _cancel_event, progress = self._jobs[job_id]
        if future.cancelled():
            return {"job_id": job_id, "status": "cancelled", "scene": scene_name}
        if not future.done():
            return {"job_id": job_id, **progress, "scene": scene_name}
        try:
            return {"job_id": job_id, **future.result()}
        except Exception as error:
            return {
                "job_id": job_id,
                "status": "failed",
                "scene": scene_name,
                "diagnostics": [{"code": "job_failed", "message": str(error)}],
            }

    def cancel_job(self, job_id: str) -> dict[str, Any]:
        """Cancel a queued job; running native work completes safely."""
        with self._jobs_lock:
            future, scene_name, cancel_event, progress = self._jobs[job_id]
        cancel_event.set()
        cancelled = future.cancel()
        progress["status"] = "cancelled" if cancelled else "cancelling"
        return {
            "job_id": job_id,
            "scene": scene_name,
            "status": progress["status"],
            "cancelled": True,
        }

    def export_program(self, scene_name: str, program_output: str, scene_output: str) -> dict[str, Any]:
        """Export a reproducible Python program and its scene JSON."""
        scene = self.store.load(scene_name)
        program = self._executor().export(scene, output=Path(program_output), format="python")
        scene_json = self._executor().export(scene, output=Path(scene_output), format="json")
        return {
            "status": "completed",
            "scene": scene_name,
            "program": program["output"],
            "scene_json": scene_json["output"],
        }

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
    if name not in {"objects", "constraints", "fields", "actions"}:
        raise ValueError("collection must be 'objects', 'constraints', 'fields', or 'actions'")
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
    references.extend(f"action:{item.id}" for item in scene.actions.values() if item.object_id == object_id)
    return references


def _merge_patch(target: Any, patch: Any) -> Any:
    if not isinstance(target, dict) or not isinstance(patch, dict):
        return deepcopy(patch)
    for key, value in patch.items():
        if value is None:
            target.pop(key, None)
        else:
            target[key] = _merge_patch(target.get(key), value)
    return target
