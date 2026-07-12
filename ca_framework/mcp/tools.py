# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Protocol-independent tools exposed by the MCP server."""

from __future__ import annotations

import json
import multiprocessing
import os
import signal
from concurrent.futures import Future, ThreadPoolExecutor
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from threading import Event, Lock
from typing import Any, Protocol
from uuid import uuid4

from ca_framework.scene import Scene, SceneStore
from ca_framework.scene.model import (
    Camera,
    _action_from_dict,
    _constraint_from_dict,
    _field_from_dict,
    _object_from_dict,
)
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

    def resume_render(self, scene: Scene, *, output_dir: Path) -> dict[str, Any]: ...


def _preview_worker(executor: SceneExecutor, scene: Scene, frames: int | None, connection: Any) -> None:
    """Run preview in a process group that the parent can terminate atomically."""
    os.setsid()
    try:
        connection.send((True, executor.preview(scene, frames=frames)))
    except BaseException as error:
        connection.send((False, f"{type(error).__name__}: {error}"))
    finally:
        connection.close()


def _terminate_preview_process(process: multiprocessing.Process, sig: signal.Signals) -> None:
    """Signal a preview and its group when isolated, otherwise signal the child itself."""
    try:
        process_group = os.getpgid(process.pid)
        if process_group == process.pid:
            os.killpg(process_group, sig)
        else:
            os.kill(process.pid, sig)
    except ProcessLookupError:
        pass


class SceneTools:
    """Safe scene editing operations shared by MCP and local callers."""

    def __init__(self, store: SceneStore, executor: SceneExecutor | None = None):
        self.store = store
        self.executor = executor
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ca-scene")
        self._jobs: dict[str, tuple[Future[dict[str, Any]], str, Event, dict[str, Any]]] = {}
        self._jobs_lock = Lock()
        self._jobs_dir = self.store.root / ".jobs"
        self._jobs_dir.mkdir(exist_ok=True)
        self._restore_jobs()

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
            "schema_version": 3,
            "objects": ["rigid", "cloth", "fluid", "container"],
            "rigid_shapes": ["box", "sphere", "capsule", "compound"],
            "fluid_phases": ["smoke", "liquid"],
            "solvers": {
                "rigid": ["xpbd", "vbd"],
                "cloth": {"recommended": "vbd", "supported": ["xpbd", "vbd"]},
                "smoke": "smoke",
                "liquid": "apic",
            },
            "actions": ["transform", "impulse", "force", "emit"],
            "outputs": ["mp4", "scene-json", "python", "metrics", "diagnostics"],
            "camera": {"modes": ["auto", "look-at"], "supports_up": True},
            "recommended_simulation": {
                "rigid": {"fps": [30, 60], "substeps": [4, 12]},
                "cloth": {"fps": [30, 30], "substeps": [8, 12]},
                "smoke": {"fps": [20, 30], "substeps": [1, 4]},
                "liquid": {"fps": [20, 25], "substeps": [2, 8]},
            },
            "preview_limits": {
                "duration": 1.0,
                "fps": 30,
                "substeps": 12,
                "rigid": {"iterations": 10},
                "cloth": {"iterations": 10},
                "resolution": [640, 360],
                "liquid_capacity": 50_000,
                "cloth_axis": 24,
            },
        }

    def set_camera(
        self,
        scene_name: str,
        *,
        position: tuple[float, float, float],
        target: tuple[float, float, float],
        up: tuple[float, float, float] = (0.0, 0.0, 1.0),
        field_of_view: float = 45.0,
    ) -> dict[str, Any]:
        """Set a fixed look-at camera for a scene."""
        camera = Camera(
            position=position,
            target=target,
            up=up,
            field_of_view=field_of_view,
            auto_frame=False,
        )
        scene = self.store.load(scene_name)
        scene.render.camera = camera
        report = validate_scene(scene)
        if not report["valid"]:
            camera_errors = [item for item in report["diagnostics"] if item["path"].startswith("render.camera")]
            if camera_errors:
                raise ValueError(f"Invalid camera: {camera_errors}")
        self.store.save(scene)
        return item_to_dict(camera)

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
        """Apply a validated recursive patch to one typed collection."""
        scene = self.store.load(scene_name)
        items = _collection(scene, collection)
        if item_id not in items:
            raise KeyError(f"Unknown {collection} item: {item_id}")
        original = item_to_dict(items[item_id])
        merged = _merge_patch(deepcopy(original), patch)
        if merged.get("id") != item_id:
            raise ValueError("Item ids cannot be changed")
        if merged.get("kind") != original.get("kind"):
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
        timeout = float(os.environ.get("CA_SCENE_PREVIEW_TIMEOUT", "90"))
        parent, child = multiprocessing.Pipe(duplex=False)
        # Spawn is required here: forking a process after Warp or FastMCP has
        # initialized threads can deadlock before the child reaches its target.
        process = multiprocessing.get_context("spawn").Process(
            target=_preview_worker,
            args=(self._executor(), self.store.load(scene_name), frames, child),
        )
        process.start()
        child.close()
        process.join(timeout)
        if process.is_alive():
            _terminate_preview_process(process, signal.SIGTERM)
            process.join(5.0)
            if process.is_alive():
                _terminate_preview_process(process, signal.SIGKILL)
                process.join(5.0)
            return {
                "status": "preview_timeout",
                "preview": True,
                "elapsed_seconds": timeout,
                "diagnostics": [{"code": "preview_timeout", "message": f"Preview exceeded {timeout:g} seconds."}],
                "recommended_actions": ["Reduce one simulation scale dimension before retrying preview."],
            }
        if not parent.poll():
            return {"status": "preview_failed", "preview": True, "diagnostics": [{"code": "preview_worker_exit"}]}
        succeeded, value = parent.recv()
        if not succeeded:
            raise RuntimeError(value)
        return value

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
        result = {"job_id": job_id, **progress, "output_dir": str(Path(output_dir).resolve())}
        self._write_job_record(job_id, result)
        return result

    def get_job(self, job_id: str) -> dict[str, Any]:
        """Return asynchronous job state or result."""
        with self._jobs_lock:
            future, scene_name, _cancel_event, progress = self._jobs[job_id]
        if future.cancelled():
            result = {"job_id": job_id, "status": "cancelled", "scene": scene_name}
            self._write_job_record(job_id, result)
            return result
        if not future.done():
            result = {"job_id": job_id, **progress, "scene": scene_name}
            self._write_job_record(job_id, result)
            return result
        try:
            result = {"job_id": job_id, **future.result()}
            self._write_job_record(job_id, result)
            return result
        except Exception as error:
            result = {
                "job_id": job_id,
                "status": "failed",
                "scene": scene_name,
                "diagnostics": [{"code": "job_failed", "message": str(error)}],
            }
            self._write_job_record(job_id, result)
            return result

    def cancel_job(self, job_id: str) -> dict[str, Any]:
        """Cancel a queued job; running native work completes safely."""
        with self._jobs_lock:
            future, scene_name, cancel_event, progress = self._jobs[job_id]
        cancel_event.set()
        cancelled = future.cancel()
        progress["status"] = "cancelled" if cancelled else "cancelling"
        result = {
            "job_id": job_id,
            "scene": scene_name,
            "status": progress["status"],
            "cancelled": True,
        }
        self._write_job_record(job_id, result)
        return result

    def _write_job_record(self, job_id: str, record: dict[str, Any]) -> None:
        path = self._jobs_dir / f"{job_id}.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)

    def _restore_jobs(self) -> None:
        """Restore durable job state and resume render-only work when safe."""
        for path in self._jobs_dir.glob("*.json"):
            record = json.loads(path.read_text(encoding="utf-8"))
            job_id = record["job_id"]
            scene_name = record["scene"]
            status = record.get("status", "interrupted")
            output_dir = Path(record.get("output_dir", ""))
            cache_manifest = output_dir / "cache" / "manifest.json"
            cancel_event = Event()
            progress = dict(record)
            if status in {"queued", "running", "cancelling"} and cache_manifest.is_file() and self.executor:
                try:
                    complete = json.loads(cache_manifest.read_text(encoding="utf-8")).get("complete", False)
                except (OSError, json.JSONDecodeError):
                    complete = False
                if complete:
                    progress.update(status="running", stage="render")
                    future = self._pool.submit(
                        self.executor.resume_render,
                        self.store.load(scene_name),
                        output_dir=output_dir,
                    )
                    self._jobs[job_id] = (future, scene_name, cancel_event, progress)
                    continue
            if status in {"queued", "running", "cancelling"}:
                progress.update(status="interrupted", stage="interrupted")
                self._write_job_record(job_id, progress)
            future = Future()
            future.set_result(progress)
            self._jobs[job_id] = (future, scene_name, cancel_event, progress)

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
