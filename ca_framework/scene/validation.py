# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Static scene validation and resource estimation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from itertools import pairwise
from math import isfinite, prod, sqrt
from typing import Any, Literal

import numpy as np

from .model import (
    ActionEmit,
    ActionForce,
    ActionImpulse,
    ActionTransform,
    ConstraintFixedPoint,
    ObjectCloth,
    ObjectContainer,
    ObjectFluid,
    ObjectRigid,
    Scene,
    VertexSelector,
)


@dataclass(slots=True)
class SceneDiagnostic:
    """One machine-readable validation finding."""

    path: str
    code: str
    message: str
    suggestion: str
    severity: Literal["error", "warning"] = "error"


def estimate_resources(scene: Scene) -> dict[str, int]:
    """Estimate particles, grid cells, and rendered frames."""
    particles = 0
    initial_particles = 0
    emitted_particles = 0
    grid_cells = 0
    for item in scene.objects.values():
        if isinstance(item, ObjectCloth):
            count = prod(item.resolution)
            particles += count
            initial_particles += count
        elif isinstance(item, ObjectFluid):
            grid_cells += prod(item.grid_resolution)
            if item.phase == "liquid" and item.particle_spacing > 0.0:
                count = prod(max(1, int(size / item.particle_spacing)) for size in item.size)
                particles += count
                initial_particles += count
                for emitter in item.emitters:
                    active_substeps = max(
                        0,
                        round(
                            (min(emitter.end_time, scene.settings.duration) - max(0.0, emitter.start_time))
                            * scene.settings.fps
                            * scene.settings.substeps
                        ),
                    )
                    emitted = active_substeps * prod(max(1, int(size / item.particle_spacing)) for size in emitter.size)
                    particles += emitted
                    emitted_particles += emitted
    # MAC smoke/APIC grids store several scalar grids plus three face grids.
    # This deliberately overestimates rather than allowing jobs to OOM late.
    has_liquid = any(isinstance(item, ObjectFluid) and item.phase == "liquid" for item in scene.objects.values())
    particle_capacity = scene.settings.max_particles if has_liquid else particles
    estimated_bytes = particle_capacity * 64 + grid_cells * 64
    return {
        "particles": particles,
        "estimated_initial_particles": initial_particles,
        "estimated_emitted_particles": emitted_particles,
        "particle_capacity": scene.settings.max_particles if has_liquid else particles,
        "grid_cells": grid_cells,
        "simulation_frames": round(scene.settings.duration * scene.settings.fps),
        "render_frames": round(scene.settings.duration * scene.render.fps),
        "estimated_memory_bytes": estimated_bytes,
        "estimated_contact_pairs": sum(
            1
            for index, item_a in enumerate(scene.objects.values())
            for item_b in list(scene.objects.values())[index + 1 :]
            if isinstance(item_a, (ObjectRigid, ObjectContainer)) and isinstance(item_b, (ObjectRigid, ObjectContainer))
        ),
    }


def validate_scene(scene: Scene) -> dict[str, Any]:
    """Validate references, dimensions, solver routing, penetration, and budget."""
    findings: list[SceneDiagnostic] = []

    def error(
        path: str, code: str, message: str, suggestion: str, severity: Literal["error", "warning"] = "error"
    ) -> None:
        findings.append(SceneDiagnostic(path, code, message, suggestion, severity))

    if scene.settings.fps <= 0:
        error("settings.fps", "non_positive", "Simulation FPS must be positive.", "Use an FPS such as 60.")
    if scene.settings.duration <= 0.0:
        error(
            "settings.duration",
            "non_positive",
            "Duration must be positive.",
            "Use a duration in seconds greater than zero.",
        )
    if scene.render.fps <= 0 or any(value <= 0 for value in scene.render.resolution):
        error(
            "render",
            "invalid_render_settings",
            "Resolution and render FPS must be positive.",
            "Use a positive width, height, and FPS.",
        )

    fluid_phases: set[str] = set()
    for key, item in scene.objects.items():
        path = f"objects.{key}"
        if item.id != key:
            error(f"{path}.id", "id_mismatch", "Object key and id differ.", f"Set id to {key!r}.")
        if any(scale <= 0.0 for scale in item.transform.scale):
            error(
                f"{path}.transform.scale",
                "invalid_dimension",
                "Scale components must be positive.",
                "Use positive dimensionless scale values.",
            )
        quaternion_norm = sqrt(sum(component * component for component in item.transform.rotation))
        if not all(isfinite(component) for component in item.transform.rotation) or abs(quaternion_norm - 1.0) > 1.0e-3:
            error(
                f"{path}.transform.rotation",
                "invalid_quaternion",
                "Rotation must be a finite unit quaternion in (x, y, z, w) order.",
                "Normalize the quaternion before submitting the scene.",
            )
        if item.physical_material.density <= 0.0:
            error(
                f"{path}.physical_material.density",
                "invalid_density",
                "Density must be positive.",
                "Use density in kg/m^3 greater than zero.",
            )
        if isinstance(item, ObjectRigid):
            if any(size <= 0.0 for size in item.size):
                error(
                    f"{path}.size",
                    "invalid_dimension",
                    "Rigid dimensions must be positive.",
                    "Use dimensions in metres greater than zero.",
                )
            if item.shape == "compound" and not item.shapes:
                error(
                    f"{path}.shapes",
                    "empty_compound",
                    "A compound rigid body needs at least one shape.",
                    "Add box, sphere, or capsule shapes.",
                )
        elif isinstance(item, ObjectContainer):
            if any(size <= 0.0 for size in item.inner_size) or item.wall_thickness <= 0.0:
                error(
                    path,
                    "invalid_dimension",
                    "Container dimensions and wall thickness must be positive.",
                    "Use dimensions in metres greater than zero.",
                )
        elif isinstance(item, ObjectCloth):
            if any(size <= 0.0 for size in item.size) or any(value < 2 for value in item.resolution):
                error(
                    path,
                    "invalid_cloth",
                    "Cloth dimensions must be positive and resolution at least 2 by 2.",
                    "Increase cloth size or resolution.",
                )
            vertex_count = item.resolution[0] * item.resolution[1]
            for selector_index, selector in enumerate(item.pinned):
                _validate_selector(selector, vertex_count, f"{path}.pinned.{selector_index}", error)
        elif isinstance(item, ObjectFluid):
            fluid_phases.add(item.phase)
            if any(size <= 0.0 for size in item.size) or any(value < 2 for value in item.grid_resolution):
                error(
                    path,
                    "invalid_fluid_grid",
                    "Fluid size must be positive and grid axes at least 2 cells.",
                    "Increase fluid size or resolution.",
                )
            if item.phase == "liquid" and item.particle_spacing <= 0.0:
                error(
                    f"{path}.particle_spacing",
                    "invalid_spacing",
                    "Liquid particle spacing must be positive.",
                    "Use spacing in metres greater than zero.",
                )
            for emitter_index, emitter in enumerate(item.emitters):
                emitter_path = f"{path}.emitters.{emitter_index}"
                if emitter.start_time < 0.0 or emitter.end_time < emitter.start_time:
                    error(
                        emitter_path,
                        "invalid_action_time",
                        "Emitter times must be ordered and non-negative.",
                        "Use 0 <= start_time <= end_time.",
                    )
                if any(value <= 0.0 for value in emitter.size) or emitter.density < 0.0:
                    error(
                        emitter_path,
                        "invalid_emitter",
                        "Emitter dimensions must be positive and density non-negative.",
                        "Use a positive emitter size and non-negative density.",
                    )
                domain_min = tuple(item.transform.position[axis] - item.size[axis] * 0.5 for axis in range(3))
                domain_max = tuple(item.transform.position[axis] + item.size[axis] * 0.5 for axis in range(3))
                emitter_min = tuple(emitter.position[axis] - emitter.size[axis] * 0.5 for axis in range(3))
                emitter_max = tuple(emitter.position[axis] + emitter.size[axis] * 0.5 for axis in range(3))
                if item.phase == "smoke" and any(
                    emitter_min[axis] < domain_min[axis] or emitter_max[axis] > domain_max[axis] for axis in range(3)
                ):
                    error(
                        emitter_path,
                        "emitter_outside_domain",
                        "Emitter extends outside the fluid domain.",
                        "Move or resize the emitter so it fits inside the fluid volume.",
                    )

    for collection_name in ("constraints", "fields", "actions"):
        for key, item in getattr(scene, collection_name).items():
            mapping = asdict(item)
            references = [value for name, value in mapping.items() if name in {"object_id", "object_a", "object_b"}]
            references += mapping.get("object_ids") or []
            for object_id in references:
                if object_id not in scene.objects:
                    error(
                        f"{collection_name}.{key}",
                        "dangling_reference",
                        f"Unknown object id: {object_id}",
                        "Reference an existing scene object.",
                    )

    for key, action in scene.actions.items():
        path = f"actions.{key}"
        if isinstance(action, ActionTransform):
            times = [keyframe.time for keyframe in action.keyframes]
            if not times or any(time < 0.0 for time in times) or any(a >= b for a, b in pairwise(times)):
                error(
                    f"{path}.keyframes",
                    "invalid_action_time",
                    "Transform keyframes must have strictly increasing non-negative times.",
                    "Sort keyframes and remove duplicate timestamps.",
                )
            for index, keyframe in enumerate(action.keyframes):
                norm = sqrt(sum(value * value for value in keyframe.transform.rotation))
                if not all(isfinite(value) for value in keyframe.transform.rotation) or abs(norm - 1.0) > 1.0e-3:
                    error(
                        f"{path}.keyframes.{index}.transform.rotation",
                        "invalid_quaternion",
                        "Keyframe rotation must be a finite unit quaternion.",
                        "Normalize the quaternion.",
                    )
        elif isinstance(action, ActionImpulse):
            if action.time < 0.0 or action.time > scene.settings.duration:
                error(path, "invalid_action_time", "Impulse time is outside the scene.", "Move it inside the duration.")
        elif isinstance(action, (ActionForce, ActionEmit)):
            if action.start_time < 0.0 or action.end_time < action.start_time:
                error(
                    path,
                    "invalid_action_time",
                    "Action times must be ordered and non-negative.",
                    "Use 0 <= start <= end.",
                )
        if isinstance(action, ActionEmit):
            fluid = scene.objects.get(action.object_id)
            if isinstance(fluid, ObjectFluid) and not 0 <= action.emitter_index < len(fluid.emitters):
                error(path, "invalid_emitter", "Emitter index is out of range.", "Reference an existing emitter.")

    for key, constraint in scene.constraints.items():
        if isinstance(constraint, ConstraintFixedPoint) and constraint.selector is not None:
            target = scene.objects.get(constraint.object_id)
            if not isinstance(target, ObjectCloth):
                error(
                    f"constraints.{key}.selector",
                    "invalid_selector_target",
                    "Vertex selectors can only target cloth objects.",
                    "Remove the selector or reference a cloth object.",
                )
            else:
                _validate_selector(
                    constraint.selector,
                    target.resolution[0] * target.resolution[1],
                    f"constraints.{key}.selector",
                    error,
                )

    # Fluids entirely larger than an enclosing container cannot be initialized
    # without wall penetration. Containers are open at the top, so only X/Y
    # clearance and the floor-relative height are checked.
    for fluid_key, fluid in ((key, value) for key, value in scene.objects.items() if isinstance(value, ObjectFluid)):
        for container_key, container in (
            (key, value) for key, value in scene.objects.items() if isinstance(value, ObjectContainer)
        ):
            if all(
                abs(fluid.transform.position[axis] - container.transform.position[axis])
                + fluid.size[axis] * fluid.transform.scale[axis] * 0.5
                <= container.inner_size[axis] * container.transform.scale[axis] * 0.5
                for axis in (0, 1)
            ):
                fluid_bottom = fluid.transform.position[2] - fluid.size[2] * fluid.transform.scale[2] * 0.5
                container_floor = container.transform.position[2]
                if fluid_bottom < container_floor - 1.0e-6:
                    error(
                        f"objects.{fluid_key}",
                        "container_clearance",
                        f"Fluid starts below the floor of container {container_key!r}.",
                        "Raise or resize the initial fluid volume.",
                    )

    expected = {"smoke": "smoke", "liquid": "apic"}
    if scene.settings.solver != "auto":
        for phase in fluid_phases:
            if scene.settings.solver != expected[phase]:
                error(
                    "settings.solver",
                    "solver_mismatch",
                    f"{phase} requires the {expected[phase]} solver.",
                    "Use solver='auto' or the matching fluid solver.",
                )
    if len(fluid_phases) > 1 and scene.settings.solver != "auto":
        error(
            "settings.solver",
            "mixed_fluid_solver",
            "Mixed smoke and liquid scenes require automatic routing.",
            "Use solver='auto'.",
        )

    resources = estimate_resources(scene)
    if resources["particles"] > scene.settings.max_particles:
        error(
            "settings.max_particles",
            "particle_budget",
            f"Estimated {resources['particles']} particles exceeds the {scene.settings.max_particles} budget.",
            "Increase particle spacing, lower cloth resolution, or explicitly raise the budget.",
        )

    # Conservative AABB overlap catches obvious initial rigid/container penetrations.
    solids = [(key, item) for key, item in scene.objects.items() if isinstance(item, (ObjectRigid, ObjectContainer))]
    for index, (key_a, item_a) in enumerate(solids):
        for key_b, item_b in solids[index + 1 :]:
            if item_a.motion == "dynamic" or item_b.motion == "dynamic":
                extent_a = item_a.size if isinstance(item_a, ObjectRigid) else item_a.inner_size
                extent_b = item_b.size if isinstance(item_b, ObjectRigid) else item_b.inner_size
                overlap = all(
                    abs(item_a.transform.position[i] - item_b.transform.position[i]) < (extent_a[i] + extent_b[i]) * 0.5
                    for i in range(3)
                )
                if overlap:
                    error(
                        f"objects.{key_a}",
                        "initial_overlap",
                        f"Initial bounds overlap object {key_b!r}.",
                        "Separate the initial transforms or use an enclosing container intentionally.",
                        "warning",
                    )

    camera = scene.render.camera
    if not 0.0 < camera.field_of_view < 180.0:
        error(
            "render.camera.field_of_view",
            "invalid_camera_fov",
            "Camera field of view must be between 0 and 180 degrees.",
            "Choose a field of view such as 45 degrees.",
        )
    if (camera.position is None) != (camera.target is None):
        error(
            "render.camera",
            "incomplete_camera",
            "Camera position and target must be supplied together.",
            "Set both values or enable automatic framing.",
        )
    if camera.position is not None and camera.target is not None:
        view = np.asarray(camera.target, dtype=float) - np.asarray(camera.position, dtype=float)
        up = np.asarray(camera.up, dtype=float)
        if np.linalg.norm(view) < 1.0e-8:
            error(
                "render.camera.target",
                "degenerate_camera",
                "Camera position and target must differ.",
                "Move the camera away from its target.",
            )
        if np.linalg.norm(up) < 1.0e-8:
            error(
                "render.camera.up",
                "degenerate_camera_up",
                "Camera up vector must be non-zero.",
                "Use an axis such as [0, 0, 1].",
            )
        elif np.linalg.norm(view) >= 1.0e-8 and np.linalg.norm(np.cross(view, up)) < 1.0e-8:
            error(
                "render.camera.up",
                "parallel_camera_up",
                "Camera up vector must not be parallel to the view direction.",
                "Choose a perpendicular up vector.",
            )

    return {
        "valid": not any(item.severity == "error" for item in findings),
        "diagnostics": [asdict(item) for item in findings],
        "resources": resources,
        "pipeline": select_pipeline(scene),
    }


def select_pipeline(scene: Scene) -> list[str]:
    """Return the deterministic solver schedule required by a scene."""
    pipeline = []
    cloth = [item for item in scene.objects.values() if isinstance(item, ObjectCloth)]
    if any(not isinstance(item, ObjectFluid) for item in scene.objects.values()):
        rigid_solver = "vbd" if any(item.self_collision for item in cloth) else "xpbd"
        pipeline.append(rigid_solver if scene.settings.solver in {"auto", "smoke", "apic"} else scene.settings.solver)
    phases = {item.phase for item in scene.objects.values() if isinstance(item, ObjectFluid)}
    if "smoke" in phases:
        pipeline.append("smoke")
    if "liquid" in phases:
        pipeline.append("apic")
    return pipeline or ["xpbd"]


def _validate_selector(selector: VertexSelector, vertex_count: int, path: str, error: Any) -> None:
    """Validate one cloth selector against its stable row-major vertex map."""
    if selector.kind == "indices":
        if not selector.indices or any(index < 0 or index >= vertex_count for index in selector.indices):
            error(path, "invalid_selector", "Selector contains an invalid particle index.", "Use in-range indices.")
    elif selector.kind == "edge" and selector.edge is None:
        error(path, "invalid_selector", "Edge selector requires an edge.", "Set top, bottom, left, or right.")
    elif selector.kind == "uv-corners" and not selector.corners:
        error(path, "invalid_selector", "Corner selector cannot be empty.", "Select at least one UV corner.")
