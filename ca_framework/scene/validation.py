# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Static scene validation and resource estimation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from math import prod
from typing import Any, Literal

from .model import ObjectCloth, ObjectContainer, ObjectFluid, ObjectRigid, Scene


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
    grid_cells = 0
    for item in scene.objects.values():
        if isinstance(item, ObjectCloth):
            particles += prod(item.resolution)
        elif isinstance(item, ObjectFluid):
            grid_cells += prod(item.grid_resolution)
            if item.phase == "liquid" and item.particle_spacing > 0.0:
                particles += prod(max(1, int(size / item.particle_spacing)) for size in item.size)
    return {
        "particles": particles,
        "grid_cells": grid_cells,
        "simulation_frames": round(scene.settings.duration * scene.settings.fps),
        "render_frames": round(scene.settings.duration * scene.render.fps),
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

    return {
        "valid": not any(item.severity == "error" for item in findings),
        "diagnostics": [asdict(item) for item in findings],
        "resources": resources,
        "pipeline": select_pipeline(scene),
    }


def select_pipeline(scene: Scene) -> list[str]:
    """Return the deterministic solver schedule required by a scene."""
    pipeline = []
    if any(not isinstance(item, ObjectFluid) for item in scene.objects.values()):
        pipeline.append("xpbd" if scene.settings.solver in {"auto", "xpbd", "smoke", "apic"} else scene.settings.solver)
    phases = {item.phase for item in scene.objects.values() if isinstance(item, ObjectFluid)}
    if "smoke" in phases:
        pipeline.append("smoke")
    if "liquid" in phases:
        pipeline.append("apic")
    return pipeline or ["xpbd"]
