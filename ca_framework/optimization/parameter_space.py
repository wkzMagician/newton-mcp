# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Scene-aware activation and bounded optimization search spaces."""

from __future__ import annotations

from dataclasses import replace
from typing import Iterable

from ca_framework.scene import ObjectCloth, ObjectFluid, ObjectRigid, Scene

from .model import ParameterSpec


def activate_parameters(
    scene: Scene,
    specs: Iterable[ParameterSpec],
    *,
    optimizer: str = "tpe",
) -> tuple[ParameterSpec, ...]:
    """Return relevant enabled specs, capped by optimizer dimensionality."""
    has_rigid = any(isinstance(item, ObjectRigid) for item in scene.objects.values())
    has_cloth = any(isinstance(item, ObjectCloth) for item in scene.objects.values())
    fluids = [item for item in scene.objects.values() if isinstance(item, ObjectFluid)]
    has_liquid = any(item.phase == "liquid" for item in fluids)
    has_smoke = any(item.phase == "smoke" for item in fluids)
    active = []
    for spec in specs:
        if not spec.enabled or not _relevant(
            scene, spec.path, has_rigid=has_rigid, has_cloth=has_cloth, has_liquid=has_liquid, has_smoke=has_smoke
        ):
            continue
        if optimizer == "cmaes" and (spec.kind not in {"float", "int"} or spec.stage != "refine"):
            continue
        active.append(replace(spec, enabled=True))
    limit = 10 if optimizer == "cmaes" else 15
    return tuple(active[:limit])


def _relevant(
    scene: Scene,
    path: str,
    *,
    has_rigid: bool,
    has_cloth: bool,
    has_liquid: bool,
    has_smoke: bool,
) -> bool:
    parts = path.split(".")
    if len(parts) >= 2 and parts[0] == "objects":
        item = scene.objects.get(parts[1])
        return item is not None
    if path.startswith("settings.rigid"):
        return has_rigid
    if path.startswith("settings.cloth"):
        return has_cloth
    if path.startswith("settings.fluid"):
        return has_liquid or has_smoke
    if path.startswith("settings.coupling"):
        if "rigid_cloth" in path:
            return has_rigid and has_cloth
        if "rigid_fluid" in path:
            return has_rigid and (has_liquid or has_smoke)
        if "smoke_drag" in path:
            return has_smoke and (has_rigid or has_cloth)
        if "cloth_fluid" in path or "cloth_permeability" in path:
            return has_cloth and (has_liquid or has_smoke)
        return sum((has_rigid, has_cloth, has_liquid or has_smoke)) >= 2
    return True
