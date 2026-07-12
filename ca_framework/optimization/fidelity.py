# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Framework-owned fidelity transformations, never optimizer parameters."""

from __future__ import annotations

from copy import deepcopy
from math import ceil

from ca_framework.scene import ObjectCloth, ObjectFluid, Scene

from .model import FidelitySettings


def make_f1_scene(scene: Scene, settings: FidelitySettings) -> Scene:
    """Return a shorter, coarser rejection-only simulation scene."""
    candidate = deepcopy(scene)
    candidate.settings.duration *= settings.short_duration_fraction
    indexed_cloth = {
        constraint.object_id
        for constraint in candidate.constraints.values()
        if getattr(getattr(constraint, "selector", None), "kind", None) == "indices"
    }
    for item in candidate.objects.values():
        if isinstance(item, ObjectCloth) and item.id not in indexed_cloth and not any(
            selector.kind == "indices" for selector in item.pinned
        ):
            item.resolution = tuple(max(3, ceil(axis * settings.low_resolution_scale)) for axis in item.resolution)
        elif isinstance(item, ObjectFluid):
            item.grid_resolution = tuple(
                max(4, ceil(axis * settings.low_resolution_scale)) for axis in item.grid_resolution
            )
            if item.phase == "liquid":
                item.particle_spacing /= settings.low_resolution_scale
    return candidate
