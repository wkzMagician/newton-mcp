# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Scene-derived enabled multiphysics interaction graph."""

from __future__ import annotations

from dataclasses import dataclass

from ca_framework.scene.model import ObjectCloth, ObjectFluid, ObjectRigid, Scene


@dataclass(frozen=True, slots=True)
class CouplingEdge:
    """One enabled object pair and its coupling type."""

    pair_type: str
    object_a: str
    object_b: str


@dataclass(frozen=True, slots=True)
class CouplingGraph:
    """Immutable coupling edges compiled from scene types and settings."""

    edges: tuple[CouplingEdge, ...]

    @classmethod
    def from_scene(cls, scene: Scene) -> CouplingGraph:
        """Construct all enabled rigid-cloth, rigid-fluid, and cloth-fluid edges."""
        rigid = [item.id for item in scene.objects.values() if isinstance(item, ObjectRigid)]
        cloth = [item.id for item in scene.objects.values() if isinstance(item, ObjectCloth)]
        fluid = [item.id for item in scene.objects.values() if isinstance(item, ObjectFluid)]
        edges = []
        if scene.settings.coupling.rigid_cloth:
            edges.extend(CouplingEdge("rigid-cloth", left, right) for left in rigid for right in cloth)
        if scene.settings.coupling.rigid_fluid:
            edges.extend(CouplingEdge("rigid-fluid", left, right) for left in rigid for right in fluid)
        if scene.settings.coupling.cloth_fluid:
            edges.extend(CouplingEdge("cloth-fluid", left, right) for left in cloth for right in fluid)
        return cls(tuple(edges))

    def enables(self, pair_type: str) -> bool:
        """Return whether the graph contains at least one edge of a type."""
        return any(edge.pair_type == pair_type for edge in self.edges)
