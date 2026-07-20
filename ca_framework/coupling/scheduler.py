# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Deterministic solid-to-fluid-to-solid interface scheduler."""

from __future__ import annotations

from typing import Any

from ca_framework.scene.model import Scene

from .cloth_fluid import CouplerClothFluid
from .exchange import CouplingExchange
from .graph import CouplingGraph
from .rigid_cloth import ArrayCopyFunction, CorrectionFunction, CouplerRigidCloth
from .rigid_fluid import CouplerRigidFluid


class CoupledSimulationScheduler:
    """Execute enabled interface solvers in one stable substep order."""

    def __init__(
        self,
        scene: Scene,
        correction: CorrectionFunction,
        copy_array: ArrayCopyFunction,
    ) -> None:
        self.graph = CouplingGraph.from_scene(scene)
        self.rigid_cloth = CouplerRigidCloth(correction, copy_array)
        self.rigid_fluid = CouplerRigidFluid()
        self.cloth_fluid = CouplerClothFluid()

    def step_interfaces(
        self,
        scene: Scene,
        compiled: Any,
        state_in: Any,
        state_out: Any,
        dt: float,
    ) -> tuple[int, set[tuple[str, str]], list[CouplingExchange]]:
        """Resolve solid contact, step fluids, and collect all exchanges."""
        triangle_contacts = 0
        contact_pairs: set[tuple[str, str]] = set()
        exchanges: list[CouplingExchange] = []
        if self.graph.enables("rigid-cloth") or scene.render.ground:
            # Rigid-cloth uses position projections, so a single pass can leave a
            # fast rigid body embedded in a thin cloth. Re-evaluate contacts after
            # every projection; this is the numerical coupling iteration exposed
            # by the scene settings.
            for _ in range(scene.settings.coupling.iterations):
                contacts, pairs, rigid_cloth = self.rigid_cloth.solve(scene, compiled, state_out, dt)
                triangle_contacts += contacts
                contact_pairs.update(pairs)
                exchanges.extend(rigid_cloth)
        body_ids = {body: object_id for object_id, body in compiled.body_indices.items()}
        for fluid_id, solver in compiled.fluid_solvers.items():
            solver.step(state_in, state_out, compiled.control, compiled.contacts, dt)
            if self.graph.enables("rigid-fluid") and hasattr(solver, "rigid_linear_impulse"):
                exchanges.extend(self.rigid_fluid.collect(fluid_id, solver, body_ids))
            if self.graph.enables("cloth-fluid") and hasattr(solver, "cloth_linear_impulse"):
                exchanges.extend(self.cloth_fluid.collect(fluid_id, solver))
        return triangle_contacts, contact_pairs, exchanges
