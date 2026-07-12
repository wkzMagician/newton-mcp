# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Normalize cloth-liquid pressure feedback into coupling exchanges."""

from __future__ import annotations

from typing import Any

import numpy as np

from .exchange import CouplingExchange


class CouplerClothFluid:
    """Collect APIC cloth-boundary pressure exchanges and diagnostics."""

    def collect(self, fluid_id: str, solver: Any) -> list[CouplingExchange]:
        """Return balanced exchanges for every registered cloth boundary."""
        exchanges = []
        for cloth_id, cloth_value in solver.cloth_linear_impulse.items():
            cloth_impulse = np.asarray(cloth_value, dtype=np.float64)
            fluid_impulse = np.asarray(
                solver.fluid_cloth_linear_impulse.get(cloth_id, -cloth_impulse), dtype=np.float64
            )
            cloth_angular_impulse = np.asarray(
                getattr(solver, "cloth_angular_impulse", {}).get(cloth_id, np.zeros(3)),
                dtype=np.float64,
            )
            fluid_angular_impulse = np.asarray(
                getattr(solver, "fluid_cloth_angular_impulse", {}).get(
                    cloth_id, -cloth_angular_impulse
                ),
                dtype=np.float64,
            )
            residual = self._interface_velocity_residual(solver, cloth_id)
            penetration = float(getattr(solver, "cloth_density_penetration", {}).get(cloth_id, 0.0))
            exchanges.append(
                CouplingExchange(
                    pair_type="cloth-fluid",
                    object_a=cloth_id,
                    object_b=fluid_id,
                    linear_impulse_a=cloth_impulse,
                    linear_impulse_b=fluid_impulse,
                    angular_impulse_a=cloth_angular_impulse,
                    angular_impulse_b=fluid_angular_impulse,
                    interface_residual=residual,
                    penetration=penetration,
                    exchange_energy_error=residual * float(np.linalg.norm(cloth_impulse)),
                )
            )
        return exchanges

    @staticmethod
    def _interface_velocity_residual(solver: Any, cloth_id: str) -> float:
        if not hasattr(solver, "solid_cloth"):
            solid_cloth = getattr(solver, "_solid_cloth", None)
            if solid_cloth is None:
                return 0.0
            cloth_index = next(
                (index for index, boundary in enumerate(solver.cloth_boundaries) if boundary.object_id == cloth_id),
                -1,
            )
            mask = solid_cloth == cloth_index
            if cloth_index < 0 or not np.any(mask):
                return 0.0
            u, v, w = solver.u.numpy(), solver.v.numpy(), solver.w.numpy()
            velocity = np.stack(
                (
                    0.5 * (u[:-1] + u[1:]),
                    0.5 * (v[:, :-1] + v[:, 1:]),
                    0.5 * (w[:, :, :-1] + w[:, :, 1:]),
                ),
                axis=-1,
            )
            return float(
                np.linalg.norm(velocity[mask] - solver._solid_velocity[mask], axis=1).max(initial=0.0)
            )
        cloth_index = next(
            (index for index, boundary in enumerate(solver.cloth_boundaries) if boundary.object_id == cloth_id),
            -1,
        )
        if cloth_index < 0:
            return 0.0
        differences: list[np.ndarray] = []
        for axis in range(3):
            for direction in (-1, 1):
                neighbour = solver._shifted(solver.solid_cloth, axis, direction, -1)
                interface = solver.fluid & (neighbour == cloth_index)
                if not np.any(interface):
                    continue
                solid_velocity = solver._shifted(solver.solid_velocity, axis, direction, 0.0)
                differences.append(solver.grid_velocity[interface] - solid_velocity[interface])
        if not differences:
            return 0.0
        values = np.concatenate(differences, axis=0)
        return float(np.linalg.norm(values, axis=1).max(initial=0.0))
