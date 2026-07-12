# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Normalize APIC rigid-boundary feedback into coupling exchanges."""

from __future__ import annotations

from typing import Any

import numpy as np

from .exchange import CouplingExchange


class CouplerRigidFluid:
    """Read one APIC step's rigid feedback and emit action-reaction records."""

    def collect(self, fluid_id: str, solver: Any, body_ids: dict[int, str]) -> list[CouplingExchange]:
        """Return balanced exchanges for all dynamic rigid boundaries."""
        exchanges = []
        residual = self._interface_velocity_residual(solver)
        for body, rigid_impulse_value in solver.rigid_linear_impulse.items():
            object_id = body_ids.get(body)
            if object_id is None:
                continue
            rigid_impulse = np.asarray(rigid_impulse_value, dtype=np.float64)
            angular_impulse = np.asarray(solver.rigid_angular_impulse.get(body, np.zeros(3)), dtype=np.float64)
            exchanges.append(
                CouplingExchange(
                    pair_type="rigid-fluid",
                    object_a=object_id,
                    object_b=fluid_id,
                    linear_impulse_a=rigid_impulse,
                    linear_impulse_b=-rigid_impulse,
                    angular_impulse_a=angular_impulse,
                    angular_impulse_b=-angular_impulse,
                    interface_residual=residual,
                    penetration=0.0,
                    exchange_energy_error=residual * float(np.linalg.norm(rigid_impulse)),
                )
            )
        return exchanges

    @staticmethod
    def diagnostics(fluid_id: str, solver: Any) -> dict[str, object]:
        """Return pressure/stabilization and clipping diagnostics."""
        return {
            "object_id": fluid_id,
            "pressure_impulse": {
                str(body): np.asarray(value).tolist() for body, value in solver.rigid_pressure_impulse.items()
            },
            "stabilization_impulse": {
                str(body): np.asarray(value).tolist()
                for body, value in solver.rigid_stabilization_impulse.items()
            },
            "impulse_clip_count": int(solver.impulse_clip_count),
            "interface_velocity_residual": CouplerRigidFluid._interface_velocity_residual(solver),
        }

    @staticmethod
    def _interface_velocity_residual(solver: Any) -> float:
        if not hasattr(solver, "_shifted"):
            solid_body = getattr(solver, "_solid_body", None)
            if solid_body is None or not np.any(solid_body >= 0):
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
            mask = solid_body >= 0
            return float(np.linalg.norm(velocity[mask] - solver._solid_velocity[mask], axis=1).max(initial=0.0))
        differences: list[np.ndarray] = []
        for axis in range(3):
            for direction in (-1, 1):
                neighbour_body = solver._shifted(solver.solid_body, axis, direction, -1)
                interface = solver.fluid & (neighbour_body >= 0)
                if not np.any(interface):
                    continue
                neighbour_velocity = solver._shifted(solver.solid_velocity, axis, direction, 0.0)
                differences.append(solver.grid_velocity[interface] - neighbour_velocity[interface])
        if not differences:
            return 0.0
        values = np.concatenate(differences, axis=0)
        return float(np.linalg.norm(values, axis=1).max(initial=0.0))
