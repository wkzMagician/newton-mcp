# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Bidirectional rigid-cloth contact coupling."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from ca_framework.scene.model import ObjectRigid, Scene

from .exchange import CouplingExchange

CorrectionFunction = Callable[
    [np.ndarray, str, np.ndarray, np.ndarray, np.ndarray, float], tuple[np.ndarray | None, np.ndarray]
]
ArrayCopyFunction = Callable[[Any, np.ndarray], None]


class CouplerRigidCloth:
    """Detect and resolve triangle contacts against simple rigid shapes."""

    def __init__(self, correction: CorrectionFunction, copy_array: ArrayCopyFunction) -> None:
        self._correction = correction
        self._copy_array = copy_array

    def solve(
        self, scene: Scene, compiled: Any, state_out: Any, dt: float
    ) -> tuple[int, set[tuple[str, str]], list[CouplingExchange]]:
        """Apply normal/friction impulses and return all contact exchanges."""
        if state_out.particle_q is None:
            return 0, set(), []
        positions = state_out.particle_q.numpy()
        velocities = state_out.particle_qd.numpy()
        inverse_masses = compiled.model.particle_inv_mass.numpy()
        body_q = state_out.body_q.numpy() if state_out.body_q is not None else np.empty((0, 7))
        body_qd = state_out.body_qd.numpy() if state_out.body_qd is not None else np.empty((0, 6))
        body_inv_mass = compiled.model.body_inv_mass.numpy() if compiled.model.body_inv_mass is not None else np.empty(0)
        body_inv_inertia = (
            compiled.model.body_inv_inertia.numpy()
            if compiled.model.body_inv_inertia is not None
            else np.empty((0, 3, 3))
        )
        changed = False
        body_changed = False
        contact_count = 0
        contact_pairs: set[tuple[str, str]] = set()
        exchanges: list[CouplingExchange] = []
        colliders: list[tuple[str | None, str, np.ndarray, np.ndarray, np.ndarray, int]] = []
        if scene.render.ground:
            colliders.append(
                (None, "plane", np.zeros(3), np.array([1.0, 1.0, 0.0]), np.array([0, 0, 0, 1]), -1)
            )
        for object_id, item in scene.objects.items():
            if not isinstance(item, ObjectRigid) or item.shape not in {"sphere", "box"}:
                continue
            body = compiled.body_indices.get(object_id, -1)
            center = body_q[body, :3].copy() if body >= 0 else np.asarray(item.transform.position, dtype=float)
            rotation = body_q[body, 3:].copy() if body >= 0 else np.asarray(item.transform.rotation, dtype=float)
            size = np.asarray(item.size, dtype=float) * np.asarray(item.transform.scale, dtype=float)
            colliders.append((object_id, item.shape, center, size, rotation, body))

        for cloth_id, indices in compiled.cloth_particle_indices.items():
            cloth = scene.objects[cloth_id]
            width, height = cloth.resolution
            radius = cloth.collision_radius or cloth.thickness
            offset = indices[0]
            for y in range(height - 1):
                for x in range(width - 1):
                    lower = offset + y * width + x
                    triangles = (
                        (lower, lower + 1, lower + width + 1),
                        (lower, lower + width + 1, lower + width),
                    )
                    for triangle in triangles:
                        triangle_indices = np.asarray(triangle)
                        for collider_id, kind, center, size, rotation, body in colliders:
                            correction, barycentric = self._correction(
                                positions[triangle_indices], kind, center, size, rotation, radius
                            )
                            if correction is None:
                                continue
                            weights = barycentric * inverse_masses[triangle_indices]
                            cloth_weight = float(np.dot(barycentric, weights))
                            rigid_weight = float(body_inv_mass[body]) if body >= 0 else 0.0
                            normal = correction / max(float(np.linalg.norm(correction)), 1.0e-12)
                            contact_point = barycentric @ positions[triangle_indices]
                            lever = contact_point - body_q[body, :3] if body >= 0 else np.zeros(3)
                            rotational_weight = 0.0
                            if body >= 0 and body_inv_inertia.size:
                                lever_cross_normal = np.cross(lever, normal)
                                rotational_weight = float(
                                    lever_cross_normal @ body_inv_inertia[body] @ lever_cross_normal
                                )
                            denominator = cloth_weight + rigid_weight + rotational_weight
                            if denominator <= 0.0:
                                continue
                            compliance = scene.settings.rigid.contact_compliance
                            delta_lambda = float(np.linalg.norm(correction)) / (
                                denominator + compliance / max(dt * dt, 1.0e-12)
                            )
                            for vertex, weight in zip(triangle, weights, strict=True):
                                positions[vertex] += normal * delta_lambda * weight
                            if rigid_weight:
                                body_q[body, :3] -= normal * delta_lambda * rigid_weight
                                body_changed = True

                            cloth_velocity = barycentric @ velocities[triangle_indices]
                            rigid_velocity = (
                                body_qd[body, :3] + np.cross(body_qd[body, 3:], lever)
                                if body >= 0
                                else np.zeros(3)
                            )
                            relative_velocity = cloth_velocity - rigid_velocity
                            normal_speed = float(np.dot(relative_velocity, normal))
                            # ``normal`` points from the rigid body toward the cloth.  A
                            # positive relative speed therefore means that the rigid body
                            # is approaching the cloth and needs a separating impulse.
                            normal_impulse_magnitude = max(0.0, normal_speed / denominator)
                            cloth_impulse = normal * normal_impulse_magnitude
                            tangent_velocity = relative_velocity - normal * normal_speed
                            tangent_speed = float(np.linalg.norm(tangent_velocity))
                            tangential_impulse = np.zeros(3)
                            if tangent_speed > 1.0e-12 and normal_impulse_magnitude > 0.0:
                                tangent = tangent_velocity / tangent_speed
                                unconstrained = tangent_speed / denominator
                                friction = float(cloth.physical_material.friction_dynamic)
                                tangential_impulse = -tangent * min(
                                    unconstrained, friction * normal_impulse_magnitude
                                )
                                cloth_impulse += tangential_impulse
                            for vertex, barycentric_weight in zip(
                                triangle, barycentric, strict=True
                            ):
                                velocities[vertex] += (
                                    inverse_masses[vertex] * barycentric_weight * cloth_impulse
                                )
                            rigid_impulse = -cloth_impulse
                            angular_impulse = np.cross(lever, rigid_impulse)
                            if rigid_weight:
                                body_qd[body, :3] += rigid_impulse * rigid_weight
                                if body_inv_inertia.size:
                                    body_qd[body, 3:] += body_inv_inertia[body] @ angular_impulse
                            changed = True
                            contact_count += 1
                            if collider_id is not None:
                                contact_pairs.add(tuple(sorted((cloth_id, collider_id))))
                                exchanges.append(
                                    CouplingExchange(
                                        pair_type="rigid-cloth",
                                        object_a=cloth_id,
                                        object_b=collider_id,
                                        linear_impulse_a=cloth_impulse,
                                        linear_impulse_b=rigid_impulse,
                                        angular_impulse_a=-angular_impulse,
                                        angular_impulse_b=angular_impulse,
                                        interface_residual=max(0.0, -normal_speed),
                                        penetration=float(np.linalg.norm(correction)),
                                        contact_point=contact_point.copy(),
                                        normal=normal.copy(),
                                        barycentric_weights=barycentric.copy(),
                                        normal_impulse=normal_impulse_magnitude,
                                        tangential_impulse=float(np.linalg.norm(tangential_impulse)),
                                        exchange_energy_error=(
                                            max(0.0, float(np.dot(cloth_impulse, relative_velocity)))
                                        ),
                                    )
                                )
        if changed:
            self._copy_array(state_out.particle_q, positions)
            self._copy_array(state_out.particle_qd, velocities)
        if body_changed:
            self._copy_array(state_out.body_q, body_q)
            self._copy_array(state_out.body_qd, body_qd)
        return contact_count, contact_pairs, exchanges
