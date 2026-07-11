# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Local compilation, preview, trajectory caching, rendering, and export."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
import tempfile
import time
from contextlib import suppress
from pathlib import Path
from threading import Event
from typing import Any

import numpy as np
import warp as wp

from newton.solvers import SolverNotifyFlags

from .compiler import SceneCompilerNewton
from .model import (
    ActionForce,
    ActionImpulse,
    ActionTransform,
    FieldRadial,
    FieldUniform,
    ObjectCloth,
    ObjectContainer,
    ObjectFluid,
    ObjectRigid,
    Scene,
)
from .validation import estimate_resources, validate_scene


class _SimulationCancelled(Exception):
    """Internal cooperative-cancellation signal."""


_PHYSICS_FAILURE_CODES = frozenset(
    {
        "state_escape",
        "non_finite_state",
        "energy_explosion",
        "pressure_nonconvergence",
        "particle_capacity_overflow",
        "contact_capacity_overflow",
    }
)


def _physics_validation(diagnostics: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize diagnostics that invalidate simulated physics."""
    failures = [item for item in diagnostics if item.get("code") in _PHYSICS_FAILURE_CODES]
    return {
        "valid": not failures,
        "failure_count": len(failures),
        "failure_codes": sorted({str(item["code"]) for item in failures}),
    }


class SceneExecutorLocal:
    """Deterministic local executor using Newton public solver routes."""

    def __init__(self) -> None:
        self.compiler = SceneCompilerNewton()

    def simulate(
        self,
        scene: Scene,
        *,
        frames: int | None = None,
        preview: bool = False,
        capture_cache: bool = False,
        cancel_event: Event | None = None,
        progress: dict[str, Any] | None = None,
        frame_callback: Any | None = None,
        compiled_scene: Any | None = None,
    ) -> dict[str, Any]:
        """Compile and advance the scene through Newton collision and solvers."""
        compiled = compiled_scene or self.compiler.compile(scene)
        scene_hash = hashlib.sha256(
            json.dumps(scene.to_dict(), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
        frame_count = frames if frames is not None else round(scene.settings.duration * scene.settings.fps)
        if frame_count <= 0:
            raise ValueError("frames must be positive")
        if preview:
            frame_count = min(frame_count, max(2, scene.settings.fps))
        frame_dt = 1.0 / scene.settings.fps
        substep_dt = frame_dt / scene.settings.substeps
        trajectories: dict[str, list[list[float]]] = {object_id: [] for object_id in scene.objects}
        diagnostics: list[dict[str, Any]] = []
        contact_count = 0
        rigid_contact_count = 0
        rigid_contact_candidate_count = 0
        active_rigid_contact_count = 0
        soft_contact_count = 0
        first_contact_time: float | None = None
        max_penetration = 0.0
        max_soft_penetration = 0.0
        first_contact_times: dict[str, float] = {}
        first_pair_contact_times: dict[str, float] = {}
        first_contact_pair_times: dict[str, float] = {}
        first_active_contact_pair_times: dict[str, float] = {}
        contact_pairs: set[tuple[str, str]] = set()
        active_contact_pairs: set[tuple[str, str]] = set()
        soft_contact_pairs: set[tuple[str, str]] = set()
        shape_objects = {shape: object_id for object_id, shapes in compiled.shape_indices.items() for shape in shapes}
        particle_objects = {
            particle: object_id
            for object_id, particles in compiled.cloth_particle_indices.items()
            for particle in particles
        }
        fired_impulses: set[str] = set()
        state_frames: list[dict[str, np.ndarray]] = []
        contact_records: list[dict[str, Any]] = []
        divergence_history: dict[str, list[float]] = {key: [] for key in compiled.fluid_solvers}
        fluid_mass_history: dict[str, list[float]] = {key: [] for key in compiled.fluid_solvers}
        max_body_linear_speed = 0.0
        max_body_angular_speed = 0.0
        max_particle_speed = 0.0
        state_in, state_out = compiled.state_0, compiled.state_1
        initial_particle_q = state_in.particle_q.numpy().copy() if state_in.particle_q is not None else np.empty((0, 3))
        completed_frames = 0
        for frame_index in range(frame_count):
            for substep_index in range(scene.settings.substeps):
                if cancel_event is not None and cancel_event.is_set():
                    raise _SimulationCancelled
                time_value = frame_index * frame_dt + substep_index * substep_dt
                state_in.clear_forces()
                self._apply_kinematics(scene, compiled, state_in, time_value, substep_dt)
                self._apply_particle_damping(scene, compiled, state_in, substep_dt)
                self._apply_forces(scene, compiled, state_in, time_value, fired_impulses)
                compiled.model.collide(state_in, compiled.contacts)
                current_contacts = int(compiled.contacts.rigid_contact_count.numpy()[0])
                current_soft_contacts = int(compiled.contacts.soft_contact_count.numpy()[0])
                if current_soft_contacts:
                    soft_penetrations = self._soft_contact_penetrations(compiled, state_in, current_soft_contacts)
                    max_soft_penetration = max(max_soft_penetration, float(soft_penetrations.max(initial=0.0)))
                    soft_particles = compiled.contacts.soft_contact_particle.numpy()[:current_soft_contacts]
                    soft_shapes = compiled.contacts.soft_contact_shape.numpy()[:current_soft_contacts]
                    for particle, shape in zip(soft_particles, soft_shapes, strict=True):
                        particle_object = particle_objects.get(int(particle))
                        shape_object = shape_objects.get(int(shape))
                        if particle_object is not None and shape_object is not None:
                            soft_contact_pairs.add(tuple(sorted((particle_object, shape_object))))
                rigid_contact_candidate_count += current_contacts
                if current_contacts:
                    penetrations = self._contact_penetrations(compiled, state_in, current_contacts)
                    active_contact_indices = np.flatnonzero(penetrations >= 0.0)
                    max_penetration = max(max_penetration, float(penetrations[active_contact_indices].max(initial=0.0)))
                else:
                    active_contact_indices = np.empty(0, dtype=int)
                active_contacts = len(active_contact_indices)
                rigid_contact_count += current_contacts
                active_rigid_contact_count += active_contacts
                soft_contact_count += current_soft_contacts
                contact_count += current_contacts + current_soft_contacts
                if (current_contacts or current_soft_contacts) and first_contact_time is None:
                    first_contact_time = time_value
                if current_contacts:
                    shapes_a = compiled.contacts.rigid_contact_shape0.numpy()[:current_contacts]
                    shapes_b = compiled.contacts.rigid_contact_shape1.numpy()[:current_contacts]
                    pairs = []
                    active_index_set = set(active_contact_indices.tolist())
                    for contact_index, (shape_a, shape_b) in enumerate(zip(shapes_a, shapes_b, strict=True)):
                        object_a = shape_objects.get(int(shape_a))
                        object_b = shape_objects.get(int(shape_b))
                        for object_id in (object_a, object_b):
                            if object_id is not None:
                                first_contact_times.setdefault(object_id, time_value)
                        if object_a is not None and object_b is not None and object_a != object_b:
                            pair = tuple(sorted((object_a, object_b)))
                            contact_pairs.add(pair)
                            if contact_index in active_index_set:
                                active_contact_pairs.add(pair)
                            pair_key = "|".join(sorted((object_a, object_b)))
                            if contact_index in active_index_set:
                                first_active_contact_pair_times.setdefault(pair_key, time_value)
                            first_contact_pair_times.setdefault(pair_key, time_value)
                            first_pair_contact_times.setdefault(object_a, time_value)
                            first_pair_contact_times.setdefault(object_b, time_value)
                            pairs.append([object_a, object_b])
                    contact_records.append(
                        {
                            "frame": frame_index,
                            "substep": substep_index,
                            "time": time_value,
                            "candidate_count": current_contacts,
                            "count": active_contacts,
                            "max_penetration": float(penetrations[active_contact_indices].max(initial=0.0)),
                            "object_pairs": pairs,
                        }
                    )
                compiled.solver.step(
                    state_in,
                    state_out,
                    compiled.control,
                    compiled.contacts,
                    substep_dt,
                )
                for fluid_solver in compiled.fluid_solvers.values():
                    fluid_solver.step(state_in, state_out, compiled.control, compiled.contacts, substep_dt)
                state_in, state_out = state_out, state_in
            self._record_trajectories(scene, compiled, state_in, trajectories)
            frame_state: dict[str, np.ndarray] = {}
            if state_in.body_q is not None:
                frame_state["body_q"] = state_in.body_q.numpy().astype(np.float32)
                frame_state["body_qd"] = state_in.body_qd.numpy().astype(np.float32)
                max_body_linear_speed = max(
                    max_body_linear_speed,
                    float(np.linalg.norm(frame_state["body_qd"][:, :3], axis=1).max(initial=0.0)),
                )
                max_body_angular_speed = max(
                    max_body_angular_speed,
                    float(np.linalg.norm(frame_state["body_qd"][:, 3:], axis=1).max(initial=0.0)),
                )
            if state_in.particle_q is not None:
                frame_state["cloth_q"] = state_in.particle_q.numpy().astype(np.float32)
                frame_state["cloth_qd"] = state_in.particle_qd.numpy().astype(np.float32)
                max_particle_speed = max(
                    max_particle_speed,
                    float(np.linalg.norm(frame_state["cloth_qd"], axis=1).max(initial=0.0)),
                )
            for object_id, solver in compiled.fluid_solvers.items():
                if scene.objects[object_id].phase == "smoke":
                    frame_state[f"{object_id}_density"] = solver.density.numpy().astype(np.float32)
                    fluid_mass_history[object_id].append(float(frame_state[f"{object_id}_density"].sum()))
                else:
                    frame_state[f"{object_id}_particles"] = solver.particle_position[: solver.particle_count].copy()
                    frame_state[f"{object_id}_velocities"] = solver.particle_velocity[: solver.particle_count].copy()
                    frame_state[f"{object_id}_masses"] = solver.particle_mass[: solver.particle_count].copy()
                    fluid_mass_history[object_id].append(float(frame_state[f"{object_id}_masses"].sum()))
                divergence = solver.divergence.numpy() if hasattr(solver.divergence, "numpy") else solver.divergence
                divergence_history[object_id].append(float(np.max(np.abs(divergence), initial=0.0)))
            if capture_cache:
                state_frames.append(frame_state)
            if frame_callback is not None:
                frame_callback(
                    frame_index,
                    {
                        **frame_state,
                        **{
                            object_id: np.asarray(samples[-1], dtype=np.float32)
                            for object_id, samples in trajectories.items()
                        },
                    },
                )
            completed_frames = frame_index + 1
            issue = self._state_diagnostic(state_in, frame_index)
            if issue is not None:
                diagnostics.append(issue)
                break
            for object_id, fluid_solver in compiled.fluid_solvers.items():
                if getattr(fluid_solver, "capacity_overflow", False):
                    diagnostics.append(
                        {
                            "code": "particle_capacity_overflow",
                            "frame": frame_index,
                            "object_id": object_id,
                            "message": "Liquid emitter exceeded its fixed particle capacity.",
                        }
                    )
                    fluid_solver.capacity_overflow = False
                divergence = getattr(fluid_solver, "divergence", None)
                divergence_values = divergence.numpy() if hasattr(divergence, "numpy") else divergence
                if divergence_values is not None and (
                    not np.isfinite(divergence_values).all() or np.max(np.abs(divergence_values)) > 100.0
                ):
                    diagnostics.append(
                        {
                            "code": "pressure_nonconvergence",
                            "frame": frame_index,
                            "object_id": object_id,
                            "message": "Pressure projection divergence exceeded the stability threshold.",
                        }
                    )
            if (
                int(compiled.contacts.rigid_contact_count.numpy()[0]) >= compiled.model.rigid_contact_max
                and compiled.model.rigid_contact_max > 0
            ):
                diagnostics.append(
                    {
                        "code": "contact_capacity_overflow",
                        "frame": frame_index,
                        "message": "Rigid contact storage reached its configured capacity.",
                    }
                )
            if progress is not None:
                progress.update(stage="simulation", frame=frame_index + 1, total_frames=frame_count)
        fluid_stats = {}
        for object_id, item in scene.objects.items():
            if isinstance(item, ObjectFluid):
                resources = estimate_resources(Scene(name="estimate", objects={object_id: item}))
                stats = {
                    "phase": item.phase,
                    "particles": resources["particles"] if item.phase == "liquid" else 0,
                    "grid_cells": math.prod(item.grid_resolution),
                    "finite": True,
                }
                if item.phase == "smoke":
                    solver = compiled.fluid_solvers[object_id]
                    density = solver.density.numpy()
                    divergence = solver.divergence.numpy()
                    stats.update(
                        density_mass=float(density.sum()),
                        occupied_cells=int(np.count_nonzero(density > 1.0e-5)),
                        max_divergence=float(np.max(np.abs(divergence))),
                        finite=bool(np.isfinite(density).all() and np.isfinite(divergence).all()),
                    )
                else:
                    solver = compiled.fluid_solvers[object_id]
                    positions = solver.particle_position[: solver.particle_count]
                    cell_counts = solver.particle_cell_counts()
                    stats.update(
                        particles=solver.particle_count,
                        fluid_mass=float(solver.particle_mass[: solver.particle_count].sum()),
                        min_particles_per_cell=int(cell_counts.min()) if cell_counts.size else 0,
                        max_particles_per_cell=int(cell_counts.max()) if cell_counts.size else 0,
                        active_cells=int(np.count_nonzero(solver.fluid)),
                        max_divergence=float(np.max(np.abs(solver.divergence))),
                        rigid_linear_impulse={
                            str(body): impulse.tolist() for body, impulse in solver.rigid_linear_impulse.items()
                        },
                        rigid_angular_impulse={
                            str(body): impulse.tolist() for body, impulse in solver.rigid_angular_impulse.items()
                        },
                        finite=bool(
                            np.isfinite(positions).all()
                            and np.isfinite(solver.grid_velocity).all()
                            and np.isfinite(solver.pressure).all()
                        ),
                    )
                mass_samples = fluid_mass_history[object_id]
                initial_mass = mass_samples[0] if mass_samples else 0.0
                final_mass = mass_samples[-1] if mass_samples else 0.0
                stats.update(
                    initial_mass=initial_mass,
                    final_mass=final_mass,
                    relative_mass_change=(final_mass - initial_mass) / initial_mass if initial_mass > 0.0 else 0.0,
                    divergence_history=divergence_history[object_id],
                    peak_divergence=max(divergence_history[object_id], default=0.0),
                    mass_conservation_applicable=not item.emitters,
                )
                fluid_stats[object_id] = stats
        cloth_stats = {}
        final_particle_q = state_in.particle_q.numpy() if state_in.particle_q is not None else np.empty((0, 3))
        final_particle_qd = state_in.particle_qd.numpy() if state_in.particle_qd is not None else np.empty((0, 3))
        for object_id, indices in compiled.cloth_particle_indices.items():
            selected = np.asarray(indices, dtype=int)
            initial = initial_particle_q[selected]
            final = final_particle_q[selected]
            velocities = final_particle_qd[selected]
            cloth_stats[object_id] = {
                "finite": bool(np.isfinite(final).all() and np.isfinite(velocities).all()),
                "max_displacement": float(np.linalg.norm(final - initial, axis=1).max(initial=0.0)),
                "max_speed": float(np.linalg.norm(velocities, axis=1).max(initial=0.0)),
                "bounds_min": final.min(axis=0).tolist(),
                "bounds_max": final.max(axis=0).tolist(),
            }
        physics = _physics_validation(diagnostics)
        return {
            "status": "completed" if physics["valid"] else "physics_failed",
            "scene": scene.name,
            "scene_hash": scene_hash,
            "frames": completed_frames,
            "backend": "newton",
            "pipeline": compiled.pipeline,
            "metrics": {
                "contacts": contact_count,
                "rigid_contacts": rigid_contact_count,
                "rigid_contact_candidates": rigid_contact_candidate_count,
                "active_rigid_contacts": active_rigid_contact_count,
                "soft_contacts": soft_contact_count,
                "first_contact_time": first_contact_time,
                "max_penetration": max_penetration,
                "max_soft_penetration": max_soft_penetration,
                "solver_residual": None,
                "first_contact_times": first_contact_times,
                "first_pair_contact_times": first_pair_contact_times,
                "first_contact_pair_times": first_contact_pair_times,
                "first_active_contact_pair_times": first_active_contact_pair_times,
                "contact_pairs": [list(pair) for pair in sorted(contact_pairs)],
                "active_contact_pairs": [list(pair) for pair in sorted(active_contact_pairs)],
                "soft_contact_pairs": [list(pair) for pair in sorted(soft_contact_pairs)],
                "fluid": fluid_stats,
                "cloth": cloth_stats,
                "kinematics": {
                    "max_body_linear_speed": max_body_linear_speed,
                    "max_body_angular_speed": max_body_angular_speed,
                    "max_particle_speed": max_particle_speed,
                },
                "resources": estimate_resources(scene),
                "physics": physics,
            },
            "trajectories": trajectories,
            "diagnostics": diagnostics,
            **({"state_frames": state_frames, "contact_records": contact_records} if capture_cache else {}),
        }

    @staticmethod
    def _copy_array(destination: Any, values: np.ndarray) -> None:
        wp.copy(destination, wp.array(values, dtype=destination.dtype, device=destination.device))

    def _apply_particle_damping(self, scene: Scene, compiled: Any, state: Any, dt: float) -> None:
        if state.particle_qd is None:
            return
        velocities = state.particle_qd.numpy()
        changed = False
        for object_id, indices in compiled.cloth_particle_indices.items():
            item = scene.objects[object_id]
            damping = max(0.0, item.damping + item.air_drag)
            if damping:
                velocities[indices] *= math.exp(-damping * dt)
                changed = True
        if changed:
            self._copy_array(state.particle_qd, velocities)

    def _apply_kinematics(self, scene: Scene, compiled: Any, state: Any, time_value: float, substep_dt: float) -> None:
        if state.body_q is None:
            return
        body_q = state.body_q.numpy()
        body_qd = state.body_qd.numpy()
        changed = False
        shape_scales = compiled.model.shape_scale.numpy()
        scales_changed = False
        for action in scene.actions.values():
            if not isinstance(action, ActionTransform) or action.object_id not in compiled.body_indices:
                continue
            transform = self._sample_transform(action, time_value)
            if transform is None:
                continue
            body_q[compiled.body_indices[action.object_id], :3] = transform[0]
            body_q[compiled.body_indices[action.object_id], 3:] = transform[1]
            item = scene.objects[action.object_id]
            base_scale = np.asarray(item.transform.scale, dtype=np.float32)
            ratio = np.divide(transform[2], base_scale, out=np.ones(3, dtype=np.float32), where=base_scale != 0.0)
            for shape_index in compiled.shape_indices.get(action.object_id, []):
                shape_scales[shape_index] = compiled.initial_shape_scales[shape_index] * ratio
                scales_changed = True
            next_transform = self._sample_transform(action, time_value + substep_dt)
            if next_transform is not None:
                body_qd[compiled.body_indices[action.object_id], :3] = (next_transform[0] - transform[0]) / substep_dt
                body_qd[compiled.body_indices[action.object_id], 3:] = self._quat_angular_velocity(
                    transform[1], next_transform[1], substep_dt
                )
            changed = True
        if changed:
            self._copy_array(state.body_q, body_q)
            self._copy_array(state.body_qd, body_qd)
        if scales_changed:
            self._copy_array(compiled.model.shape_scale, shape_scales)
            compiled.solver.notify_model_changed(SolverNotifyFlags.SHAPE_PROPERTIES)

    def _apply_forces(
        self,
        scene: Scene,
        compiled: Any,
        state: Any,
        time_value: float,
        fired_impulses: set[str],
    ) -> None:
        forces = state.body_f.numpy() if state.body_f is not None else np.empty((0, 6), dtype=np.float32)
        velocities = state.body_qd.numpy() if state.body_qd is not None else np.empty((0, 6), dtype=np.float32)
        positions = state.body_q.numpy()[:, :3] if state.body_q is not None else np.empty((0, 3), dtype=np.float32)
        masses = compiled.model.body_mass.numpy() if compiled.model.body_mass is not None else np.empty(0)
        changed_velocity = False
        for field in scene.fields.values():
            targets = field.object_ids or list(compiled.body_indices)
            for object_id in targets:
                body = compiled.body_indices.get(object_id)
                if body is None or scene.objects[object_id].motion != "dynamic":
                    continue
                if isinstance(field, FieldUniform):
                    vector = np.asarray(field.vector, dtype=np.float32)
                elif isinstance(field, FieldRadial):
                    offset = positions[body] - np.asarray(field.center)
                    distance = max(float(np.linalg.norm(offset)), 1.0e-6)
                    factor = 1.0 if field.falloff == "constant" else max(0.0, 1.0 - distance)
                    if field.falloff == "inverse-square":
                        factor = 1.0 / (distance * distance)
                    vector = offset / distance * field.strength * factor
                else:
                    continue
                forces[body, :3] += vector * (masses[body] if field.mode == "acceleration" else 1.0)
        for action in scene.actions.values():
            body = compiled.body_indices.get(action.object_id)
            if body is None:
                continue
            if isinstance(action, ActionForce) and action.start_time <= time_value < action.end_time:
                forces[body, :3] += action.force
            elif isinstance(action, ActionImpulse) and action.id not in fired_impulses and time_value >= action.time:
                if masses[body] > 0.0:
                    impulse = np.asarray(action.impulse)
                    velocities[body, :3] += impulse / masses[body]
                    if action.point is not None:
                        angular_impulse = np.cross(np.asarray(action.point), impulse)
                        inertia = compiled.model.body_inertia.numpy()[body]
                        velocities[body, 3:] += np.linalg.solve(
                            inertia + np.eye(3) * 1.0e-8,
                            angular_impulse,
                        )
                    changed_velocity = True
                fired_impulses.add(action.id)
        if state.body_f is not None:
            self._copy_array(state.body_f, forces)
        if changed_velocity and state.body_qd is not None:
            self._copy_array(state.body_qd, velocities)
        if state.particle_f is None:
            return
        particle_forces = state.particle_f.numpy()
        particle_positions = state.particle_q.numpy()
        inverse_masses = compiled.model.particle_inv_mass.numpy()
        particle_masses = np.divide(
            1.0,
            inverse_masses,
            out=np.zeros_like(inverse_masses),
            where=inverse_masses > 0.0,
        )
        for field in scene.fields.values():
            targets = field.object_ids or list(compiled.cloth_particle_indices)
            for object_id in targets:
                indices = compiled.cloth_particle_indices.get(object_id)
                if indices is None:
                    continue
                if isinstance(field, FieldUniform):
                    vectors = np.broadcast_to(np.asarray(field.vector), (len(indices), 3))
                elif isinstance(field, FieldRadial):
                    offsets = particle_positions[indices] - np.asarray(field.center)
                    distances = np.maximum(np.linalg.norm(offsets, axis=1), 1.0e-6)
                    factors = np.ones_like(distances)
                    if field.falloff == "linear":
                        factors = np.maximum(0.0, 1.0 - distances)
                    elif field.falloff == "inverse-square":
                        factors = 1.0 / (distances * distances)
                    vectors = offsets / distances[:, None] * field.strength * factors[:, None]
                else:
                    continue
                if field.mode == "acceleration":
                    vectors = vectors * particle_masses[np.asarray(indices)][:, None]
                particle_forces[indices] += vectors
        for action in scene.actions.values():
            indices = compiled.cloth_particle_indices.get(action.object_id)
            if (
                indices is not None
                and isinstance(action, ActionForce)
                and action.start_time <= time_value < action.end_time
            ):
                particle_forces[indices] += np.asarray(action.force) / len(indices)
        self._copy_array(state.particle_f, particle_forces)

    @staticmethod
    def _sample_transform(
        action: ActionTransform, time_value: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        if not action.keyframes:
            return None
        if time_value <= action.keyframes[0].time:
            sample = action.keyframes[0].transform
            return np.asarray(sample.position), np.asarray(sample.rotation), np.asarray(sample.scale)
        if time_value >= action.keyframes[-1].time:
            sample = action.keyframes[-1].transform
            return np.asarray(sample.position), np.asarray(sample.rotation), np.asarray(sample.scale)
        for left, right in zip(action.keyframes, action.keyframes[1:], strict=False):
            if left.time <= time_value <= right.time:
                alpha = (time_value - left.time) / (right.time - left.time)
                position = (
                    np.asarray(left.transform.position) * (1.0 - alpha) + np.asarray(right.transform.position) * alpha
                )
                quaternion = SceneExecutorLocal._slerp(left.transform.rotation, right.transform.rotation, alpha)
                scale = np.asarray(left.transform.scale) * (1.0 - alpha) + np.asarray(right.transform.scale) * alpha
                return position, quaternion, scale
        return None

    @staticmethod
    def _slerp(left: Any, right: Any, alpha: float) -> np.ndarray:
        q0, q1 = np.asarray(left, dtype=np.float64), np.asarray(right, dtype=np.float64)
        dot = float(np.dot(q0, q1))
        if dot < 0.0:
            q1, dot = -q1, -dot
        if dot > 0.9995:
            result = q0 + alpha * (q1 - q0)
            return result / np.linalg.norm(result)
        theta = math.acos(max(-1.0, min(1.0, dot)))
        return (math.sin((1.0 - alpha) * theta) * q0 + math.sin(alpha * theta) * q1) / math.sin(theta)

    @staticmethod
    def _quat_angular_velocity(left: np.ndarray, right: np.ndarray, dt: float) -> np.ndarray:
        x0, y0, z0, w0 = left
        x1, y1, z1, w1 = right
        delta = np.asarray(
            [
                w1 * -x0 + x1 * w0 + y1 * -z0 - z1 * -y0,
                w1 * -y0 - x1 * -z0 + y1 * w0 + z1 * -x0,
                w1 * -z0 + x1 * -y0 - y1 * -x0 + z1 * w0,
                w1 * w0 - x1 * -x0 - y1 * -y0 - z1 * -z0,
            ]
        )
        if delta[3] < 0.0:
            delta = -delta
        vector_norm = float(np.linalg.norm(delta[:3]))
        if vector_norm < 1.0e-8:
            return np.zeros(3)
        angle = 2.0 * math.atan2(vector_norm, max(float(delta[3]), 1.0e-8))
        return delta[:3] / vector_norm * angle / dt

    @staticmethod
    def _record_trajectories(scene: Scene, compiled: Any, state: Any, trajectories: dict[str, Any]) -> None:
        body_q = state.body_q.numpy() if state.body_q is not None else np.empty((0, 7))
        particle_q = state.particle_q.numpy() if state.particle_q is not None else np.empty((0, 3))
        for object_id, item in scene.objects.items():
            if object_id in compiled.body_indices:
                position = body_q[compiled.body_indices[object_id], :3]
            elif isinstance(item, ObjectCloth):
                position = particle_q[compiled.cloth_particle_indices[object_id]].mean(axis=0)
            else:
                position = np.asarray(item.transform.position)
            trajectories[object_id].append([round(float(value), 8) for value in position])

    @staticmethod
    def _state_diagnostic(state: Any, frame_index: int) -> dict[str, Any] | None:
        arrays = tuple(
            array.numpy()
            for array in (state.body_q, state.body_qd, state.particle_q, state.particle_qd)
            if array is not None
        )
        if any(not np.isfinite(values).all() for values in arrays):
            return {"code": "non_finite_state", "frame": frame_index, "message": "Newton state contains NaN or Inf."}
        if any(values.size and np.max(np.abs(values)) > 1.0e6 for values in arrays):
            return {"code": "state_escape", "frame": frame_index, "message": "State exceeded the safety bound."}
        velocity_arrays = tuple(array.numpy() for array in (state.body_qd, state.particle_qd) if array is not None)
        if any(values.size and np.max(np.linalg.norm(values, axis=1)) > 1.0e4 for values in velocity_arrays):
            return {
                "code": "energy_explosion",
                "frame": frame_index,
                "message": "State velocity exceeded the energy safety bound.",
            }
        return None

    @staticmethod
    def _contact_penetrations(compiled: Any, state: Any, count: int) -> np.ndarray:
        """Return signed penetration depths for collision candidates [m]."""
        contacts = compiled.contacts
        shape_a = contacts.rigid_contact_shape0.numpy()[:count]
        shape_b = contacts.rigid_contact_shape1.numpy()[:count]
        point_a = contacts.rigid_contact_point0.numpy()[:count]
        point_b = contacts.rigid_contact_point1.numpy()[:count]
        normal = contacts.rigid_contact_normal.numpy()[:count]
        margin = contacts.rigid_contact_margin0.numpy()[:count] + contacts.rigid_contact_margin1.numpy()[:count]
        shape_bodies = compiled.model.shape_body.numpy()
        transforms = state.body_q.numpy() if state.body_q is not None else np.empty((0, 7))

        def world_point(point: np.ndarray, shape: int) -> np.ndarray:
            if shape < 0:
                return point
            body = int(shape_bodies[shape])
            if body < 0:
                return point
            transform = transforms[body]
            xyz = transform[3:6]
            rotated = point + 2.0 * (transform[6] * np.cross(xyz, point) + np.cross(xyz, np.cross(xyz, point)))
            return transform[:3] + rotated

        penetrations = np.empty(count, dtype=np.float64)
        for index in range(count):
            a = world_point(point_a[index], int(shape_a[index]))
            b = world_point(point_b[index], int(shape_b[index]))
            distance = float(np.dot(-normal[index], b - a) - margin[index])
            penetrations[index] = -distance
        return penetrations

    @staticmethod
    def _soft_contact_penetrations(compiled: Any, state: Any, count: int) -> np.ndarray:
        """Return non-negative cloth/shape penetration depths [m]."""
        contacts = compiled.contacts
        particles = contacts.soft_contact_particle.numpy()[:count]
        shapes = contacts.soft_contact_shape.numpy()[:count]
        body_points = contacts.soft_contact_body_pos.numpy()[:count]
        normals = contacts.soft_contact_normal.numpy()[:count]
        positions = state.particle_q.numpy()
        radii = compiled.model.particle_radius.numpy()
        shape_bodies = compiled.model.shape_body.numpy()
        body_transforms = state.body_q.numpy() if state.body_q is not None else np.empty((0, 7))
        penetrations = np.zeros(count, dtype=np.float64)
        for index, (particle, shape) in enumerate(zip(particles, shapes, strict=True)):
            body = int(shape_bodies[shape])
            body_point = body_points[index]
            if body >= 0:
                transform = body_transforms[body]
                xyz = transform[3:6]
                body_point = (
                    transform[:3]
                    + body_point
                    + 2.0 * (transform[6] * np.cross(xyz, body_point) + np.cross(xyz, np.cross(xyz, body_point)))
                )
            separation = float(np.dot(normals[index], positions[particle] - body_point) - radii[particle])
            penetrations[index] = max(0.0, -separation)
        return penetrations

    def preview(self, scene: Scene, *, frames: int | None = None) -> dict[str, Any]:
        """Run a short, reduced-cost simulation and return sampled keyframes."""
        preview_scene = Scene.from_dict(scene.to_dict())
        preview_scene.settings.duration = min(1.0, scene.settings.duration)
        preview_scene.settings.max_particles = min(50_000, scene.settings.max_particles)
        preview_scene.render.resolution = (640, 360)
        for item in preview_scene.objects.values():
            if not isinstance(item, ObjectFluid):
                continue
            limit = 32 if item.phase == "smoke" else 24
            item.grid_resolution = tuple(min(axis, limit) for axis in item.grid_resolution)
            if item.phase == "liquid":
                estimated = math.prod(max(1, int(size / item.particle_spacing)) for size in item.size)
                if estimated > preview_scene.settings.max_particles:
                    item.particle_spacing *= (estimated / preview_scene.settings.max_particles) ** (1.0 / 3.0)
        result = self.simulate(preview_scene, frames=frames, preview=True)
        result["preview"] = True
        result["effective_settings"] = {
            "duration": preview_scene.settings.duration,
            "resolution": list(preview_scene.render.resolution),
            "max_particles": preview_scene.settings.max_particles,
            "fluids": {
                item.id: {
                    "grid_resolution": list(item.grid_resolution),
                    "particle_spacing": item.particle_spacing,
                }
                for item in preview_scene.objects.values()
                if isinstance(item, ObjectFluid)
            },
        }
        trajectories = result.pop("trajectories")
        result["keyframes"] = {
            key: [samples[index] for index in sorted({0, len(samples) // 2, len(samples) - 1})]
            for key, samples in trajectories.items()
        }
        result["keyframe_paths"] = {
            key: [
                {
                    "frame": index,
                    "time": index / preview_scene.settings.fps,
                    "path": f"objects.{key}.transform.position",
                    "value": samples[index],
                }
                for index in sorted({0, len(samples) // 2, len(samples) - 1})
            ]
            for key, samples in trajectories.items()
        }
        result.pop("state_frames", None)
        result.pop("contact_records", None)
        return result

    def render(self, scene: Scene, *, output: Path) -> dict[str, Any]:
        """Render a fixed-rate placeholder MP4 from a compiled trajectory cache.

        The video encoder is intentionally discovered at runtime, keeping it out
        of Newton's core dependencies. A render failure never invalidates the
        returned simulation metrics.
        """
        simulation = self.simulate(scene)
        return {**simulation, **self._encode_video(scene, output)}

    @staticmethod
    def _encode_video(scene: Scene, output: Path, cache_dir: Path | None = None) -> dict[str, Any]:
        """Encode rendered frames without advancing simulation state."""
        output = output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            return {
                "status": "render_failed",
                "output": str(output),
                "diagnostics": [{"code": "encoder_missing", "message": "Install the render extra or ffmpeg."}],
            }
        width, height = scene.render.resolution
        with tempfile.TemporaryDirectory() as temporary:
            temporary_path = Path(temporary)
            if cache_dir is not None:
                try:
                    render_metadata = SceneExecutorLocal._render_cache_frames(scene, cache_dir, temporary_path)
                except Exception as exc:
                    output.unlink(missing_ok=True)
                    return {
                        "status": "render_failed",
                        "output": str(output),
                        "diagnostics": [{"code": "opengl_render_failed", "message": str(exc)}],
                    }
                input_arguments = ["-framerate", str(scene.render.fps), "-i", str(temporary_path / "frame-%06d.ppm")]
            else:
                render_metadata = {}
                frame = temporary_path / "frame-000000.ppm"
                frame.write_bytes(f"P6\n{width} {height}\n255\n".encode() + bytes((28, 31, 38)) * width * height)
                input_arguments = ["-loop", "1", "-i", str(frame), "-t", str(scene.settings.duration)]
            command = [
                ffmpeg,
                "-loglevel",
                "error",
                "-y",
                *input_arguments,
                "-r",
                str(scene.render.fps),
                "-c:v",
                "libx264",
                "-vf",
                "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                "-pix_fmt",
                "yuv420p",
                str(output),
            ]
            completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if completed.returncode:
            return {
                "status": "render_failed",
                "output": str(output),
                "diagnostics": [{"code": "encoder_failed", "message": completed.stderr.strip()}],
            }
        return {
            "status": "completed",
            "output": str(output),
            "render_frames": round(scene.settings.duration * scene.render.fps),
            **render_metadata,
        }

    @staticmethod
    def _render_cache_frames(scene: Scene, cache_dir: Path, output_dir: Path) -> dict[str, Any]:
        """Rasterize cached bodies, cloth, smoke, and liquid without simulation."""
        cache_manifest = json.loads((cache_dir / "manifest.json").read_text(encoding="utf-8"))
        simulation_frames = cache_manifest["frames"]
        if simulation_frames <= 0:
            raise ValueError("Cannot render an empty simulation cache")
        frame_paths = [cache_dir / "frames" / f"{index:06d}.npz" for index in range(simulation_frames)]
        return SceneExecutorLocal._render_cache_frames_gl(scene, frame_paths, output_dir)
        bounds_points = []
        for frame_path in frame_paths:
            with np.load(frame_path) as state:
                for key in state.files:
                    values = state[key]
                    if key == "body_q" and values.size:
                        bounds_points.append(values[:, :3])
                    elif (key == "cloth_q" or key.endswith("_particles")) and values.size:
                        bounds_points.append(values.reshape(-1, 3))
        for item in scene.objects.values():
            bounds_points.append(np.asarray(item.transform.position, dtype=np.float32).reshape(1, 3))
            object_size = getattr(item, "size", getattr(item, "inner_size", None))
            if object_size is not None and len(object_size) == 3:
                center = np.asarray(item.transform.position, dtype=np.float32)
                half_extent = np.asarray(object_size, dtype=np.float32) * np.asarray(item.transform.scale) * 0.5
                if isinstance(item, ObjectContainer):
                    center = center + np.asarray([0.0, 0.0, half_extent[2]])
                bounds_points.append(
                    np.asarray(
                        [center + (np.asarray(signs) * 2.0 - 1.0) * half_extent for signs in np.ndindex(2, 2, 2)]
                    )
                )
        points = np.concatenate(bounds_points) if bounds_points else np.asarray([[0.0, 0.0, 0.0]])
        camera = scene.render.camera
        if camera.position is not None and camera.target is not None:
            view = np.asarray(camera.target, dtype=float) - np.asarray(camera.position, dtype=float)
            view /= max(np.linalg.norm(view), 1.0e-8)
            right = np.cross(view, np.asarray(camera.up, dtype=float))
            if np.linalg.norm(right) < 1.0e-6:
                right = np.asarray([1.0, 0.0, 0.0])
            right /= np.linalg.norm(right)
            camera_up = np.cross(right, view)
            camera_up /= max(np.linalg.norm(camera_up), 1.0e-8)

            def project_position(position: np.ndarray) -> np.ndarray:
                relative = position - np.asarray(camera.position)
                return np.asarray([np.dot(relative, right), np.dot(relative, view), np.dot(relative, camera_up)])

            projected_points = np.asarray([project_position(position) for position in points])
        else:

            def project_position(position: np.ndarray) -> np.ndarray:
                return position

            projected_points = points
        minimum, maximum = projected_points.min(axis=0), projected_points.max(axis=0)
        span_x = max(float(maximum[0] - minimum[0]), 1.0)
        span_z = max(float(maximum[2] - minimum[2]), 1.0)
        margin = 0.12
        minimum[0] -= span_x * margin
        maximum[0] += span_x * margin
        minimum[2] -= span_z * margin
        maximum[2] += span_z * margin
        width, height = scene.render.resolution
        render_frames = max(1, round(scene.settings.duration * scene.render.fps))
        body_items = [
            item
            for item in scene.objects.values()
            if isinstance(item, (ObjectRigid, ObjectContainer)) and item.motion != "static"
        ]

        def pixel(position: np.ndarray) -> tuple[int, int]:
            projected = project_position(position)
            x = int((projected[0] - minimum[0]) / (maximum[0] - minimum[0]) * (width - 1))
            y = height - 1 - int((projected[2] - minimum[2]) / (maximum[2] - minimum[2]) * (height - 1))
            return x, y

        def draw_disc(image: np.ndarray, center: tuple[int, int], radius: int, color: tuple[int, int, int]) -> None:
            x, y = center
            x0, x1 = max(0, x - radius), min(width, x + radius + 1)
            y0, y1 = max(0, y - radius), min(height, y + radius + 1)
            yy, xx = np.ogrid[y0:y1, x0:x1]
            mask = (xx - x) ** 2 + (yy - y) ** 2 <= radius * radius
            image[y0:y1, x0:x1][mask] = color

        def draw_line(
            image: np.ndarray,
            start: tuple[int, int],
            end: tuple[int, int],
            color: tuple[int, int, int],
            thickness: int = 1,
        ) -> None:
            count = max(abs(end[0] - start[0]), abs(end[1] - start[1]), 1) + 1
            xs = np.linspace(start[0], end[0], count).astype(int)
            ys = np.linspace(start[1], end[1], count).astype(int)
            for offset_x in range(-thickness + 1, thickness):
                for offset_y in range(-thickness + 1, thickness):
                    valid = (
                        (xs + offset_x >= 0) & (xs + offset_x < width) & (ys + offset_y >= 0) & (ys + offset_y < height)
                    )
                    image[ys[valid] + offset_y, xs[valid] + offset_x] = color

        def object_bounds(item: ObjectRigid | ObjectContainer, position: np.ndarray) -> tuple[int, int, int, int]:
            size = np.asarray(item.size if isinstance(item, ObjectRigid) else item.inner_size, dtype=float)
            center = np.asarray(position, dtype=float)
            if isinstance(item, ObjectContainer):
                center = center + np.asarray([0.0, 0.0, size[2] * 0.5])
            corners = [
                center + (np.asarray(signs, dtype=float) * 2.0 - 1.0) * size * 0.5 for signs in np.ndindex(2, 2, 2)
            ]
            pixels = np.asarray([pixel(corner) for corner in corners])
            return int(pixels[:, 0].min()), int(pixels[:, 1].min()), int(pixels[:, 0].max()), int(pixels[:, 1].max())

        def draw_object(image: np.ndarray, item: ObjectRigid | ObjectContainer, position: np.ndarray) -> None:
            color = tuple(int(255 * value) for value in item.visual_material.color[:3])
            x0, y0, x1, y1 = object_bounds(item, position)
            if isinstance(item, ObjectRigid) and item.shape == "sphere":
                draw_disc(image, pixel(position), max(3, min(x1 - x0, y1 - y0) // 2), color)
            elif isinstance(item, ObjectContainer):
                draw_line(image, (x0, y0), (x0, y1), color, 2)
                draw_line(image, (x0, y1), (x1, y1), color, 2)
                draw_line(image, (x1, y1), (x1, y0), color, 2)
            else:
                x0, x1 = max(0, x0), min(width - 1, x1)
                y0, y1 = max(0, y0), min(height - 1, y1)
                if x1 >= x0 and y1 >= y0:
                    image[y0 : y1 + 1, x0 : x1 + 1] = color

        for render_index in range(render_frames):
            simulation_index = min(simulation_frames - 1, render_index * simulation_frames // render_frames)
            image = np.empty((height, width, 3), dtype=np.uint8)
            image[:] = (28, 31, 38)
            with np.load(frame_paths[simulation_index]) as state:
                for item in scene.objects.values():
                    if isinstance(item, (ObjectRigid, ObjectContainer)) and item.motion == "static":
                        draw_object(image, item, np.asarray(item.transform.position))
                if "body_q" in state:
                    for item, transform in zip(body_items, state["body_q"], strict=False):
                        draw_object(image, item, transform[:3])
                if "cloth_q" in state:
                    offset = 0
                    for item in scene.objects.values():
                        if not isinstance(item, ObjectCloth):
                            continue
                        count = item.resolution[0] * item.resolution[1]
                        positions = state["cloth_q"][offset : offset + count]
                        for row in range(item.resolution[1]):
                            for column in range(item.resolution[0]):
                                index = row * item.resolution[0] + column
                                if column + 1 < item.resolution[0]:
                                    draw_line(
                                        image, pixel(positions[index]), pixel(positions[index + 1]), (210, 110, 80)
                                    )
                                if row + 1 < item.resolution[1]:
                                    draw_line(
                                        image,
                                        pixel(positions[index]),
                                        pixel(positions[index + item.resolution[0]]),
                                        (210, 110, 80),
                                    )
                        offset += count
                for key in state.files:
                    if key.endswith("_particles"):
                        for position in state[key]:
                            draw_disc(image, pixel(position), 2, (45, 125, 220))
                    elif key.endswith("_density"):
                        density = state[key]
                        projection = density.max(axis=1).T
                        projection = np.flipud(projection)
                        ys = np.linspace(0, projection.shape[0] - 1, height).astype(int)
                        xs = np.linspace(0, projection.shape[1] - 1, width).astype(int)
                        alpha = np.clip(projection[np.ix_(ys, xs)], 0.0, 1.0)[..., None]
                        smoke = np.full_like(image, (180, 190, 205))
                        image = (image * (1.0 - alpha * 0.7) + smoke * alpha * 0.7).astype(np.uint8)
            path = output_dir / f"frame-{render_index:06d}.ppm"
            path.write_bytes(f"P6\n{width} {height}\n255\n".encode() + image.tobytes())

    @staticmethod
    def _render_cache_frames_gl(scene: Scene, frame_paths: list[Path], output_dir: Path) -> dict[str, Any]:
        """Replay cache through the hardware OpenGL viewer."""
        device = wp.get_device()
        viewer = None
        try:
            from newton.viewer import (  # noqa: PLC0415
                RendererFluidScreenSpace,
                RendererSmokeVolume,
                ViewerFluidGL,
            )

            compiled = SceneCompilerNewton().compile(scene)
            width, height = scene.render.resolution
            viewer = ViewerFluidGL(width=width, height=height, headless=True)
            from pyglet import gl

            def gl_string(name) -> str:
                value = gl.glGetString(name)
                return value.decode("utf-8", errors="replace") if value else "unknown"

            gl_vendor = gl_string(gl.GL_VENDOR)
            gl_renderer = gl_string(gl.GL_RENDERER)
            gl_version = gl_string(gl.GL_VERSION)
            software_markers = ("llvmpipe", "softpipe", "software rasterizer", "osmesa")
            if any(marker in gl_renderer.lower() for marker in software_markers):
                raise RuntimeError(f"Hardware OpenGL is required; detected software renderer: {gl_renderer}")
            viewer.set_model(compiled.model)
            camera = scene.render.camera
            if camera.position is not None and camera.target is not None:
                position = np.asarray(camera.position, dtype=float)
                direction = np.asarray(camera.target, dtype=float) - position
                direction /= max(np.linalg.norm(direction), 1.0e-8)
                yaw = math.degrees(math.atan2(direction[1], direction[0]))
                pitch = math.degrees(math.asin(np.clip(direction[2], -1.0, 1.0)))
                viewer.set_camera(wp.vec3(*position), pitch, yaw)
                viewer.camera.fov = camera.field_of_view
            smoke_renderers = {}
            liquid_renderers = {}
            for object_id, item in scene.objects.items():
                if not isinstance(item, ObjectFluid):
                    continue
                if item.phase == "smoke":
                    domain_size = np.asarray(item.size) * np.asarray(item.transform.scale)
                    domain_min = np.asarray(item.transform.position) - domain_size * 0.5
                    renderer = RendererSmokeVolume(
                        viewer,
                        world_min=tuple(domain_min),
                        world_max=tuple(domain_min + domain_size),
                        volume_resolution=item.grid_resolution,
                    )
                    viewer.register_post_render_callback(renderer.render)
                    smoke_renderers[object_id] = (renderer, tuple(domain_min), domain_size[0] / item.grid_resolution[0])
                else:
                    renderer = RendererFluidScreenSpace(
                        viewer,
                        max_particles=scene.settings.max_particles,
                        particle_radius=item.particle_spacing * 0.55,
                        device=device,
                    )
                    viewer.register_post_render_callback(renderer.render)
                    liquid_renderers[object_id] = renderer
            render_frames = max(1, round(scene.settings.duration * scene.render.fps))
            for render_index in range(render_frames):
                simulation_index = min(len(frame_paths) - 1, render_index * len(frame_paths) // render_frames)
                with np.load(frame_paths[simulation_index]) as cached:
                    if "body_q" in cached and compiled.state_0.body_q is not None:
                        SceneExecutorLocal._copy_array(compiled.state_0.body_q, cached["body_q"])
                        SceneExecutorLocal._copy_array(compiled.state_0.body_qd, cached["body_qd"])
                    if "cloth_q" in cached and compiled.state_0.particle_q is not None:
                        SceneExecutorLocal._copy_array(compiled.state_0.particle_q, cached["cloth_q"])
                        SceneExecutorLocal._copy_array(compiled.state_0.particle_qd, cached["cloth_qd"])
                    for object_id, (renderer, domain_min, cell_size) in smoke_renderers.items():
                        density = wp.array(
                            cached[f"{object_id}_density"],
                            dtype=float,
                            device=device,
                        )
                        renderer.set_dense_density(density, domain_min, cell_size)
                    for object_id, renderer in liquid_renderers.items():
                        particles = wp.array(
                            cached[f"{object_id}_particles"],
                            dtype=wp.vec3,
                            device=device,
                        )
                        renderer.set_particles(particles)
                    viewer.begin_frame(render_index / scene.render.fps)
                    viewer.log_state(compiled.state_0)
                    viewer.end_frame()
                    image = viewer.get_frame().numpy()
                    if not np.any(image):
                        raise RuntimeError(f"Framebuffer readback returned an empty image at frame {render_index}")
                    if any(not renderer.available for renderer in liquid_renderers.values()):
                        raise RuntimeError(f"Screen-space fluid renderer failed at frame {render_index}")
                    path = output_dir / f"frame-{render_index:06d}.ppm"
                    path.write_bytes(f"P6\n{width} {height}\n255\n".encode() + image.tobytes())
            return {
                "simulation_device": str(device),
                "render_backend": "opengl_hardware",
                "render_data_path": "cuda_interop" if device.is_cuda else "host_staging",
                "gl_vendor": gl_vendor,
                "gl_renderer": gl_renderer,
                "gl_version": gl_version,
                "render_equivalent": True,
            }
        except Exception as exc:
            raise RuntimeError(
                f"Hardware OpenGL rendering unavailable ({type(exc).__name__}: {exc}). "
                "Pass through a graphics device and matching GPU driver; software rendering is not supported."
            ) from exc
        finally:
            if viewer is not None:
                with suppress(Exception):
                    viewer.close()

    def run(
        self,
        scene: Scene,
        *,
        output_dir: Path,
        cancel_event: Event | None = None,
        progress: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a durable simulation, render, and reproducibility bundle.

        Simulation data is committed before rendering so an encoder failure can
        be retried without repeating the simulation. ``cancel_event`` is checked
        at every cached frame, which provides cooperative cancellation between
        native solver steps.

        Args:
            scene: Scene to execute.
            output_dir: Directory receiving the complete output bundle.
            cancel_event: Optional cooperative cancellation flag.
            progress: Optional mutable job progress mapping.

        Returns:
            Final job result including artifacts and diagnostics.
        """
        compile_started = time.perf_counter()
        compiled = self.compiler.compile(scene)
        compile_seconds = time.perf_counter() - compile_started
        output_dir = output_dir.resolve()
        cache_dir = output_dir / "cache"
        frames_dir = cache_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        scene_path = output_dir / "scene.json"
        program_path = output_dir / "program.py"
        metrics_path = output_dir / "metrics.json"
        diagnostics_path = output_dir / "diagnostics.jsonl"
        contacts_path = cache_dir / "contacts.jsonl"
        cache_manifest_path = cache_dir / "manifest.json"
        job_manifest_path = output_dir / "manifest.json"
        animation_path = output_dir / "animation.mp4"
        scene_payload = scene.to_dict()
        scene_bytes = json.dumps(scene_payload, sort_keys=True, separators=(",", ":")).encode()
        scene_hash = hashlib.sha256(scene_bytes).hexdigest()
        artifacts = {
            "animation": str(animation_path),
            "scene": str(scene_path),
            "program": str(program_path),
            "metrics": str(metrics_path),
            "diagnostics": str(diagnostics_path),
            "cache": str(cache_dir),
        }

        def update(stage: str, frame: int = 0, status: str = "running") -> None:
            values = {
                "status": status,
                "stage": stage,
                "frame": frame,
                "total_frames": round(scene.settings.duration * scene.settings.fps),
                "scene": scene.name,
                "scene_hash": scene_hash,
                "output_dir": str(output_dir),
                "artifacts": artifacts,
            }
            if progress is not None:
                progress.update(values)
            job_manifest_path.write_text(json.dumps(values, indent=2) + "\n", encoding="utf-8")

        update("compile")
        scene_path.write_text(json.dumps(scene_payload, indent=2) + "\n", encoding="utf-8")
        self.compiler.export_program(scene, program_path)
        diagnostics_path.write_text("", encoding="utf-8")
        contacts_path.write_text("", encoding="utf-8")
        cache_manifest_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "scene_hash": scene_hash,
                    "complete": False,
                    "frames": 0,
                    "fps": scene.settings.fps,
                    "frame_pattern": "frames/%06d.npz",
                    "contacts": contacts_path.name,
                    "metrics": "../metrics.json",
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

        def cache_frame(frame_index: int, frame_state: dict[str, np.ndarray]) -> None:
            np.savez_compressed(frames_dir / f"{frame_index:06d}.npz", **frame_state)
            update("simulation", frame_index + 1)

        simulation_started = time.perf_counter()
        try:
            simulation = self.simulate(
                scene,
                capture_cache=True,
                cancel_event=cancel_event,
                progress=progress,
                frame_callback=cache_frame,
                compiled_scene=compiled,
            )
        except _SimulationCancelled:
            update("simulation", int((progress or {}).get("frame", 0)), "cancelled")
            return {
                "status": "cancelled",
                "stage": "simulation",
                "scene": scene.name,
                "output_dir": str(output_dir),
                "artifacts": artifacts,
            }
        simulation.pop("trajectories")
        simulation.pop("state_frames")
        contact_records = simulation.pop("contact_records")
        frame_count = simulation["frames"]
        for frame_index in range(frame_count):
            if cancel_event is not None and cancel_event.is_set():
                update("simulation", frame_index, "cancelled")
                return {"status": "cancelled", "scene": scene.name, "artifacts": artifacts}
        simulation_seconds = time.perf_counter() - simulation_started
        device = wp.get_device()
        peak_device_memory = 0
        if device.is_cuda:
            try:
                peak_device_memory = int(wp.get_mempool_used_mem_high(device))
            except RuntimeError:
                peak_device_memory = 0
        diagnostics_path.write_text(
            "".join(json.dumps(item, sort_keys=True) + "\n" for item in simulation["diagnostics"]),
            encoding="utf-8",
        )
        contacts_path.write_text(
            "".join(json.dumps(item, sort_keys=True) + "\n" for item in contact_records),
            encoding="utf-8",
        )
        metrics = {
            **simulation["metrics"],
            "scene_hash": scene_hash,
            "output_dir": str(output_dir),
            "timings": {
                "compile_seconds": compile_seconds,
                "simulation_seconds": simulation_seconds,
            },
            "peak_device_memory_bytes": peak_device_memory,
        }
        metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        cache_manifest = {
            "schema_version": 1,
            "scene_hash": scene_hash,
            "complete": True,
            "physics_valid": simulation["metrics"]["physics"]["valid"],
            "frames": frame_count,
            "fps": scene.settings.fps,
            "frame_pattern": "frames/%06d.npz",
            "contacts": contacts_path.name,
            "metrics": "../metrics.json",
            "simulation_device": str(device),
        }
        cache_manifest_path.write_text(json.dumps(cache_manifest, indent=2) + "\n", encoding="utf-8")

        if cancel_event is not None and cancel_event.is_set():
            update("render", frame_count, "cancelled")
            return {"status": "cancelled", "scene": scene.name, "artifacts": artifacts}
        update("render", frame_count)
        render_started = time.perf_counter()
        rendered = self._encode_video(scene, animation_path, cache_dir)
        for key in (
            "simulation_device",
            "render_backend",
            "render_data_path",
            "gl_vendor",
            "gl_renderer",
            "gl_version",
            "render_equivalent",
        ):
            if key in rendered:
                metrics[key] = rendered[key]
        metrics["timings"]["render_seconds"] = time.perf_counter() - render_started
        metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        render_diagnostics = rendered.get("diagnostics", [])
        if render_diagnostics:
            with diagnostics_path.open("a", encoding="utf-8") as stream:
                for diagnostic in render_diagnostics:
                    stream.write(json.dumps(diagnostic, sort_keys=True) + "\n")
        render_status = rendered["status"]
        physics_valid = simulation["metrics"]["physics"]["valid"]
        status = render_status if render_status != "completed" else ("completed" if physics_valid else "physics_failed")
        stage = (
            "render_failed"
            if render_status != "completed"
            else ("completed" if physics_valid else "physics_validation")
        )
        update(stage, frame_count, status)
        return {
            "status": status,
            "stage": stage,
            "scene": scene.name,
            "scene_hash": scene_hash,
            "frames": frame_count,
            "artifacts": artifacts,
            "metrics": metrics,
            "diagnostics": [*simulation["diagnostics"], *render_diagnostics],
        }

    def resume_render(self, scene: Scene, *, output_dir: Path) -> dict[str, Any]:
        """Resume only the render stage of a job with a complete cache."""
        output_dir = output_dir.resolve()
        cache_manifest = output_dir / "cache" / "manifest.json"
        cache_metadata = json.loads(cache_manifest.read_text()) if cache_manifest.is_file() else {}
        if not cache_metadata.get("complete", False):
            raise ValueError("A complete simulation cache is required to resume rendering")
        rendered = self._encode_video(scene, output_dir / "animation.mp4", output_dir / "cache")
        physics_valid = cache_metadata.get("physics_valid", True)
        status = (
            rendered["status"]
            if rendered["status"] != "completed"
            else ("completed" if physics_valid else "physics_failed")
        )
        return {
            **rendered,
            "status": status,
            "scene": scene.name,
            "output_dir": str(output_dir),
            "stage": "render_failed"
            if rendered["status"] != "completed"
            else ("completed" if physics_valid else "physics_validation"),
            "artifacts": {
                "animation": str(output_dir / "animation.mp4"),
                "scene": str(output_dir / "scene.json"),
                "program": str(output_dir / "program.py"),
                "metrics": str(output_dir / "metrics.json"),
                "diagnostics": str(output_dir / "diagnostics.jsonl"),
                "cache": str(output_dir / "cache"),
            },
        }

    def export(self, scene: Scene, *, output: Path, format: str) -> dict[str, Any]:
        """Export scene JSON or a reproducible Python program."""
        output = output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        if format == "json":
            output.write_text(json.dumps(scene.to_dict(), indent=2) + "\n", encoding="utf-8")
        elif format == "python":
            self.compiler.export_program(scene, output)
        else:
            raise ValueError("format must be 'json' or 'python'")
        return {"status": "completed", "scene": scene.name, "output": str(output), "format": format}

    def validate(self, scene: Scene) -> dict[str, Any]:
        """Return structured static validation without running a simulation."""
        return validate_scene(scene)
