# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Public fluid solver implementations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

from .ca_exercises.solver_exercise5_fluid import SolverExercise5Fluid, apply_density_source_kernel
from .solver import SolverBase


def _aggregate_cloth_angular_impulses(boundaries, positions, vertex_impulses):
    """Aggregate distributed cloth impulses about each cloth centroid [N m s]."""
    result = {}
    for boundary in boundaries:
        vertices = np.unique(np.asarray(boundary.triangles, dtype=np.int32).reshape(-1))
        if not len(vertices):
            result[boundary.object_id] = np.zeros(3, dtype=np.float32)
            continue
        center = positions[vertices].mean(axis=0)
        result[boundary.object_id] = (
            np.cross(positions[vertices] - center, vertex_impulses[vertices]).sum(axis=0).astype(np.float32)
        )
    return result


@wp.kernel
def _scale_density(field: wp.array3d(dtype=float), factor: float):
    i, j, k = wp.tid()
    field[i, j, k] = field[i, j, k] * factor


@wp.kernel
def _add_emitter_velocity(
    u: wp.array3d(dtype=float),
    v: wp.array3d(dtype=float),
    w: wp.array3d(dtype=float),
    source_min: wp.vec3i,
    source_max: wp.vec3i,
    velocity: wp.vec3,
):
    i, j, k = wp.tid()
    if (
        i >= source_min[0]
        and i < source_max[0]
        and j >= source_min[1]
        and j < source_max[1]
        and k >= source_min[2]
        and k < source_max[2]
    ):
        u[i, j, k] = velocity[0]
        v[i, j, k] = velocity[1]
        w[i, j, k] = velocity[2]


class SolverFluidSmoke(SolverExercise5Fluid):
    """Incompressible Eulerian smoke solver on a staggered MAC grid.

    Args:
        model: Newton model containing rigid obstacles.
        res: Grid cell counts.
        domain_size: Physical domain size [m].
        pressure_iters: Pressure projection iteration count.
        buoyancy: Density buoyancy acceleration scale [m/s^2].
        wind_on: Whether to apply the built-in wind field.
        wind_strength: Wind acceleration [m/s^2].
        source_box: Source minimum and maximum coordinates [m].
        obstacle_box: Solid minimum and maximum coordinates [m], or ``None``.
        obstacle_on: Whether the solid obstacle is active.
    """

    @dataclass(slots=True)
    class Emitter:
        """Timed smoke source in world coordinates.

        Attributes:
            position: Source center [m].
            size: Source dimensions [m].
            start_time: Activation start [s].
            end_time: Activation end [s].
            density: Source density.
            velocity: Initial source velocity [m/s].
        """

        position: tuple[float, float, float]
        size: tuple[float, float, float]
        start_time: float = 0.0
        end_time: float = 1.0
        density: float = 1.0
        velocity: tuple[float, float, float] = (0.0, 0.0, 0.0)

    @dataclass(slots=True)
    class Boundary:
        """Axis-aligned smoke obstacle."""

        position: tuple[float, float, float]
        half_extent: tuple[float, float, float]
        body: int | None = None
        local_position: tuple[float, float, float] | None = None

    @dataclass(slots=True)
    class ClothBoundary:
        """Triangulated moving smoke obstacle."""

        object_id: str
        triangles: np.ndarray

    def __init__(
        self,
        model,
        res: tuple[int, int, int] = (48, 48, 72),
        domain_size: tuple[float, float, float] = (1.0, 1.0, 1.5),
        domain_min: tuple[float, float, float] = (0.0, 0.0, 0.0),
        pressure_iters: int = 25,
        buoyancy: float = 0.1,
        dissipation: float = 0.01,
        emitters: list[Emitter] | None = None,
        drag_density: float = 1.225,
        drag_coefficient: float = 0.0,
        coupling_iterations: int = 1,
        coupling_relaxation: float = 1.0,
    ):
        placeholder = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        super().__init__(
            model,
            res=res,
            domain_size=domain_size,
            pressure_iters=pressure_iters,
            buoyancy=buoyancy,
            source_box=placeholder,
            obstacle_box=None,
            obstacle_on=False,
        )
        if not 0.0 <= dissipation:
            raise ValueError("dissipation must be non-negative")
        self.grid_min = tuple(float(value) for value in domain_min)
        self.grid_max = tuple(self.grid_min[axis] + domain_size[axis] for axis in range(3))
        self.dissipation = float(dissipation)
        self.emitters = list(emitters or [])
        self.boundaries: list[SolverFluidSmoke.Boundary] = []
        self.cloth_boundaries: list[SolverFluidSmoke.ClothBoundary] = []
        self.drag_density = drag_density
        self.drag_coefficient = drag_coefficient
        if coupling_iterations <= 0 or not 0.0 < coupling_relaxation <= 1.0:
            raise ValueError("Smoke coupling iterations must be positive and relaxation must be in (0, 1]")
        self.coupling_iterations = coupling_iterations
        self.coupling_relaxation = coupling_relaxation
        shape = (self.nx, self.ny, self.nz)
        self._solid_cloth = np.full(shape, -1, dtype=np.int32)
        self._solid_cloth_vertices = np.full((*shape, 3), -1, dtype=np.int32)
        self._solid_cloth_weights = np.zeros((*shape, 3), dtype=np.float32)
        self.cloth_linear_impulse: dict[str, np.ndarray] = {}
        self.fluid_cloth_linear_impulse: dict[str, np.ndarray] = {}
        self.cloth_angular_impulse: dict[str, np.ndarray] = {}
        self.fluid_cloth_angular_impulse: dict[str, np.ndarray] = {}
        self.cloth_penetration_count: dict[str, int] = {}
        self.cloth_density_penetration: dict[str, float] = {}
        self._solid_velocity = np.zeros((self.nx, self.ny, self.nz, 3), dtype=np.float32)
        self._solid_body = np.full(shape, -1, dtype=np.int32)
        self.rigid_linear_impulse: dict[int, np.ndarray] = {}
        self.rigid_angular_impulse: dict[int, np.ndarray] = {}
        self.rigid_pressure_impulse: dict[int, np.ndarray] = {}
        self.rigid_stabilization_impulse: dict[int, np.ndarray] = {}
        self.impulse_clip_count = 0
        self.time = 0.0

    def add_boundary(self, boundary: Boundary) -> None:
        """Register a static, kinematic, or dynamic smoke obstacle."""
        self.boundaries.append(boundary)

    def add_cloth_boundary(self, boundary: ClothBoundary) -> None:
        """Register cloth triangles as a moving smoke obstacle."""
        self.cloth_boundaries.append(boundary)

    def _update_boundaries(self, state_in) -> None:
        solid = np.zeros((self.nx, self.ny, self.nz), dtype=np.int32)
        self._solid_velocity.fill(0.0)
        self._solid_body.fill(-1)
        self._solid_cloth.fill(-1)
        self._solid_cloth_vertices.fill(-1)
        self._solid_cloth_weights.fill(0.0)
        body_q = state_in.body_q.numpy() if state_in is not None and state_in.body_q is not None else None
        body_qd = state_in.body_qd.numpy() if state_in is not None and state_in.body_qd is not None else None
        coordinates = np.stack(
            np.meshgrid(
                self.grid_min[0] + (np.arange(self.nx) + 0.5) * self.size_x / self.nx,
                self.grid_min[1] + (np.arange(self.ny) + 0.5) * self.size_y / self.ny,
                self.grid_min[2] + (np.arange(self.nz) + 0.5) * self.size_z / self.nz,
                indexing="ij",
            ),
            axis=-1,
        )
        for boundary in self.boundaries:
            position = np.asarray(boundary.position)
            if boundary.body is not None and body_q is not None:
                position = body_q[boundary.body, :3]
            mask = np.all(np.abs(coordinates - position) <= np.asarray(boundary.half_extent), axis=-1)
            solid[mask] = 1
            if boundary.body is not None and body_qd is not None:
                self._solid_body[mask] = boundary.body
                linear = body_qd[boundary.body, :3]
                angular = body_qd[boundary.body, 3:]
                self._solid_velocity[mask] = (linear + np.cross(angular, coordinates - position))[mask]
        if self.cloth_boundaries and state_in is not None and state_in.particle_q is not None:
            particle_q = state_in.particle_q.numpy()
            particle_qd = state_in.particle_qd.numpy()
            cell_size = np.asarray((self.size_x / self.nx, self.size_y / self.ny, self.size_z / self.nz))
            resolution = np.asarray((self.nx, self.ny, self.nz))
            for cloth_index, boundary in enumerate(self.cloth_boundaries):
                for triangle in boundary.triangles:
                    vertices = particle_q[triangle]
                    edge_length = max(
                        float(np.linalg.norm(vertices[1] - vertices[0])),
                        float(np.linalg.norm(vertices[2] - vertices[1])),
                        float(np.linalg.norm(vertices[0] - vertices[2])),
                    )
                    divisions = min(64, max(1, int(np.ceil(edge_length / (np.min(cell_size) * 0.5)))))
                    for first in range(divisions + 1):
                        for second in range(divisions + 1 - first):
                            weights = np.asarray(
                                [first / divisions, second / divisions, 1.0 - (first + second) / divisions],
                                dtype=np.float32,
                            )
                            sample = weights @ vertices
                            cell = np.floor((sample - np.asarray(self.grid_min)) / cell_size).astype(int)
                            if np.any(cell < 0) or np.any(cell >= resolution):
                                continue
                            key = tuple(cell)
                            solid[key] = 1
                            self._solid_cloth[key] = cloth_index
                            self._solid_cloth_vertices[key] = triangle
                            self._solid_cloth_weights[key] = weights
                            self._solid_velocity[key] = weights @ particle_qd[triangle]
        wp.copy(self.solid, wp.array(solid, dtype=wp.int32, device=self.model.device))

    def _enforce_solid_velocity(self) -> None:
        super()._enforce_solid_velocity()
        solid = self.solid.numpy().astype(bool)
        if not np.any(solid):
            return
        i, j, k = np.nonzero(solid)
        u, v, w = self.u.numpy(), self.v.numpy(), self.w.numpy()
        u[i, j, k] = self._solid_velocity[i, j, k, 0]
        u[i + 1, j, k] = self._solid_velocity[i, j, k, 0]
        v[i, j, k] = self._solid_velocity[i, j, k, 1]
        v[i, j + 1, k] = self._solid_velocity[i, j, k, 1]
        w[i, j, k] = self._solid_velocity[i, j, k, 2]
        w[i, j, k + 1] = self._solid_velocity[i, j, k, 2]
        wp.copy(self.u, wp.array(u, dtype=float, device=self.model.device))
        wp.copy(self.v, wp.array(v, dtype=float, device=self.model.device))
        wp.copy(self.w, wp.array(w, dtype=float, device=self.model.device))

    def _world_to_cell(self, point: tuple[float, float, float]) -> tuple[int, int, int]:
        return tuple(
            int(
                round(
                    (point[axis] - self.grid_min[axis])
                    * (self.nx, self.ny, self.nz)[axis]
                    / (self.size_x, self.size_y, self.size_z)[axis]
                )
            )
            for axis in range(3)
        )

    def _apply_density_source(self) -> None:
        for emitter in self.emitters:
            if not emitter.start_time <= self.time < emitter.end_time:
                continue
            lo_world = tuple(emitter.position[axis] - emitter.size[axis] * 0.5 for axis in range(3))
            hi_world = tuple(emitter.position[axis] + emitter.size[axis] * 0.5 for axis in range(3))
            lo = self._clamp_cell(self._world_to_cell(lo_world))
            hi = self._clamp_cell(self._world_to_cell(hi_world))
            wp.launch(
                apply_density_source_kernel,
                dim=(self.nx, self.ny, self.nz),
                inputs=[self.density, self.solid, wp.vec3i(*lo), wp.vec3i(*hi)],
                device=self.model.device,
            )
            if emitter.density != 1.0:
                density = self.density.numpy()
                density[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]] = emitter.density
                wp.copy(self.density, wp.array(density, dtype=float, device=self.model.device))
            wp.launch(
                _add_emitter_velocity,
                dim=(self.nx, self.ny, self.nz),
                inputs=[self.u, self.v, self.w, wp.vec3i(*lo), wp.vec3i(*hi), wp.vec3(*emitter.velocity)],
                device=self.model.device,
            )

    def step(self, state_in, state_out, control, contacts, dt):
        """Advance smoke advection and pressure projection by ``dt`` [s]."""
        self._update_boundaries(state_in)
        result = super().step(state_in, state_out, control, contacts, dt)
        density = self.density.numpy()
        density_total = max(float(density.sum()), 1.0e-12)
        self.cloth_density_penetration = {}
        self.cloth_penetration_count = {}
        for cloth_index, boundary in enumerate(self.cloth_boundaries):
            mask = self._solid_cloth == cloth_index
            values = density[mask]
            self.cloth_density_penetration[boundary.object_id] = float(values.sum()) / density_total
            self.cloth_penetration_count[boundary.object_id] = int(np.count_nonzero(values > 1.0e-5))
        coupling_dt = dt / self.coupling_iterations * self.coupling_relaxation
        for iteration in range(self.coupling_iterations):
            self._apply_rigid_drag(state_out, coupling_dt, reset=iteration == 0)
            self._apply_cloth_drag(state_out, coupling_dt, reset=iteration == 0)
        if self.dissipation:
            wp.launch(
                _scale_density,
                dim=(self.nx, self.ny, self.nz),
                inputs=[self.density, max(0.0, 1.0 - self.dissipation * dt)],
                device=self.model.device,
            )
        self.time += dt
        return result

    def _apply_rigid_drag(self, state_out, dt: float, *, reset: bool = True) -> None:
        """Apply optional smoke drag to dynamic rigid obstacles."""
        if reset:
            self.rigid_linear_impulse.clear()
            self.rigid_angular_impulse.clear()
            self.rigid_pressure_impulse.clear()
            self.rigid_stabilization_impulse.clear()
            self.impulse_clip_count = 0
        if (
            self.drag_coefficient <= 0.0
            or state_out is None
            or state_out.body_qd is None
            or not np.any(self._solid_body >= 0)
        ):
            return
        u, v, w = self.u.numpy(), self.v.numpy(), self.w.numpy()
        cell_velocity = np.stack(
            (
                0.5 * (u[:-1] + u[1:]),
                0.5 * (v[:, :-1] + v[:, 1:]),
                0.5 * (w[:, :, :-1] + w[:, :, 1:]),
            ),
            axis=-1,
        )
        velocities = state_out.body_qd.numpy()
        transforms = state_out.body_q.numpy()
        masses = self.model.body_mass.numpy()
        inverse_inertia = self.model.body_inv_inertia.numpy()
        cell_size = np.asarray((self.size_x / self.nx, self.size_y / self.ny, self.size_z / self.nz))
        area = float(np.min(cell_size) ** 2)
        volume = float(np.prod(cell_size))
        centers = np.stack(
            np.meshgrid(
                self.grid_min[0] + (np.arange(self.nx) + 0.5) * cell_size[0],
                self.grid_min[1] + (np.arange(self.ny) + 0.5) * cell_size[1],
                self.grid_min[2] + (np.arange(self.nz) + 0.5) * cell_size[2],
                indexing="ij",
            ),
            axis=-1,
        )
        for body_value in np.unique(self._solid_body[self._solid_body >= 0]):
            body = int(body_value)
            if masses[body] <= 0.0:
                continue
            mask = self._solid_body == body
            relative = cell_velocity[mask] - self._solid_velocity[mask]
            speed = np.linalg.norm(relative, axis=1)
            cell_impulses = 0.5 * self.drag_density * self.drag_coefficient * area * speed[:, None] * relative * dt
            impulse = cell_impulses.sum(axis=0)
            angular_impulse = np.cross(centers[mask] - transforms[body, :3], cell_impulses).sum(axis=0)
            maximum = masses[body] * 2.0 * dt
            norm = float(np.linalg.norm(impulse))
            if norm > maximum > 0.0:
                scale = maximum / norm
                impulse *= scale
                angular_impulse *= scale
                cell_impulses *= scale
                self.impulse_clip_count += 1
            velocities[body, :3] += impulse / masses[body]
            if inverse_inertia.size:
                velocities[body, 3:] += inverse_inertia[body] @ angular_impulse
            self.rigid_linear_impulse[body] = np.asarray(
                self.rigid_linear_impulse.get(body, np.zeros(3)) + impulse, dtype=np.float32
            )
            self.rigid_angular_impulse[body] = np.asarray(
                self.rigid_angular_impulse.get(body, np.zeros(3)) + angular_impulse,
                dtype=np.float32,
            )
            self.rigid_pressure_impulse[body] = np.zeros(3, dtype=np.float32)
            self.rigid_stabilization_impulse[body] = self.rigid_linear_impulse[body].copy()
            fluid_delta = -cell_impulses / max(self.drag_density * volume, 1.0e-12)
            for cell, delta in zip(np.argwhere(mask), fluid_delta, strict=True):
                i, j, k = cell
                u[i, j, k] += delta[0] * 0.5
                u[i + 1, j, k] += delta[0] * 0.5
                v[i, j, k] += delta[1] * 0.5
                v[i, j + 1, k] += delta[1] * 0.5
                w[i, j, k] += delta[2] * 0.5
                w[i, j, k + 1] += delta[2] * 0.5
        wp.copy(self.u, wp.array(u, dtype=float, device=self.model.device))
        wp.copy(self.v, wp.array(v, dtype=float, device=self.model.device))
        wp.copy(self.w, wp.array(w, dtype=float, device=self.model.device))
        wp.copy(
            state_out.body_qd,
            wp.array(velocities, dtype=state_out.body_qd.dtype, device=state_out.body_qd.device),
        )

    def _apply_cloth_drag(self, state_out, dt: float, *, reset: bool = True) -> None:
        if reset:
            self.cloth_linear_impulse.clear()
            self.fluid_cloth_linear_impulse.clear()
            self.cloth_angular_impulse.clear()
            self.fluid_cloth_angular_impulse.clear()
        if (
            not self.cloth_boundaries
            or self.drag_coefficient <= 0.0
            or state_out is None
            or state_out.particle_qd is None
        ):
            return
        u, v, w = self.u.numpy(), self.v.numpy(), self.w.numpy()
        cell_velocity = np.stack(
            (
                0.5 * (u[:-1] + u[1:]),
                0.5 * (v[:, :-1] + v[:, 1:]),
                0.5 * (w[:, :, :-1] + w[:, :, 1:]),
            ),
            axis=-1,
        )
        particle_velocity = state_out.particle_qd.numpy()
        inverse_mass = self.model.particle_inv_mass.numpy()
        vertex_impulses = np.zeros_like(particle_velocity, dtype=np.float64)
        cell_size = np.asarray((self.size_x / self.nx, self.size_y / self.ny, self.size_z / self.nz))
        area = float(np.min(cell_size) ** 2)
        volume = float(np.prod(cell_size))
        for cell in np.argwhere(self._solid_cloth >= 0):
            key = tuple(cell)
            cloth_index = int(self._solid_cloth[key])
            boundary = self.cloth_boundaries[cloth_index]
            vertices = self._solid_cloth_vertices[key]
            weights = self._solid_cloth_weights[key]
            cloth_velocity = weights @ particle_velocity[vertices]
            relative = cell_velocity[key] - cloth_velocity
            speed = float(np.linalg.norm(relative))
            impulse = 0.5 * self.drag_density * self.drag_coefficient * area * speed * relative * dt
            movable_mass = np.divide(
                1.0,
                inverse_mass[vertices],
                out=np.zeros(3),
                where=inverse_mass[vertices] > 0.0,
            ).sum()
            impulse_norm = float(np.linalg.norm(impulse))
            maximum_impulse = movable_mass * 0.25
            if impulse_norm > maximum_impulse > 0.0:
                impulse *= maximum_impulse / impulse_norm
            for vertex, weight in zip(vertices, weights, strict=True):
                vertex_impulses[vertex] += impulse * float(weight)
            self.cloth_linear_impulse[boundary.object_id] = (
                self.cloth_linear_impulse.get(boundary.object_id, np.zeros(3)) + impulse
            )
            fluid_delta = -impulse / max(self.drag_density * volume, 1.0e-12)
            i, j, k = key
            u[i, j, k] += fluid_delta[0] * 0.5
            u[i + 1, j, k] += fluid_delta[0] * 0.5
            v[i, j, k] += fluid_delta[1] * 0.5
            v[i, j + 1, k] += fluid_delta[1] * 0.5
            w[i, j, k] += fluid_delta[2] * 0.5
            w[i, j, k + 1] += fluid_delta[2] * 0.5
        for vertex in np.flatnonzero(np.linalg.norm(vertex_impulses, axis=1) > 0.0):
            particle_velocity[vertex] += vertex_impulses[vertex] * inverse_mass[vertex]
        angular_impulses = _aggregate_cloth_angular_impulses(
            self.cloth_boundaries, state_out.particle_q.numpy(), vertex_impulses
        )
        for object_id, impulse in angular_impulses.items():
            self.cloth_angular_impulse[object_id] = np.asarray(
                self.cloth_angular_impulse.get(object_id, np.zeros(3)) + impulse,
                dtype=np.float32,
            )
        self.fluid_cloth_angular_impulse = {
            object_id: -impulse for object_id, impulse in self.cloth_angular_impulse.items()
        }
        for object_id, impulse in self.cloth_linear_impulse.items():
            self.cloth_linear_impulse[object_id] = np.asarray(impulse, dtype=np.float32)
            self.fluid_cloth_linear_impulse[object_id] = -np.asarray(impulse, dtype=np.float32)
        wp.copy(self.u, wp.array(u, dtype=float, device=self.model.device))
        wp.copy(self.v, wp.array(v, dtype=float, device=self.model.device))
        wp.copy(self.w, wp.array(w, dtype=float, device=self.model.device))
        wp.copy(
            state_out.particle_qd,
            wp.array(particle_velocity, dtype=state_out.particle_qd.dtype, device=state_out.particle_qd.device),
        )


class SolverFluidAPIC(SolverBase):
    """Fixed-capacity APIC/FLIP liquid solver for scene pipelines.

    Particles transfer mass, momentum, and their affine velocity matrix to a
    regular pressure grid. A red-black pressure solve projects the grid before
    PIC/FLIP transfer back to particles. The implementation intentionally owns
    its particle pool instead of treating Newton contact particles as liquid.

    Args:
        model: Newton model containing liquid particles and rigid colliders.
        grid_resolution: Pressure-grid cell counts.
        domain_size: Physical pressure-grid size [m].
        flip_ratio: FLIP contribution in the APIC/FLIP blend.
        pressure_iters: Pressure projection iteration count.
        domain_min: Pressure-domain minimum [m].
        max_particles: Fixed particle-pool capacity.
        density: Liquid mass density [kg/m^3].
        viscosity: Kinematic viscosity [m^2/s].
        surface_tension: Surface tension [N/m].
    """

    @dataclass(slots=True)
    class Emitter:
        """Timed liquid particle source.

        Attributes:
            position: Source center [m].
            size: Source dimensions [m].
            start_time: Activation start [s].
            end_time: Activation end [s].
            velocity: Initial velocity [m/s].
            spacing: Particle spacing [m].
        """

        position: tuple[float, float, float]
        size: tuple[float, float, float]
        start_time: float
        end_time: float
        velocity: tuple[float, float, float]
        spacing: float

    @dataclass(slots=True)
    class Boundary:
        """Box solid coupled to the liquid grid."""

        position: tuple[float, float, float]
        half_extent: tuple[float, float, float]
        body: int | None = None
        local_position: tuple[float, float, float] | None = None
        rotation: tuple[float, float, float, float] | None = None

    @dataclass(slots=True)
    class ClothBoundary:
        """Triangulated cloth coupled as a moving liquid boundary."""

        object_id: str
        triangles: np.ndarray

    def __init__(
        self,
        model,
        grid_resolution: tuple[int, int, int] = (32, 32, 32),
        domain_size: tuple[float, float, float] = (1.0, 1.0, 1.0),
        flip_ratio: float = 0.95,
        pressure_iters: int = 40,
        domain_min: tuple[float, float, float] = (0.0, 0.0, 0.0),
        max_particles: int = 250_000,
        density: float = 1000.0,
        viscosity: float = 0.001,
        surface_tension: float = 0.072,
        coupling_iterations: int = 1,
        coupling_relaxation: float = 1.0,
        cfl_number: float = 0.5,
        cloth_drag: float = 1.0,
        cloth_permeability: float = 0.0,
        boundary_friction: float = 0.0,
    ):
        if any(value < 2 for value in grid_resolution):
            raise ValueError("grid_resolution axes must contain at least two cells")
        if any(value <= 0.0 for value in domain_size):
            raise ValueError("domain_size components must be positive")
        if not 0.0 <= flip_ratio <= 1.0:
            raise ValueError("flip_ratio must be between zero and one")
        if pressure_iters <= 0:
            raise ValueError("pressure_iters must be positive")
        if max_particles <= 0:
            raise ValueError("max_particles must be positive")
        if not 1 <= coupling_iterations <= 4:
            raise ValueError("coupling_iterations must be between one and four")
        if not 0.0 < coupling_relaxation <= 1.0:
            raise ValueError("coupling_relaxation must be in (0, 1]")
        if cfl_number <= 0.0:
            raise ValueError("cfl_number must be positive")
        if cloth_drag < 0.0 or not 0.0 <= cloth_permeability <= 1.0 or boundary_friction < 0.0:
            raise ValueError("Invalid cloth-fluid boundary coefficients")
        super().__init__(model)
        self.grid_resolution = tuple(int(value) for value in grid_resolution)
        self.domain_size = np.asarray(domain_size, dtype=np.float64)
        self.domain_min = np.asarray(domain_min, dtype=np.float64)
        self.domain_max = self.domain_min + self.domain_size
        self.cell_size = self.domain_size / np.asarray(self.grid_resolution)
        self.flip_ratio = flip_ratio
        self.pressure_iters = pressure_iters
        self.max_particles = max_particles
        self.density = density
        self.viscosity = viscosity
        self.surface_tension = surface_tension
        self.coupling_iterations = coupling_iterations
        self.coupling_relaxation = coupling_relaxation
        self.cfl_number = cfl_number
        self.cloth_drag = cloth_drag
        self.cloth_permeability = cloth_permeability
        self.boundary_friction = boundary_friction
        self.particle_count = 0
        self.particle_position = np.zeros((max_particles, 3), dtype=np.float32)
        self.particle_velocity = np.zeros((max_particles, 3), dtype=np.float32)
        self.particle_affine = np.zeros((max_particles, 3, 3), dtype=np.float32)
        self.particle_mass = np.zeros(max_particles, dtype=np.float32)
        shape = self.grid_resolution
        self.grid_mass = np.zeros(shape, dtype=np.float32)
        self.grid_velocity = np.zeros((*shape, 3), dtype=np.float32)
        self.grid_velocity_old = np.zeros_like(self.grid_velocity)
        self.pressure = np.zeros(shape, dtype=np.float32)
        self.divergence = np.zeros(shape, dtype=np.float32)
        self.fluid = np.zeros(shape, dtype=bool)
        self.solid = np.zeros(shape, dtype=bool)
        self.solid_body = np.full(shape, -1, dtype=np.int32)
        self.solid_velocity = np.zeros((*shape, 3), dtype=np.float32)
        self.emitters: list[SolverFluidAPIC.Emitter] = []
        self.boundaries: list[SolverFluidAPIC.Boundary] = []
        self.cloth_boundaries: list[SolverFluidAPIC.ClothBoundary] = []
        self.solid_cloth = np.full(shape, -1, dtype=np.int32)
        self.solid_cloth_vertices = np.full((*shape, 3), -1, dtype=np.int32)
        self.solid_cloth_weights = np.zeros((*shape, 3), dtype=np.float32)
        self.cloth_linear_impulse: dict[str, np.ndarray] = {}
        self.fluid_cloth_linear_impulse: dict[str, np.ndarray] = {}
        self.cloth_angular_impulse: dict[str, np.ndarray] = {}
        self.fluid_cloth_angular_impulse: dict[str, np.ndarray] = {}
        self.cloth_penetration_count: dict[str, int] = {}
        self.capacity_overflow = False
        self.rigid_linear_impulse: dict[int, np.ndarray] = {}
        self.rigid_angular_impulse: dict[int, np.ndarray] = {}
        self.rigid_pressure_impulse: dict[int, np.ndarray] = {}
        self.rigid_stabilization_impulse: dict[int, np.ndarray] = {}
        self.impulse_clip_count = 0
        self._last_emission_step: dict[int, int] = {}
        self._step_count = 0
        self._nominal_particle_mass = 0.0
        self.time = 0.0

    def initialize_box(
        self,
        minimum: tuple[float, float, float],
        maximum: tuple[float, float, float],
        spacing: float,
        velocity: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ) -> int:
        """Seed a regular particle volume and return the number added.

        Args:
            minimum: Volume minimum [m].
            maximum: Volume maximum [m].
            spacing: Particle spacing [m].
            velocity: Initial velocity [m/s].

        Returns:
            Number of particles added to the fixed-capacity pool.
        """
        axes = [np.arange(minimum[axis] + spacing * 0.5, maximum[axis], spacing) for axis in range(3)]
        if any(len(axis) == 0 for axis in axes):
            return 0
        points = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
        available = self.max_particles - self.particle_count
        if len(points) > available:
            raise OverflowError(f"Liquid particle capacity exceeded: need {len(points)}, have {available}")
        start, end = self.particle_count, self.particle_count + len(points)
        self.particle_position[start:end] = points
        self.particle_velocity[start:end] = velocity
        particle_mass = self.density * spacing**3
        self.particle_mass[start:end] = particle_mass
        if self._nominal_particle_mass == 0.0:
            self._nominal_particle_mass = particle_mass
        self.particle_count = end
        return len(points)

    def add_emitter(self, emitter: Emitter) -> None:
        """Register a timed liquid emitter."""
        self.emitters.append(emitter)

    def add_boundary(self, boundary: Boundary) -> None:
        """Register an axis-aligned solid boundary."""
        self.boundaries.append(boundary)

    def add_cloth_boundary(self, boundary: ClothBoundary) -> None:
        """Register cloth triangles as a moving no-penetration boundary."""
        self.cloth_boundaries.append(boundary)

    def step(self, state_in, state_out, control, contacts, dt):
        """Advance APIC transfer, pressure projection, and particles by ``dt`` [s]."""
        del control, contacts
        self._emit(dt)
        if self.particle_count == 0:
            self.time += dt
            return None
        self._build_solid_mask(state_in)
        self._particles_to_grid()
        self.grid_velocity_old[:] = self.grid_velocity
        gravity = self.model.gravity.numpy()[0]
        self.grid_velocity += gravity * dt
        if self.viscosity:
            self._apply_viscosity(dt)
        if self.surface_tension:
            self._apply_surface_tension(dt)
        self._project(dt)
        self._reset_coupling_diagnostics()
        coupling_dt = dt / self.coupling_iterations
        for iteration in range(self.coupling_iterations):
            self._couple_cloth_boundaries(state_out, coupling_dt, reset=iteration == 0)
            self._couple_rigid_bodies(
                state_out,
                coupling_dt,
                relaxation=self.coupling_relaxation,
                reset=False,
            )
        self._grid_to_particles(dt)
        self._apply_particle_boundary_impulses(state_in)
        if self._step_count % 4 == 0:
            self._maintain_particle_density()
        self._step_count += 1
        self.time += dt
        return None

    def _couple_cloth_boundaries(self, state_out, dt: float, *, reset: bool = True) -> None:
        if reset:
            self.cloth_linear_impulse.clear()
            self.fluid_cloth_linear_impulse.clear()
            self.cloth_penetration_count = {boundary.object_id: 0 for boundary in self.cloth_boundaries}
            self.cloth_angular_impulse.clear()
            self.fluid_cloth_angular_impulse.clear()
        if not self.cloth_boundaries or state_out is None or state_out.particle_qd is None:
            return
        particle_velocity = state_out.particle_qd.numpy()
        inverse_mass = self.model.particle_inv_mass.numpy()
        vertex_impulses = np.zeros_like(particle_velocity, dtype=np.float64)
        for axis in range(3):
            cell_area = float(np.prod(np.delete(self.cell_size, axis)))
            for direction in (-1, 1):
                neighbour_cloth = self._shifted(self.solid_cloth, axis, direction, -1)
                interface = self.fluid & (neighbour_cloth >= 0)
                if not np.any(interface):
                    continue
                neighbour_vertices = self._shifted(self.solid_cloth_vertices, axis, direction, -1)
                neighbour_weights = self._shifted(self.solid_cloth_weights, axis, direction, 0.0)
                cells = np.argwhere(interface)
                for cell in cells:
                    key = tuple(cell)
                    cloth_index = int(neighbour_cloth[key])
                    boundary = self.cloth_boundaries[cloth_index]
                    pressure = max(float(self.pressure[key]), 0.0)
                    impulse = np.zeros(3, dtype=np.float64)
                    impulse[axis] = (
                        direction * pressure * cell_area * dt * self.cloth_drag * (1.0 - self.cloth_permeability)
                    )
                    vertices = neighbour_vertices[key]
                    weights = neighbour_weights[key]
                    movable_mass = np.divide(
                        1.0,
                        inverse_mass[vertices],
                        out=np.zeros(3),
                        where=inverse_mass[vertices] > 0.0,
                    ).sum()
                    impulse_norm = float(np.linalg.norm(impulse))
                    maximum_impulse = movable_mass
                    if impulse_norm > maximum_impulse > 0.0:
                        impulse *= maximum_impulse / impulse_norm
                        self.impulse_clip_count += 1
                    for vertex, weight in zip(vertices, weights, strict=True):
                        if vertex >= 0:
                            vertex_impulses[vertex] += impulse * float(weight)
                    self.cloth_linear_impulse[boundary.object_id] = (
                        self.cloth_linear_impulse.get(boundary.object_id, np.zeros(3)) + impulse
                    )
        for vertex in np.flatnonzero(np.linalg.norm(vertex_impulses, axis=1) > 0.0):
            particle_velocity[vertex] += vertex_impulses[vertex] * inverse_mass[vertex]
        angular_impulses = _aggregate_cloth_angular_impulses(
            self.cloth_boundaries, state_out.particle_q.numpy(), vertex_impulses
        )
        for object_id, impulse in angular_impulses.items():
            self.cloth_angular_impulse[object_id] = np.asarray(
                self.cloth_angular_impulse.get(object_id, np.zeros(3)) + impulse,
                dtype=np.float32,
            )
        self.fluid_cloth_angular_impulse = {
            object_id: -impulse for object_id, impulse in self.cloth_angular_impulse.items()
        }
        for object_id, impulse in self.cloth_linear_impulse.items():
            self.cloth_linear_impulse[object_id] = np.asarray(impulse, dtype=np.float32)
            self.fluid_cloth_linear_impulse[object_id] = -np.asarray(impulse, dtype=np.float32)
        wp.copy(
            state_out.particle_qd,
            wp.array(particle_velocity, dtype=state_out.particle_qd.dtype, device=state_out.particle_qd.device),
        )

    def _apply_particle_boundary_impulses(self, state_in) -> None:
        if state_in is None or state_in.body_q is None or self.particle_count == 0:
            return
        body_q = state_in.body_q.numpy()
        body_qd = state_in.body_qd.numpy()
        particles = self.particle_position[: self.particle_count]
        velocities = self.particle_velocity[: self.particle_count]
        for boundary in self.boundaries:
            if boundary.body is None:
                continue
            body_velocity = body_qd[boundary.body, :3]
            if body_velocity[2] >= -0.25:
                continue
            transform = body_q[boundary.body]
            center = transform[:3] + self._quat_rotate(
                transform[3:], np.asarray(boundary.local_position or (0.0, 0.0, 0.0))
            )
            relative = particles - center
            horizontal_distance = np.linalg.norm(relative[:, :2], axis=1)
            radius = max(boundary.half_extent[0], boundary.half_extent[1]) + 2.0 * max(
                self.cell_size[0], self.cell_size[1]
            )
            vertical_band = boundary.half_extent[2] + 2.0 * self.cell_size[2]
            mask = (horizontal_distance < radius) & (np.abs(relative[:, 2]) < vertical_band)
            if not np.any(mask):
                continue
            radial = relative[mask, :2]
            radial_norm = np.maximum(np.linalg.norm(radial, axis=1), 1.0e-6)
            impact_speed = abs(float(body_velocity[2]))
            velocities[mask, :2] += radial / radial_norm[:, None] * impact_speed * 0.35
            velocities[mask, 2] += impact_speed * 0.65

    def _emit(self, dt: float) -> None:
        step_index = int(round(self.time / max(dt, 1.0e-8)))
        for index, emitter in enumerate(self.emitters):
            if not emitter.start_time <= self.time < emitter.end_time:
                continue
            if self._last_emission_step.get(index) == step_index:
                continue
            minimum = tuple(emitter.position[axis] - emitter.size[axis] * 0.5 for axis in range(3))
            maximum = tuple(emitter.position[axis] + emitter.size[axis] * 0.5 for axis in range(3))
            try:
                self.initialize_box(minimum, maximum, emitter.spacing, emitter.velocity)
            except OverflowError:
                self.capacity_overflow = True
            self._last_emission_step[index] = step_index

    def _build_solid_mask(self, state_in) -> None:
        self.solid.fill(False)
        self.solid_body.fill(-1)
        self.solid_velocity.fill(0.0)
        self.solid_cloth.fill(-1)
        self.solid_cloth_vertices.fill(-1)
        self.solid_cloth_weights.fill(0.0)
        body_q = state_in.body_q.numpy() if state_in is not None and state_in.body_q is not None else None
        body_qd = state_in.body_qd.numpy() if state_in is not None and state_in.body_qd is not None else None
        cell_centers = np.stack(
            np.meshgrid(
                *(
                    self.domain_min[axis] + (np.arange(self.grid_resolution[axis]) + 0.5) * self.cell_size[axis]
                    for axis in range(3)
                ),
                indexing="ij",
            ),
            axis=-1,
        )
        for boundary in self.boundaries:
            position = np.asarray(boundary.position)
            velocity = np.zeros(3, dtype=np.float32)
            local_coordinates = cell_centers - position
            if boundary.body is not None and body_q is not None:
                body_position = body_q[boundary.body, :3]
                quaternion = body_q[boundary.body, 3:]
                local_position = np.asarray(boundary.local_position or (0.0, 0.0, 0.0))
                position = body_position + self._quat_rotate(quaternion, local_position)
                local_coordinates = self._quat_rotate_inverse(quaternion, cell_centers - position)
            elif boundary.rotation is not None:
                local_coordinates = self._quat_rotate_inverse(np.asarray(boundary.rotation), cell_centers - position)
            half_extent = np.asarray(boundary.half_extent)
            mask = np.all(np.abs(local_coordinates) <= half_extent + self.cell_size * 0.5, axis=-1)
            self.solid[mask] = True
            if boundary.body is not None and body_qd is not None:
                self.solid_body[mask] = boundary.body
                linear_velocity = body_qd[boundary.body, :3]
                angular_velocity = body_qd[boundary.body, 3:]
                velocity = linear_velocity + np.cross(angular_velocity, cell_centers - body_position)
                self.solid_velocity[mask] = velocity[mask]
            else:
                self.solid_velocity[mask] = velocity
        self._rasterize_cloth_boundaries(state_in)

    def _rasterize_cloth_boundaries(self, state_in) -> None:
        if not self.cloth_boundaries or state_in is None or state_in.particle_q is None:
            return
        positions = state_in.particle_q.numpy()
        velocities = state_in.particle_qd.numpy()
        resolution = np.asarray(self.grid_resolution)
        sample_scale = max(float(np.min(self.cell_size)) * 0.5, 1.0e-8)
        for cloth_index, boundary in enumerate(self.cloth_boundaries):
            for triangle in boundary.triangles:
                vertices = positions[triangle]
                edge_length = max(
                    float(np.linalg.norm(vertices[1] - vertices[0])),
                    float(np.linalg.norm(vertices[2] - vertices[1])),
                    float(np.linalg.norm(vertices[0] - vertices[2])),
                )
                divisions = min(64, max(1, int(np.ceil(edge_length / sample_scale))))
                for first in range(divisions + 1):
                    for second in range(divisions + 1 - first):
                        weights = np.asarray(
                            [first / divisions, second / divisions, 1.0 - (first + second) / divisions],
                            dtype=np.float32,
                        )
                        sample = weights @ vertices
                        cell = np.floor((sample - self.domain_min) / self.cell_size).astype(int)
                        if np.any(cell < 0) or np.any(cell >= resolution):
                            continue
                        key = tuple(cell)
                        self.solid[key] = True
                        self.solid_cloth[key] = cloth_index
                        self.solid_cloth_vertices[key] = triangle
                        self.solid_cloth_weights[key] = weights
                        self.solid_velocity[key] = weights @ velocities[triangle]

    @staticmethod
    def _quat_rotate(quaternion: np.ndarray, vectors: np.ndarray) -> np.ndarray:
        xyz = np.asarray(quaternion[:3])
        return vectors + 2.0 * (quaternion[3] * np.cross(xyz, vectors) + np.cross(xyz, np.cross(xyz, vectors)))

    @staticmethod
    def _quat_rotate_inverse(quaternion: np.ndarray, vectors: np.ndarray) -> np.ndarray:
        inverse = np.asarray([-quaternion[0], -quaternion[1], -quaternion[2], quaternion[3]])
        return SolverFluidAPIC._quat_rotate(inverse, vectors)

    @staticmethod
    def _shifted(values: np.ndarray, axis: int, direction: int, fill: float | bool | int) -> np.ndarray:
        result = np.full_like(values, fill)
        source = [slice(None)] * 3
        destination = [slice(None)] * 3
        if direction < 0:
            source[axis] = slice(0, -1)
            destination[axis] = slice(1, None)
        else:
            source[axis] = slice(1, None)
            destination[axis] = slice(0, -1)
        result[tuple(destination)] = values[tuple(source)]
        return result

    def _particles_to_grid(self) -> None:
        self.grid_mass.fill(0.0)
        self.grid_velocity.fill(0.0)
        positions = self.particle_position[: self.particle_count]
        velocities = self.particle_velocity[: self.particle_count]
        affine = self.particle_affine[: self.particle_count]
        masses = self.particle_mass[: self.particle_count]
        grid_positions = (positions - self.domain_min) / self.cell_size - 0.5
        base = np.floor(grid_positions).astype(int)
        fractions = grid_positions - base
        resolution = np.asarray(self.grid_resolution)
        for offset in np.ndindex(2, 2, 2):
            cells = base + offset
            valid = np.all((cells >= 0) & (cells < resolution), axis=1)
            if not np.any(valid):
                continue
            valid_cells = cells[valid]
            valid_fractions = fractions[valid]
            weight = np.prod(np.where(offset, valid_fractions, 1.0 - valid_fractions), axis=1)
            distance = (valid_cells + 0.5) * self.cell_size + self.domain_min - positions[valid]
            momentum = velocities[valid] + np.einsum("nij,nj->ni", affine[valid], distance)
            deposited_mass = weight * masses[valid]
            indices = tuple(valid_cells[:, axis] for axis in range(3))
            np.add.at(self.grid_mass, indices, deposited_mass)
            for axis in range(3):
                np.add.at(self.grid_velocity[..., axis], indices, deposited_mass * momentum[:, axis])
        occupied = self.grid_mass > 1.0e-8
        self.grid_velocity[occupied] /= self.grid_mass[occupied, None]
        self.grid_velocity[self.solid] = self.solid_velocity[self.solid]
        self.fluid[:] = occupied & ~self.solid

    def _apply_viscosity(self, dt: float) -> None:
        velocity = self.grid_velocity
        laplacian = np.zeros_like(velocity)
        for axis in range(3):
            first = np.gradient(velocity, self.cell_size[axis], axis=axis, edge_order=1)
            laplacian += np.gradient(first, self.cell_size[axis], axis=axis, edge_order=1)
        self.grid_velocity += self.viscosity * dt * laplacian

    def _apply_surface_tension(self, dt: float) -> None:
        indicator = self.fluid.astype(np.float32)
        gradient = np.stack(np.gradient(indicator, *self.cell_size, edge_order=1), axis=-1)
        magnitude = np.linalg.norm(gradient, axis=-1)
        normal = np.divide(
            gradient,
            magnitude[..., None],
            out=np.zeros_like(gradient),
            where=magnitude[..., None] > 1.0e-6,
        )
        curvature = -sum(
            np.gradient(normal[..., axis], self.cell_size[axis], axis=axis, edge_order=1) for axis in range(3)
        )
        acceleration = self.surface_tension / self.density * curvature[..., None] * gradient
        self.grid_velocity += acceleration * dt

    def _project(self, dt: float) -> None:
        velocity = self.grid_velocity
        self.divergence.fill(0.0)
        for axis in range(3):
            self.divergence += np.gradient(velocity[..., axis], self.cell_size[axis], axis=axis, edge_order=1)
        self.divergence[~self.fluid] = 0.0
        self.pressure.fill(0.0)
        coordinates = np.indices(self.grid_resolution)
        parity = coordinates.sum(axis=0) & 1
        rhs = self.density * self.divergence / max(dt, 1.0e-8)

        def shifted(values: np.ndarray, axis: int, direction: int, fill: float | bool) -> np.ndarray:
            result = np.full_like(values, fill)
            source = [slice(None)] * 3
            destination = [slice(None)] * 3
            if direction < 0:
                source[axis] = slice(0, -1)
                destination[axis] = slice(1, None)
            else:
                source[axis] = slice(1, None)
                destination[axis] = slice(0, -1)
            result[tuple(destination)] = values[tuple(source)]
            return result

        for _ in range(self.pressure_iters):
            for color in (0, 1):
                neighbour_sum = np.zeros_like(self.pressure)
                diagonal = np.zeros_like(self.pressure)
                for axis in range(3):
                    coefficient = 1.0 / self.cell_size[axis] ** 2
                    for direction in (-1, 1):
                        neighbour_pressure = shifted(self.pressure, axis, direction, 0.0)
                        neighbour_solid = shifted(self.solid, axis, direction, True)
                        open_face = ~neighbour_solid
                        neighbour_sum += coefficient * neighbour_pressure * open_face
                        diagonal += coefficient * open_face
                candidate = np.divide(
                    neighbour_sum - rhs,
                    diagonal,
                    out=np.zeros_like(self.pressure),
                    where=diagonal > 0.0,
                )
                mask = self.fluid & (parity == color) & (diagonal > 0.0)
                self.pressure[mask] = candidate[mask]
        gravity_z = abs(float(self.model.gravity.numpy()[0, 2]))
        if gravity_z:
            z_centers = self.domain_min[2] + (np.arange(self.grid_resolution[2]) + 0.5) * self.cell_size[2]
            for i, j in np.ndindex(self.grid_resolution[0], self.grid_resolution[1]):
                fluid_levels = np.flatnonzero(self.fluid[i, j])
                if fluid_levels.size:
                    surface_z = z_centers[fluid_levels[-1]] + self.cell_size[2] * 0.5
                    self.pressure[i, j, fluid_levels] += (
                        self.density * gravity_z * np.maximum(0.0, surface_z - z_centers[fluid_levels])
                    )
        for axis in range(3):
            lower_pressure = shifted(self.pressure, axis, -1, 0.0)
            upper_pressure = shifted(self.pressure, axis, 1, 0.0)
            lower_solid = shifted(self.solid, axis, -1, True)
            upper_solid = shifted(self.solid, axis, 1, True)
            lower_pressure[lower_solid] = self.pressure[lower_solid]
            upper_pressure[upper_solid] = self.pressure[upper_solid]
            gradient = (upper_pressure - lower_pressure) / (2.0 * self.cell_size[axis])
            velocity[..., axis] -= dt * gradient / self.density
        velocity[~self.fluid] = 0.0
        velocity[self.solid] = self.solid_velocity[self.solid]
        self.divergence.fill(0.0)
        for axis in range(3):
            self.divergence += np.gradient(velocity[..., axis], self.cell_size[axis], axis=axis, edge_order=1)
        self.divergence[~self.fluid] = 0.0

    def _reset_coupling_diagnostics(self) -> None:
        self.rigid_linear_impulse.clear()
        self.rigid_angular_impulse.clear()
        self.rigid_pressure_impulse.clear()
        self.rigid_stabilization_impulse.clear()
        self.impulse_clip_count = 0

    def _couple_rigid_bodies(self, state_out, dt: float, *, relaxation: float = 1.0, reset: bool = True) -> None:
        if reset:
            self._reset_coupling_diagnostics()
        if state_out is None or state_out.body_qd is None:
            return
        transforms = state_out.body_q.numpy()
        velocities = state_out.body_qd.numpy()
        masses = self.model.body_mass.numpy()
        inertias = self.model.body_inertia.numpy()
        cell_centers = np.stack(
            np.meshgrid(
                *(
                    self.domain_min[axis] + (np.arange(self.grid_resolution[axis]) + 0.5) * self.cell_size[axis]
                    for axis in range(3)
                ),
                indexing="ij",
            ),
            axis=-1,
        )
        dynamic_bodies = np.unique(self.solid_body[self.solid_body >= 0])
        for body_value in dynamic_bodies:
            body = int(body_value)
            if masses[body] <= 0.0:
                continue
            position = transforms[body, :3]
            force = np.zeros(3, dtype=np.float64)
            torque = np.zeros(3, dtype=np.float64)
            for axis in range(3):
                cell_area = float(np.prod(np.delete(self.cell_size, axis)))
                for direction in (-1, 1):
                    neighbour_body = self._shifted(self.solid_body, axis, direction, -1)
                    interface = self.fluid & (neighbour_body == body)
                    if not np.any(interface):
                        continue
                    interface_pressure = np.maximum(self.pressure[interface], 0.0)
                    scalar_forces = interface_pressure.astype(np.float64) * cell_area
                    face_force = np.zeros((len(scalar_forces), 3), dtype=np.float64)
                    face_force[:, axis] = direction * scalar_forces
                    face_positions = cell_centers[interface].astype(np.float64)
                    face_positions[:, axis] += direction * self.cell_size[axis] * 0.5
                    force += face_force.sum(axis=0)
                    torque += np.cross(face_positions - position, face_force).sum(axis=0)
            rotated_basis = self._quat_rotate(transforms[body, 3:], np.eye(3))
            world_inertia = rotated_basis.T @ inertias[body] @ rotated_basis
            pressure_force = force.copy()
            stabilization_force = -masses[body] * 4.0 * velocities[body, :3]
            stabilization_torque = -(world_inertia @ (4.0 * velocities[body, 3:]))
            force += stabilization_force
            torque += stabilization_torque
            pressure_impulse = pressure_force * dt * relaxation
            stabilization_impulse = stabilization_force * dt * relaxation
            impulse = pressure_impulse + stabilization_impulse
            angular_impulse = torque * dt * relaxation
            gravity = abs(float(self.model.gravity.numpy()[0, 2]))
            max_impulse = masses[body] * 2.0 * gravity * dt
            impulse_norm = float(np.linalg.norm(impulse))
            if impulse_norm > max_impulse > 0.0:
                scale = max_impulse / impulse_norm
                impulse *= scale
                pressure_impulse *= scale
                stabilization_impulse *= scale
                self.impulse_clip_count += 1
            angular_velocity_delta = np.linalg.solve(world_inertia, angular_impulse)
            max_angular_velocity_delta = 20.0 * dt
            angular_delta_norm = float(np.linalg.norm(angular_velocity_delta))
            if angular_delta_norm > max_angular_velocity_delta:
                scale = max_angular_velocity_delta / angular_delta_norm
                angular_impulse *= scale
                angular_velocity_delta *= scale
                self.impulse_clip_count += 1
            velocities[body, :3] += impulse / masses[body]
            velocities[body, 3:] += angular_velocity_delta
            self.rigid_linear_impulse[body] = (self.rigid_linear_impulse.get(body, np.zeros(3)) + impulse).astype(
                np.float32
            )
            self.rigid_angular_impulse[body] = (
                self.rigid_angular_impulse.get(body, np.zeros(3)) + angular_impulse
            ).astype(np.float32)
            self.rigid_pressure_impulse[body] = (
                self.rigid_pressure_impulse.get(body, np.zeros(3)) + pressure_impulse
            ).astype(np.float32)
            self.rigid_stabilization_impulse[body] = (
                self.rigid_stabilization_impulse.get(body, np.zeros(3)) + stabilization_impulse
            ).astype(np.float32)
        wp.copy(
            state_out.body_qd,
            wp.array(velocities, dtype=state_out.body_qd.dtype, device=state_out.body_qd.device),
        )

    def _grid_to_particles(self, dt: float) -> None:
        grid_delta = self.grid_velocity - self.grid_velocity_old
        positions = self.particle_position[: self.particle_count]
        old_velocities = self.particle_velocity[: self.particle_count]
        cells = np.floor((positions - self.domain_min) / self.cell_size).astype(int)
        cells = np.clip(cells, 0, np.asarray(self.grid_resolution) - 1)
        indices = tuple(cells[:, axis] for axis in range(3))
        pic_velocities = self.grid_velocity[indices]
        flip_velocities = old_velocities + grid_delta[indices]
        velocities = (1.0 - self.flip_ratio) * pic_velocities + self.flip_ratio * flip_velocities
        # Keep advection within half a grid cell per step. Besides satisfying
        # the particle CFL condition, this bounds continuous-solid collision
        # sampling and prevents one unstable pressure update from producing an
        # unbounded number of trajectory samples.
        maximum_speed = self.cfl_number * float(np.min(self.cell_size)) / max(dt, 1.0e-8)
        speeds = np.linalg.norm(velocities, axis=1)
        fast_velocity = speeds > maximum_speed
        if np.any(fast_velocity):
            velocities[fast_velocity] *= maximum_speed / speeds[fast_velocity, None]
        next_positions = positions + velocities * dt
        next_cells = np.floor((next_positions - self.domain_min) / self.cell_size).astype(int)
        next_cells = np.clip(next_cells, 0, np.asarray(self.grid_resolution) - 1)
        next_indices = tuple(next_cells[:, axis] for axis in range(3))
        hits = self.solid[next_indices]
        if np.any(hits):
            hit_cloth = self.solid_cloth[next_indices][hits]
            for cloth_index in hit_cloth[hit_cloth >= 0]:
                object_id = self.cloth_boundaries[int(cloth_index)].object_id
                self.cloth_penetration_count[object_id] = self.cloth_penetration_count.get(object_id, 0) + 1
            next_positions[hits] = positions[hits]
            velocities[hits] = self.solid_velocity[next_indices][hits]
            if self.boundary_friction > 0.0:
                velocities[hits] /= 1.0 + self.boundary_friction
        displacements = next_positions - positions
        fast = np.flatnonzero(np.linalg.norm(displacements, axis=1) > np.min(self.cell_size) * 0.5)
        for particle in fast:
            sample_count = int(np.ceil(np.linalg.norm(displacements[particle]) / (np.min(self.cell_size) * 0.5)))
            previous = positions[particle].copy()
            for sample_index in range(1, sample_count + 1):
                sample = positions[particle] + displacements[particle] * (sample_index / sample_count)
                sample_cell = np.floor((sample - self.domain_min) / self.cell_size).astype(int)
                sample_cell = np.clip(sample_cell, 0, np.asarray(self.grid_resolution) - 1)
                if self.solid[tuple(sample_cell)]:
                    next_positions[particle] = previous
                    velocities[particle] = self.solid_velocity[tuple(sample_cell)]
                    break
                previous = sample
        lower = self.domain_min + self.cell_size * 0.5
        upper = self.domain_max - self.cell_size * 0.5
        clipped = np.clip(next_positions, lower, upper)
        boundary_hits = clipped != next_positions
        velocities[boundary_hits] = 0.0
        self.particle_position[: self.particle_count] = clipped
        self.particle_velocity[: self.particle_count] = velocities
        affine = self.particle_affine[: self.particle_count]
        for component in range(3):
            derivatives = np.gradient(
                self.grid_velocity[..., component],
                *self.cell_size,
                edge_order=1,
            )
            for axis in range(3):
                affine[:, component, axis] = derivatives[axis][indices]

    def _maintain_particle_density(self) -> None:
        """Keep four to eight mass-carrying particles in each occupied cell."""
        if self.particle_count == 0:
            return
        cells = np.floor((self.particle_position[: self.particle_count] - self.domain_min) / self.cell_size).astype(int)
        cells = np.clip(cells, 0, np.asarray(self.grid_resolution) - 1)
        groups: dict[tuple[int, int, int], list[int]] = {}
        for particle, cell in enumerate(cells):
            groups.setdefault(tuple(cell), []).append(particle)
        keep = np.ones(self.particle_count, dtype=bool)
        additions: list[tuple[np.ndarray, np.ndarray, np.ndarray, float]] = []
        offsets = np.asarray(
            [
                (-0.2, -0.2, -0.2),
                (0.2, -0.2, 0.2),
                (-0.2, 0.2, 0.2),
                (0.2, 0.2, -0.2),
            ],
            dtype=np.float32,
        )
        for cell, cell_particles in groups.items():
            active_particles = cell_particles
            if len(cell_particles) > 8:
                retained, merged = cell_particles[:7], cell_particles[7:]
                masses = self.particle_mass[merged]
                total_mass = float(masses.sum())
                destination = merged[0]
                if total_mass > 0.0:
                    weights = masses / total_mass
                    self.particle_position[destination] = np.sum(
                        self.particle_position[merged] * weights[:, None], axis=0
                    )
                    self.particle_velocity[destination] = np.sum(
                        self.particle_velocity[merged] * weights[:, None], axis=0
                    )
                    self.particle_affine[destination] = np.sum(
                        self.particle_affine[merged] * weights[:, None, None], axis=0
                    )
                    self.particle_mass[destination] = total_mass
                keep[merged[1:]] = False
                active_particles = [*retained, destination]
            if len(active_particles) < 4:
                total_mass = float(self.particle_mass[active_particles].sum())
                if total_mass < self._nominal_particle_mass * 0.25:
                    continue
                target_mass = total_mass / 4.0
                self.particle_mass[active_particles] = target_mass
                mean_velocity = self.particle_velocity[active_particles].mean(axis=0)
                mean_affine = self.particle_affine[active_particles].mean(axis=0)
                center = self.domain_min + (np.asarray(cell) + 0.5) * self.cell_size
                for offset in offsets[: 4 - len(active_particles)]:
                    additions.append((center + offset * self.cell_size, mean_velocity, mean_affine, target_mass))
        if not np.all(keep):
            retained = np.flatnonzero(keep)
            retained_count = len(retained)
            self.particle_position[:retained_count] = self.particle_position[retained]
            self.particle_velocity[:retained_count] = self.particle_velocity[retained]
            self.particle_affine[:retained_count] = self.particle_affine[retained]
            self.particle_mass[:retained_count] = self.particle_mass[retained]
            self.particle_count = retained_count
        available = self.max_particles - self.particle_count
        if len(additions) > available:
            self.capacity_overflow = True
            additions = additions[:available]
        for position, velocity, affine, mass in additions:
            index = self.particle_count
            self.particle_position[index] = position
            self.particle_velocity[index] = velocity
            self.particle_affine[index] = affine
            self.particle_mass[index] = mass
            self.particle_count += 1

    def update_contacts(self, contacts) -> None:
        """Leave Newton rigid contacts unchanged; liquid contacts are grid based."""
        del contacts

    def notify_model_changed(self, flags: int) -> None:
        """Acknowledge model changes used on the following liquid step."""
        del flags

    def particle_cell_counts(self) -> np.ndarray:
        """Return occupied-cell particle counts for diagnostics."""
        if self.particle_count == 0:
            return np.empty(0, dtype=np.int32)
        cells = np.floor((self.particle_position[: self.particle_count] - self.domain_min) / self.cell_size).astype(int)
        cells = np.clip(cells, 0, np.asarray(self.grid_resolution) - 1)
        _, inverse, counts = np.unique(cells, axis=0, return_inverse=True, return_counts=True)
        cell_masses = np.bincount(
            inverse,
            weights=self.particle_mass[: self.particle_count],
            minlength=len(counts),
        )
        return counts[cell_masses >= self._nominal_particle_mass * 0.25]
