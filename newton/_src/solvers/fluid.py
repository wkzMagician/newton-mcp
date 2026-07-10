# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Public fluid solver implementations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import warp as wp

from .ca_exercises.solver_exercise5_fluid import SolverExercise5Fluid, apply_density_source_kernel
from .solver import SolverBase


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
        self._solid_velocity = np.zeros((self.nx, self.ny, self.nz, 3), dtype=np.float32)
        self.time = 0.0

    def add_boundary(self, boundary: Boundary) -> None:
        """Register a static, kinematic, or dynamic smoke obstacle."""
        self.boundaries.append(boundary)

    def _update_boundaries(self, state_in) -> None:
        solid = np.zeros((self.nx, self.ny, self.nz), dtype=np.int32)
        self._solid_velocity.fill(0.0)
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
                linear = body_qd[boundary.body, :3]
                angular = body_qd[boundary.body, 3:]
                self._solid_velocity[mask] = (linear + np.cross(angular, coordinates - position))[mask]
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
        if self.dissipation:
            wp.launch(
                _scale_density,
                dim=(self.nx, self.ny, self.nz),
                inputs=[self.density, max(0.0, 1.0 - self.dissipation * dt)],
                device=self.model.device,
            )
        self.time += dt
        return result


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
        """Axis-aligned solid coupled to the liquid grid."""

        position: tuple[float, float, float]
        half_extent: tuple[float, float, float]
        body: int | None = None
        local_position: tuple[float, float, float] | None = None

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
        self.solid_velocity = np.zeros((*shape, 3), dtype=np.float32)
        self.emitters: list[SolverFluidAPIC.Emitter] = []
        self.boundaries: list[SolverFluidAPIC.Boundary] = []
        self.capacity_overflow = False
        self.rigid_linear_impulse: dict[int, np.ndarray] = {}
        self.rigid_angular_impulse: dict[int, np.ndarray] = {}
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
        self._couple_rigid_bodies(state_out, dt)
        self._grid_to_particles(dt)
        self._apply_particle_boundary_impulses(state_in)
        if self._step_count % 4 == 0:
            self._maintain_particle_density()
        self._step_count += 1
        self.time += dt
        return None

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
        self.solid_velocity.fill(0.0)
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
            half_extent = np.asarray(boundary.half_extent)
            mask = np.all(np.abs(local_coordinates) <= half_extent + self.cell_size * 0.5, axis=-1)
            self.solid[mask] = True
            if boundary.body is not None and body_qd is not None:
                linear_velocity = body_qd[boundary.body, :3]
                angular_velocity = body_qd[boundary.body, 3:]
                velocity = linear_velocity + np.cross(angular_velocity, cell_centers - body_position)
                self.solid_velocity[mask] = velocity[mask]
            else:
                self.solid_velocity[mask] = velocity

    @staticmethod
    def _quat_rotate(quaternion: np.ndarray, vectors: np.ndarray) -> np.ndarray:
        xyz = np.asarray(quaternion[:3])
        return vectors + 2.0 * (quaternion[3] * np.cross(xyz, vectors) + np.cross(xyz, np.cross(xyz, vectors)))

    @staticmethod
    def _quat_rotate_inverse(quaternion: np.ndarray, vectors: np.ndarray) -> np.ndarray:
        inverse = np.asarray([-quaternion[0], -quaternion[1], -quaternion[2], quaternion[3]])
        return SolverFluidAPIC._quat_rotate(inverse, vectors)

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
        spacing = float(np.mean(self.cell_size))
        self.divergence.fill(0.0)
        for axis in range(3):
            self.divergence += np.gradient(velocity[..., axis], self.cell_size[axis], axis=axis, edge_order=1)
        self.divergence[~self.fluid] = 0.0
        self.pressure.fill(0.0)
        coordinates = np.indices(self.grid_resolution)
        parity = coordinates.sum(axis=0) & 1
        scale = self.density * spacing * spacing / max(dt, 1.0e-8)
        for _ in range(self.pressure_iters):
            for color in (0, 1):
                neighbour_sum = np.zeros_like(self.pressure)
                for axis in range(3):
                    lower = np.roll(self.pressure, 1, axis=axis)
                    upper = np.roll(self.pressure, -1, axis=axis)
                    boundary = [slice(None)] * 3
                    boundary[axis] = 0
                    lower[tuple(boundary)] = 0.0
                    boundary[axis] = -1
                    upper[tuple(boundary)] = 0.0
                    neighbour_sum += lower + upper
                candidate = (neighbour_sum - scale * self.divergence) / 6.0
                mask = self.fluid & (parity == color)
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
            gradient = np.gradient(self.pressure, self.cell_size[axis], axis=axis, edge_order=1)
            velocity[..., axis] -= dt * gradient / self.density
        velocity[~self.fluid] = 0.0
        velocity[self.solid] = self.solid_velocity[self.solid]
        self.divergence.fill(0.0)
        for axis in range(3):
            self.divergence += np.gradient(velocity[..., axis], self.cell_size[axis], axis=axis, edge_order=1)
        self.divergence[~self.fluid] = 0.0

    def _couple_rigid_bodies(self, state_out, dt: float) -> None:
        self.rigid_linear_impulse.clear()
        self.rigid_angular_impulse.clear()
        if state_out is None or state_out.body_qd is None:
            return
        velocities = state_out.body_qd.numpy()
        masses = self.model.body_mass.numpy()
        cell_area = float(np.mean(self.cell_size) ** 2)
        for boundary in self.boundaries:
            if boundary.body is None or masses[boundary.body] <= 0.0:
                continue
            transform = state_out.body_q.numpy()[boundary.body]
            position = transform[:3] + self._quat_rotate(
                transform[3:], np.asarray(boundary.local_position or (0.0, 0.0, 0.0))
            )
            half_extent = np.asarray(boundary.half_extent)
            low = np.floor((position - half_extent - self.domain_min) / self.cell_size).astype(int)
            high = np.floor((position + half_extent - self.domain_min) / self.cell_size).astype(int)
            low = np.clip(low, 0, np.asarray(self.grid_resolution) - 1)
            high = np.clip(high, 0, np.asarray(self.grid_resolution) - 1)
            force = np.zeros(3, dtype=np.float64)
            torque = np.zeros(3, dtype=np.float64)
            for axis in range(3):
                negative = [slice(low[i], high[i] + 1) for i in range(3)]
                positive = list(negative)
                negative[axis] = max(low[axis] - 1, 0)
                positive[axis] = min(high[axis] + 1, self.grid_resolution[axis] - 1)
                for face, sign in ((negative, 1.0), (positive, -1.0)):
                    face_pressure = self.pressure[tuple(face)]
                    scalar_forces = face_pressure.reshape(-1) * cell_area
                    face_force = np.zeros((scalar_forces.size, 3), dtype=np.float64)
                    face_force[:, axis] = sign * scalar_forces
                    ranges = [
                        np.arange(low[index], high[index] + 1)
                        if isinstance(face[index], slice)
                        else np.asarray([face[index]])
                        for index in range(3)
                    ]
                    indices = np.stack(np.meshgrid(*ranges, indexing="ij"), axis=-1).reshape(-1, 3)
                    face_positions = self.domain_min + (indices + 0.5) * self.cell_size
                    force += face_force.sum(axis=0)
                    torque += np.cross(face_positions - position, face_force).sum(axis=0)
            fluid_cells = np.argwhere(self.fluid)
            if fluid_cells.size:
                surface_z = self.domain_min[2] + (float(fluid_cells[:, 2].max()) + 1.0) * self.cell_size[2]
                submerged_height = np.clip(
                    surface_z - (position[2] - half_extent[2]),
                    0.0,
                    2.0 * half_extent[2],
                )
                displaced_volume = 4.0 * half_extent[0] * half_extent[1] * submerged_height
                buoyancy = self.density * abs(float(self.model.gravity.numpy()[0, 2])) * displaced_volume
                force[2] = max(force[2], buoyancy)
                force -= masses[boundary.body] * 4.0 * velocities[boundary.body, :3]
                lateral_limit = max(buoyancy * 0.02, 1.0e-6)
                force[:2] = np.clip(force[:2], -lateral_limit, lateral_limit)
            impulse = force * dt
            angular_impulse = torque * dt
            volume = 8.0 * float(np.prod(half_extent))
            max_impulse = self.density * volume * abs(float(self.model.gravity.numpy()[0, 2])) * dt * 2.0
            impulse_norm = float(np.linalg.norm(impulse))
            if impulse_norm > max_impulse > 0.0:
                impulse *= max_impulse / impulse_norm
            max_angular_impulse = max_impulse * float(np.max(half_extent)) * 0.1
            angular_norm = float(np.linalg.norm(angular_impulse))
            if angular_norm > max_angular_impulse > 0.0:
                angular_impulse *= max_angular_impulse / angular_norm
            velocities[boundary.body, :3] += impulse / masses[boundary.body]
            inertia = (
                masses[boundary.body]
                / 3.0
                * np.asarray(
                    [
                        half_extent[1] ** 2 + half_extent[2] ** 2,
                        half_extent[0] ** 2 + half_extent[2] ** 2,
                        half_extent[0] ** 2 + half_extent[1] ** 2,
                    ]
                )
            )
            velocities[boundary.body, 3:] += angular_impulse / np.maximum(inertia, 1.0e-8)
            self.rigid_linear_impulse[boundary.body] = impulse.astype(np.float32)
            self.rigid_angular_impulse[boundary.body] = angular_impulse.astype(np.float32)
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
        next_positions = positions + velocities * dt
        next_cells = np.floor((next_positions - self.domain_min) / self.cell_size).astype(int)
        next_cells = np.clip(next_cells, 0, np.asarray(self.grid_resolution) - 1)
        next_indices = tuple(next_cells[:, axis] for axis in range(3))
        hits = self.solid[next_indices]
        if np.any(hits):
            next_positions[hits] = positions[hits]
            velocities[hits] = self.solid_velocity[next_indices][hits]
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
