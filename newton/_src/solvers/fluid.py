# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Public fluid solver implementations."""

from __future__ import annotations

from .ca_exercises.solver_exercise5_fluid import SolverExercise5Fluid
from .semi_implicit import SolverSemiImplicit


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


class SolverFluidAPIC(SolverSemiImplicit):
    """Particle fluid integration entry point for APIC/FLIP scene pipelines.

    This solver uses Newton's particle/rigid contact integration for the
    Lagrangian stage. The scene compiler owns MAC transfers and pressure
    projection, allowing one frame loop to couple the resulting impulses back
    to rigid bodies.

    Args:
        model: Newton model containing liquid particles and rigid colliders.
        grid_resolution: Pressure-grid cell counts.
        domain_size: Physical pressure-grid size [m].
        flip_ratio: FLIP contribution in the APIC/FLIP blend.
        pressure_iters: Pressure projection iteration count.
    """

    def __init__(
        self,
        model,
        grid_resolution: tuple[int, int, int] = (32, 32, 32),
        domain_size: tuple[float, float, float] = (1.0, 1.0, 1.0),
        flip_ratio: float = 0.95,
        pressure_iters: int = 40,
    ):
        if any(value < 2 for value in grid_resolution):
            raise ValueError("grid_resolution axes must contain at least two cells")
        if any(value <= 0.0 for value in domain_size):
            raise ValueError("domain_size components must be positive")
        if not 0.0 <= flip_ratio <= 1.0:
            raise ValueError("flip_ratio must be between zero and one")
        if pressure_iters <= 0:
            raise ValueError("pressure_iters must be positive")
        super().__init__(model)
        self.grid_resolution = grid_resolution
        self.domain_size = domain_size
        self.flip_ratio = flip_ratio
        self.pressure_iters = pressure_iters
