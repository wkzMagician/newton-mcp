# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Backend-neutral structured simulation telemetry."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from ca_framework.coupling import CouplingExchange


@dataclass(slots=True)
class BodyTelemetry:
    """Rigid-body summary at one substep."""

    linear_speed: float = 0.0
    angular_speed: float = 0.0
    finite: bool = True


@dataclass(slots=True)
class ClothTelemetry:
    """Cloth quality summary at one substep."""

    max_speed: float = 0.0
    min_edge_length_ratio: float = 1.0
    max_edge_length_ratio: float = 1.0
    min_triangle_area_ratio: float = 1.0
    max_triangle_area_ratio: float = 1.0
    flipped_triangle_count: int = 0
    finite: bool = True


@dataclass(slots=True)
class FluidTelemetry:
    """Fluid quality summary at one substep."""

    mass: float = 0.0
    max_divergence: float = 0.0
    particle_count: int = 0
    finite: bool = True


@dataclass(slots=True)
class ContactTelemetry:
    """Contact summary at one substep."""

    object_a: str
    object_b: str
    penetration: float = 0.0
    normal_impulse: float = 0.0
    tangential_impulse: float = 0.0


@dataclass(slots=True)
class SolverTelemetry:
    """Solver convergence and cost summary at one substep."""

    iterations: int = 0
    residual: float | None = None
    coupling_iterations: int = 0
    coupling_nonconvergence: bool = False
    impulse_clips: int = 0


@dataclass(slots=True)
class FrameTelemetry:
    """Structured observations recorded after one simulation substep."""

    frame: int
    substep: int
    time: float
    body_states: dict[str, BodyTelemetry] = field(default_factory=dict)
    cloth_states: dict[str, ClothTelemetry] = field(default_factory=dict)
    fluid_states: dict[str, FluidTelemetry] = field(default_factory=dict)
    contacts: list[ContactTelemetry] = field(default_factory=list)
    coupling_exchanges: list[CouplingExchange] = field(default_factory=list)
    solver_stats: SolverTelemetry = field(default_factory=SolverTelemetry)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable telemetry record."""
        result = asdict(self)
        result["coupling_exchanges"] = [exchange.to_dict() for exchange in self.coupling_exchanges]
        return result


@dataclass(slots=True)
class SimulationTelemetry:
    """Telemetry sequence for one complete trial."""

    frames: list[FrameTelemetry] = field(default_factory=list)

    def append(self, value: FrameTelemetry) -> None:
        """Append one substep record."""
        self.frames.append(value)

    def to_records(self) -> list[dict[str, Any]]:
        """Return JSON-serializable substep records."""
        return [frame.to_dict() for frame in self.frames]
