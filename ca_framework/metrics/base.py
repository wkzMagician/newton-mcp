# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Public metric pipeline contracts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Protocol

from ca_framework.scene.model import Scene
from ca_framework.scene.telemetry import SimulationTelemetry


@dataclass(frozen=True, slots=True)
class ConstraintResult:
    """One hard feasibility constraint."""

    name: str
    value: float
    threshold: float
    satisfied: bool


@dataclass(frozen=True, slots=True)
class MetricTerm:
    """One normalized soft objective term."""

    name: str
    value: float
    threshold: float
    weight: float = 1.0


@dataclass(slots=True)
class ObjectiveResult:
    """Constraint-first scalar objective and its complete evidence."""

    feasible: bool
    objective: float
    metrics: dict[str, float]
    constraints: dict[str, float]
    runtime_sec: float
    status: str
    failure_reason: str | None = None
    constraint_results: list[ConstraintResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""
        return {
            "feasible": self.feasible,
            "objective": self.objective,
            "metrics": self.metrics,
            "constraints": self.constraints,
            "runtime_sec": self.runtime_sec,
            "status": self.status,
            "failure_reason": self.failure_reason,
            "constraint_results": [asdict(item) for item in self.constraint_results],
        }


class TaskEvaluator(Protocol):
    """Evaluate scene-specific behavior from common telemetry."""

    def evaluate(self, scene: Scene, telemetry: SimulationTelemetry) -> dict[str, float]:
        """Return task losses, where lower values are better."""
        ...
