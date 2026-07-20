# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Serializable task specifications and reusable scene-specific evaluators."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, Mapping


@dataclass(frozen=True, slots=True)
class TaskMetricSpec:
    """Map one simulation result path to a normalized task loss."""

    name: str
    path: str
    target: float
    tolerance: float = 1.0
    mode: Literal["target", "minimum", "maximum"] = "target"
    missing_loss: float = 10.0

    def __post_init__(self) -> None:
        if not self.name or not self.path or self.tolerance <= 0.0 or self.missing_loss < 0.0:
            raise ValueError("Task metric name/path, positive tolerance, and non-negative missing loss are required")


@dataclass(frozen=True, slots=True)
class TaskSpec:
    """Agent-authored task semantics stored separately from scene IR."""

    name: str
    metrics: tuple[TaskMetricSpec, ...]

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable task specification."""
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> TaskSpec:
        """Parse a task specification mapping."""
        return cls(
            name=str(value["name"]),
            metrics=tuple(TaskMetricSpec(**item) for item in value.get("metrics", ())),
        )

    def evaluate(self, simulation: Mapping[str, Any]) -> dict[str, float]:
        """Evaluate all configured result paths as lower-is-better losses."""
        result = {}
        for spec in self.metrics:
            try:
                value = float(_resolve(simulation, spec.path))
            except ValueError as error:
                if not str(error).startswith("Unknown task metric path"):
                    raise
                # A validated path can disappear when a candidate fails to create
                # an event (for example, no contact).  This is a task miss, not a
                # framework exception that should terminate the optimization run.
                result[spec.name] = spec.missing_loss
                continue
            if spec.mode == "target":
                loss = abs(value - spec.target) / spec.tolerance
            elif spec.mode == "minimum":
                loss = max(0.0, spec.target - value) / spec.tolerance
            else:
                loss = max(0.0, value - spec.target) / spec.tolerance
            result[spec.name] = loss
        return result


def _resolve(value: Mapping[str, Any], path: str) -> Any:
    current: Any = value
    for part in path.split("."):
        if isinstance(current, Mapping) and part in current:
            current = current[part]
        else:
            raise ValueError(f"Unknown task metric path: {path!r}")
    if isinstance(current, bool) or not isinstance(current, (int, float)):
        raise ValueError(f"Task metric path is not numeric: {path!r}")
    return current
