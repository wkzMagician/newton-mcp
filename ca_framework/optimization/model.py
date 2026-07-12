# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Optimization-plan data models."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal


@dataclass(frozen=True, slots=True)
class ParameterSpec:
    """Define one tunable scene parameter.

    Args:
        name: Stable display and result key.
        path: Dot-separated path in the serialized scene.
        scope: Physical subsystem that owns the parameter.
        kind: Value distribution kind.
        lower: Inclusive numeric lower bound.
        upper: Inclusive numeric upper bound.
        choices: Allowed values for categorical parameters.
        transform: Sampling-space transform.
        stage: Optimization stage in which the parameter is active.
        enabled: Whether the parameter participates in optimization.
    """

    name: str
    path: str
    scope: Literal["scene", "material", "solver", "coupling"]
    kind: Literal["float", "int", "categorical", "bool"]
    lower: float | int | None = None
    upper: float | int | None = None
    choices: tuple[Any, ...] | None = None
    transform: Literal["linear", "log", "logit"] = "linear"
    stage: Literal["global", "refine"] = "global"
    enabled: bool = True

    def __post_init__(self) -> None:
        if not self.name or not self.path or any(not part for part in self.path.split(".")):
            raise ValueError("Parameter name and every path component must be non-empty")
        if self.kind in {"float", "int"}:
            if self.lower is None or self.upper is None or self.lower > self.upper:
                raise ValueError("Numeric parameters require ordered lower and upper bounds")
            if self.choices is not None:
                raise ValueError("Numeric parameters cannot define choices")
            if self.transform == "log" and self.lower <= 0:
                raise ValueError("Log-transformed parameters require a positive lower bound")
            if self.transform == "logit" and not (0.0 < self.lower < self.upper < 1.0):
                raise ValueError("Logit-transformed parameters require bounds strictly inside (0, 1)")
        elif self.kind == "categorical":
            if not self.choices:
                raise ValueError("Categorical parameters require non-empty choices")
            if self.lower is not None or self.upper is not None:
                raise ValueError("Categorical parameters cannot define numeric bounds")
        elif any(value is not None for value in (self.lower, self.upper, self.choices)):
            raise ValueError("Boolean parameters do not accept bounds or choices")

    def validate_value(self, value: Any) -> None:
        """Raise :class:`ValueError` when a candidate value is invalid."""
        if self.kind == "bool":
            valid = type(value) is bool
        elif self.kind == "int":
            valid = type(value) is int and self.lower <= value <= self.upper
        elif self.kind == "float":
            valid = type(value) in {float, int} and type(value) is not bool and self.lower <= value <= self.upper
        else:
            valid = value in self.choices
        if not valid:
            raise ValueError(f"Invalid value for parameter {self.name!r}: {value!r}")


@dataclass(frozen=True, slots=True)
class FidelitySettings:
    """Framework-owned fidelity levels, separate from physical parameters."""

    short_duration_fraction: float = 0.25
    low_resolution_scale: float = 0.5
    render_top_k: int = 3

    def __post_init__(self) -> None:
        if not 0.0 < self.short_duration_fraction <= 1.0:
            raise ValueError("short_duration_fraction must be in (0, 1]")
        if not 0.0 < self.low_resolution_scale <= 1.0:
            raise ValueError("low_resolution_scale must be in (0, 1]")
        if self.render_top_k < 0:
            raise ValueError("render_top_k cannot be negative")


@dataclass(frozen=True, slots=True)
class OptimizationPlan:
    """Serializable optimizer configuration stored outside :class:`Scene`."""

    name: str
    optimizer: Literal["random", "tpe", "cmaes"] = "tpe"
    parameters: tuple[ParameterSpec, ...] = ()
    seed: int = 0
    trials: int = 50
    timeout_sec: float | None = None
    max_wall_time_sec: float | None = None
    study_name: str | None = None
    storage_url: str | None = None
    fidelity: FidelitySettings = field(default_factory=FidelitySettings)
    robustness_top_k: int = 3
    robustness_samples: int = 3
    robustness_beta: float = 1.0
    acceptable_objective: float = 100.0

    def __post_init__(self) -> None:
        if not self.name or self.trials <= 0:
            raise ValueError("Optimization plan name and positive trial count are required")
        if self.timeout_sec is not None and self.timeout_sec <= 0.0:
            raise ValueError("timeout_sec must be positive")
        if self.max_wall_time_sec is not None and self.max_wall_time_sec <= 0.0:
            raise ValueError("max_wall_time_sec must be positive")
        names = [item.name for item in self.parameters]
        paths = [item.path for item in self.parameters]
        if len(names) != len(set(names)) or len(paths) != len(set(paths)):
            raise ValueError("Parameter names and paths must be unique")
        if self.optimizer == "cmaes" and any(
            item.enabled and (item.kind not in {"float", "int"} or item.stage != "refine")
            for item in self.parameters
        ):
            raise ValueError("CMA-ES plans may contain only numeric refine-stage parameters")
        if self.robustness_top_k < 0 or self.robustness_samples < 0 or self.robustness_beta < 0.0:
            raise ValueError("Robustness counts and beta must be non-negative")
        if self.acceptable_objective < 0.0:
            raise ValueError("acceptable_objective must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable plan."""
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> OptimizationPlan:
        """Parse an optimization plan mapping."""
        data = dict(value)
        data["parameters"] = tuple(ParameterSpec(**item) for item in data.get("parameters", ()))
        data["fidelity"] = FidelitySettings(**data.get("fidelity", {}))
        return cls(**data)
