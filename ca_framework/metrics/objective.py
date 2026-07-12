# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Constraint-first objective construction from simulation results."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping

from .base import ConstraintResult, MetricTerm, ObjectiveResult


@dataclass(frozen=True, slots=True)
class ObjectiveSettings:
    """Weights and feasibility thresholds for objective evaluation."""

    infeasible_base: float = 1000.0
    physics_weight: float = 1.0
    task_weight: float = 1.0
    cost_weight: float = 0.05
    reference_runtime_sec: float = 1.0
    max_penetration: float = 0.05
    max_fluid_mass_error: float = 0.1
    max_fluid_divergence: float = 10.0
    max_smoke_density_penetration: float = 0.1
    max_cloth_edge_ratio: float = 2.0
    min_cloth_area_ratio: float = 0.05
    max_cloth_area_ratio: float = 4.0
    max_impulse_balance_error: float = 0.05
    max_angular_impulse_balance_error: float = 0.05
    max_interface_velocity_residual: float = 10.0
    max_exchange_energy_error: float = 100.0
    metric_thresholds: Mapping[str, float] = field(default_factory=dict)
    metric_weights: Mapping[str, float] = field(default_factory=dict)


class MetricPipeline:
    """Evaluate hard constraints before normalized soft metric terms."""

    def __init__(self, settings: ObjectiveSettings | None = None) -> None:
        self.settings = settings or ObjectiveSettings()

    def evaluate(
        self,
        simulation: Mapping[str, Any],
        *,
        runtime_sec: float = 0.0,
        task_metrics: Mapping[str, float] | None = None,
        coupling_metrics: Mapping[str, float] | None = None,
    ) -> ObjectiveResult:
        """Build a deterministic objective from a common simulation result."""
        metrics = simulation.get("metrics", {})
        effective_coupling = coupling_metrics or metrics.get("coupling", {})
        constraints = self._constraints(simulation, effective_coupling)
        failed = [item for item in constraints if not item.satisfied]
        constraint_values = {item.name: item.value for item in constraints}
        if failed:
            violation = sum(_relative_violation(item) ** 2 for item in failed)
            return ObjectiveResult(
                feasible=False,
                objective=self.settings.infeasible_base * (1.0 + violation),
                metrics={},
                constraints=constraint_values,
                runtime_sec=runtime_sec,
                status="infeasible",
                failure_reason=",".join(item.name for item in failed),
                constraint_results=constraints,
            )

        terms = self._physics_terms(metrics)
        task_terms = [
            MetricTerm(
                name=name,
                value=float(value),
                threshold=float(self.settings.metric_thresholds.get(name, 1.0)),
                weight=float(self.settings.metric_weights.get(name, 1.0)),
            )
            for name, value in (task_metrics or {}).items()
        ]
        physics_loss = sum(term.weight * _robust(term.value / max(term.threshold, 1.0e-12)) for term in terms)
        task_loss = sum(term.weight * _robust(term.value / max(term.threshold, 1.0e-12)) for term in task_terms)
        cost = math.log1p(runtime_sec / max(self.settings.reference_runtime_sec, 1.0e-12))
        objective = (
            self.settings.task_weight * task_loss
            + self.settings.physics_weight * physics_loss
            + self.settings.cost_weight * cost
        )
        values = {term.name: term.value for term in (*terms, *task_terms)}
        values.update(runtime_cost=cost, physics_loss=physics_loss, task_loss=task_loss)
        return ObjectiveResult(
            feasible=True,
            objective=objective,
            metrics=values,
            constraints=constraint_values,
            runtime_sec=runtime_sec,
            status="completed",
            constraint_results=constraints,
        )

    def _constraints(
        self, simulation: Mapping[str, Any], coupling_metrics: Mapping[str, float]
    ) -> list[ConstraintResult]:
        metrics = simulation.get("metrics", {})
        physics = metrics.get("physics", {})
        results = [
            ConstraintResult("physics_valid", 0.0 if physics.get("valid", True) else 1.0, 0.0, physics.get("valid", True)),
            _upper("penetration", float(metrics.get("max_penetration", 0.0)), self.settings.max_penetration),
        ]
        for object_id, values in metrics.get("fluid", {}).items():
            finite = bool(values.get("finite", True))
            results.append(ConstraintResult(f"fluid.{object_id}.finite", 0.0 if finite else 1.0, 0.0, finite))
            results.append(
                _upper(
                    f"fluid.{object_id}.divergence",
                    float(values.get("peak_divergence", values.get("max_divergence", 0.0))),
                    self.settings.max_fluid_divergence,
                )
            )
            if values.get("mass_conservation_applicable", True):
                results.append(
                    _upper(
                        f"fluid.{object_id}.mass_error",
                        abs(float(values.get("relative_mass_change", 0.0))),
                        self.settings.max_fluid_mass_error,
                    )
                )
            for cloth_id, penetration in values.get("cloth_density_penetration", {}).items():
                results.append(
                    _upper(
                        f"fluid.{object_id}.cloth.{cloth_id}.density_penetration",
                        float(penetration),
                        self.settings.max_smoke_density_penetration,
                    )
                )
        for object_id, values in metrics.get("cloth", {}).items():
            finite = bool(values.get("finite", True))
            results.extend(
                [
                    ConstraintResult(f"cloth.{object_id}.finite", 0.0 if finite else 1.0, 0.0, finite),
                    _upper(
                        f"cloth.{object_id}.edge_ratio",
                        float(values.get("max_edge_length_ratio", 1.0)),
                        self.settings.max_cloth_edge_ratio,
                    ),
                    _lower(
                        f"cloth.{object_id}.min_area_ratio",
                        float(values.get("min_triangle_area_ratio", 1.0)),
                        self.settings.min_cloth_area_ratio,
                    ),
                    _upper(
                        f"cloth.{object_id}.max_area_ratio",
                        float(values.get("max_triangle_area_ratio", 1.0)),
                        self.settings.max_cloth_area_ratio,
                    ),
                    _upper(
                        f"cloth.{object_id}.flipped_triangles",
                        float(values.get("flipped_triangle_count", 0)),
                        0.0,
                    ),
                ]
            )
        if "impulse_balance_error" in coupling_metrics:
            results.append(
                _upper(
                    "coupling.impulse_balance_error",
                    float(coupling_metrics["impulse_balance_error"]),
                    self.settings.max_impulse_balance_error,
                )
            )
        if "angular_impulse_balance_error" in coupling_metrics:
            results.append(
                _upper(
                    "coupling.angular_impulse_balance_error",
                    float(coupling_metrics["angular_impulse_balance_error"]),
                    self.settings.max_angular_impulse_balance_error,
                )
            )
        if "interface_velocity_residual" in coupling_metrics:
            results.append(
                _upper(
                    "coupling.interface_velocity_residual",
                    float(coupling_metrics["interface_velocity_residual"]),
                    self.settings.max_interface_velocity_residual,
                )
            )
        if "exchange_energy_error" in coupling_metrics:
            results.append(
                _upper(
                    "coupling.exchange_energy_error",
                    float(coupling_metrics["exchange_energy_error"]),
                    self.settings.max_exchange_energy_error,
                )
            )
        if coupling_metrics.get("coupling_nonconvergence", False):
            results.append(ConstraintResult("coupling.nonconvergence", 1.0, 0.0, False))
        return results

    def _physics_terms(self, metrics: Mapping[str, Any]) -> list[MetricTerm]:
        terms = [
            MetricTerm("penetration", float(metrics.get("max_penetration", 0.0)), self.settings.max_penetration)
        ]
        for object_id, values in metrics.get("fluid", {}).items():
            terms.append(
                MetricTerm(
                    f"fluid.{object_id}.divergence",
                    float(values.get("peak_divergence", values.get("max_divergence", 0.0))),
                    self.settings.max_fluid_divergence,
                )
            )
            if values.get("mass_conservation_applicable", True):
                terms.append(
                    MetricTerm(
                        f"fluid.{object_id}.mass_error",
                        abs(float(values.get("relative_mass_change", 0.0))),
                        self.settings.max_fluid_mass_error,
                    )
                )
            for cloth_id, penetration in values.get("cloth_density_penetration", {}).items():
                terms.append(
                    MetricTerm(
                        f"fluid.{object_id}.cloth.{cloth_id}.density_penetration",
                        float(penetration),
                        self.settings.max_smoke_density_penetration,
                    )
                )
        for object_id, values in metrics.get("cloth", {}).items():
            terms.append(
                MetricTerm(
                    f"cloth.{object_id}.residual_speed",
                    float(values.get("final_residual_speed", 0.0)),
                    float(self.settings.metric_thresholds.get("cloth_residual_speed", 1.0)),
                )
            )
        return terms


def _robust(value: float) -> float:
    value = abs(value)
    return 0.5 * value * value if value <= 1.0 else value - 0.5


def _upper(name: str, value: float, threshold: float) -> ConstraintResult:
    return ConstraintResult(name, value, threshold, math.isfinite(value) and value <= threshold)


def _lower(name: str, value: float, threshold: float) -> ConstraintResult:
    return ConstraintResult(name, value, threshold, math.isfinite(value) and value >= threshold)


def _relative_violation(item: ConstraintResult) -> float:
    if item.threshold > 0.0:
        return max(0.0, item.value / item.threshold - 1.0)
    return max(1.0, abs(item.value))
