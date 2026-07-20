# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Evaluator-only validation for Agent-authored optimization objectives."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from ca_framework.metrics import MetricPipeline, ObjectiveSettings, TaskSpec


def metric_paths(metrics: Mapping[str, Any]) -> list[str]:
    """Return every numeric task path available from one simulation metrics mapping."""
    result: list[str] = []

    def visit(value: Any, parts: list[str]) -> None:
        if isinstance(value, Mapping):
            for key, child in value.items():
                visit(child, [*parts, str(key)])
        elif type(value) in {int, float}:
            result.append("metrics." + ".".join(parts))

    visit(metrics, [])
    return sorted(result)


def physics_metric_semantics(metrics: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Describe framework-owned physical constraint measurements for an Agent.

    The values are copied from the completed baseline.  This metadata is
    intentionally generic: it documents which metric path each hard physics
    check reads, without prescribing a scene-specific threshold or objective.
    """
    result: list[dict[str, Any]] = []
    fluids = metrics.get("fluid", {})
    if not isinstance(fluids, Mapping):
        return result
    for object_id, values in fluids.items():
        if not isinstance(values, Mapping):
            continue
        if values.get("divergence_constraint_applicable", True) is False:
            continue
        if "peak_divergence" in values:
            path = f"metrics.fluid.{object_id}.peak_divergence"
            value = values["peak_divergence"]
        elif "max_divergence" in values:
            path = f"metrics.fluid.{object_id}.max_divergence"
            value = values["max_divergence"]
        else:
            continue
        result.append(
            {
                "constraint": f"fluid.{object_id}.divergence",
                "path": path,
                "baseline_value": value,
                "meaning": "maximum fluid divergence over the simulated trajectory",
            }
        )
    return result


def validate_task_spec(task_spec: TaskSpec, metrics: Mapping[str, Any]) -> list[str]:
    """Return semantic errors when a task objective cannot read baseline metrics."""
    if not task_spec.metrics:
        return ["task_spec.metrics must not be empty"]
    errors = []
    for item in task_spec.metrics:
        if not item.path.startswith("metrics."):
            errors.append(f"{item.name}: task metric path must start with 'metrics.': {item.path}")
    if errors:
        return errors
    available = set(metric_paths(metrics))
    for item in task_spec.metrics:
        if item.path not in available:
            errors.append(f"Unknown task metric path: {item.path!r}")
    return errors


def validate_task_semantics(task_spec: TaskSpec, metrics: Mapping[str, Any] | None = None) -> list[str]:
    """Reject task objectives that have no physical meaning for the baseline result.

    A task contract may only use a mass-conservation metric when the simulator
    explicitly reports that conservation is applicable for that fluid phase.
    This is a metric-level semantic rule, independent of a scene, object name,
    or target value.
    """
    errors: list[str] = []
    paths = [item.path for item in task_spec.metrics]
    forbidden = (
        "metrics.resources.",
        "metrics.timings.",
        "metrics.peak_device_memory_bytes",
        "metrics.physics.",
    )
    for path in paths:
        if path.startswith(forbidden):
            errors.append(
                f"Task metric path is framework validity/resource/runtime metadata, not the requested outcome: {path}"
            )
    if metrics is None:
        return errors

    try:
        baseline_losses = task_spec.evaluate({"metrics": metrics})
    except ValueError:
        # Path existence is reported precisely by validate_task_spec().
        return errors
    if baseline_losses and all(loss == 0.0 for loss in baseline_losses.values()):
        errors.append(
            "task_spec has zero loss for every metric on the baseline, so it cannot measure an optimization improvement"
        )

    fluids = metrics.get("fluid", {})
    if not isinstance(fluids, Mapping):
        return errors
    for item in task_spec.metrics:
        parts = item.path.split(".")
        if len(parts) != 4 or parts[:2] != ["metrics", "fluid"] or parts[-1] != "relative_mass_change":
            continue
        phase_metrics = fluids.get(parts[2])
        if isinstance(phase_metrics, Mapping) and phase_metrics.get("mass_conservation_applicable") is False:
            errors.append(
                f"Task metric path is not applicable because mass conservation is disabled for that fluid phase: "
                f"{item.path}"
            )

    return errors


def validate_objective_settings(
    task_spec: TaskSpec,
    settings_data: Mapping[str, Any],
    metrics: Mapping[str, Any] | None = None,
) -> list[str]:
    """Return errors for settings that would silently fail to affect the objective."""
    try:
        settings = ObjectiveSettings(**settings_data)
    except (TypeError, ValueError) as error:
        return [f"invalid objective_settings: {type(error).__name__}: {error}"]

    errors: list[str] = []
    if settings.minimum_coupling_contact_duration:
        errors.append(
            "minimum_coupling_contact_duration is a scene-pair-specific hard constraint; "
            "represent requested contact behavior only through Agent-authored task_spec metrics"
        )
    available_names = {item.name for item in task_spec.metrics}
    for field_name, values in (
        ("metric_thresholds", settings.metric_thresholds),
        ("metric_weights", settings.metric_weights),
    ):
        for key in values:
            if key == "cloth_residual_speed":
                continue
            if key not in available_names:
                errors.append(f"{field_name} key {key!r} must name a task_spec metric, not a metrics path")
            elif float(values[key]) <= 0.0:
                errors.append(f"{field_name} value for {key!r} must be positive")
    if metrics is not None and metrics.get("physics", {}).get("valid", False):
        baseline = MetricPipeline(settings).evaluate({"metrics": metrics})
        if not baseline.feasible:
            errors.append(
                "objective_settings reject the framework-valid baseline as infeasible: "
                f"{baseline.failure_reason}; loosen generic feasibility thresholds and express visible quality "
                "defects through task_spec"
            )
        for constraint in baseline.constraint_results:
            if constraint.value <= 0.0 or constraint.threshold <= 0.0:
                continue
            if constraint.name.endswith(".min_area_ratio"):
                has_margin = constraint.threshold <= 0.9 * constraint.value
            elif constraint.name.endswith(".penetration_fraction") and constraint.threshold >= 1.0:
                # Fractions cannot exceed their natural upper bound, so 1.0
                # already accepts every possible repeat of the baseline.
                has_margin = True
            else:
                has_margin = constraint.threshold >= 1.1 * constraint.value
            if not has_margin:
                errors.append(
                    "objective_settings hard gate has less than 10 percent baseline reproduction "
                    f"margin: {constraint.name} value={constraint.value:g}, "
                    f"threshold={constraint.threshold:g}; keep visible quality targets in task_spec"
                )
    return errors
