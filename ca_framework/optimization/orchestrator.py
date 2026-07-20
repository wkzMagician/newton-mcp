# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Independent Random, TPE, and CMA-ES optimizer suite orchestration."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from ca_framework.metrics import MetricPipeline
from ca_framework.scene import Scene, SceneExecutorLocal

from .model import OptimizationPlan, ParameterSpec
from .parameter_space import activate_parameters
from .runner import OptimizationRunner, TaskMetrics


def run_optimizer_suite(
    base_scene: Scene,
    plan: OptimizationPlan,
    output_dir: str | Path,
    *,
    optimizers: tuple[str, ...] = ("random", "tpe", "cmaes"),
    trials: int | None = None,
    executor: SceneExecutorLocal | None = None,
    metrics: MetricPipeline | None = None,
    task_metrics: TaskMetrics | None = None,
) -> dict[str, Any]:
    """Run independent optimizer studies with the same bounded budget.

    Args:
        base_scene: Scene used as the immutable baseline for every optimizer.
        plan: Base optimization plan containing the candidate parameter space.
        output_dir: Directory where per-optimizer study artifacts are written.
        optimizers: Optimizers to run independently.
        trials: Optional per-optimizer trial count override.
        executor: Optional scene executor shared by all studies.
        metrics: Optional objective evaluator shared by all studies.
        task_metrics: Optional task metric evaluator.

    Returns:
        Mapping from optimizer name to each study summary.
    """
    root = Path(output_dir).resolve()
    summaries = {}
    effective_trials = trials if trials is not None else plan.trials
    for optimizer in optimizers:
        parameters = _parameters_for_optimizer(plan.parameters, optimizer)
        active_parameters = activate_parameters(base_scene, parameters, optimizer=optimizer)
        active_paths = {item.path for item in active_parameters}
        filtered_initial_parameters = []
        for candidate in plan.initial_parameters:
            filtered = {path: value for path, value in candidate.items() if path in active_paths}
            if filtered:
                filtered_initial_parameters.append(filtered)
        initial_parameters = tuple(filtered_initial_parameters)
        if not initial_parameters:
            initial_parameters = _baseline_initial_parameters(base_scene, active_parameters)
        optimizer_plan = replace(
            plan,
            name=f"{plan.name}-{optimizer}",
            study_name=f"{plan.study_name or plan.name}-{optimizer}",
            optimizer=optimizer,
            parameters=parameters,
            initial_parameters=initial_parameters,
            trials=effective_trials,
            # A small benchmark budget must still contain a model-guided TPE
            # phase instead of spending every trial on random initialization.
            tpe_startup_trials=min(plan.tpe_startup_trials, max(1, effective_trials // 2)),
            fidelity=plan.fidelity,
            robustness_top_k=0,
            robustness_samples=0,
        )
        summaries[optimizer] = OptimizationRunner(root / optimizer, executor=executor, metrics=metrics).run(
            base_scene, optimizer_plan, task_metrics=task_metrics
        )
    return summaries


def _baseline_initial_parameters(
    scene: Scene, parameters: tuple[ParameterSpec, ...]
) -> tuple[dict[str, Any], ...]:
    """Return a baseline control only when every active value is inside its search space."""
    payload: Any = scene.to_dict()
    candidate: dict[str, Any] = {}
    try:
        for spec in parameters:
            if not spec.enabled:
                continue
            value: Any = payload
            for part in spec.path.split("."):
                value = value[int(part)] if isinstance(value, list) else value[part]
            spec.validate_value(value)
            candidate[spec.path] = value
    except (IndexError, KeyError, TypeError, ValueError):
        return ()
    return (candidate,) if candidate else ()


def _parameters_for_optimizer(parameters: tuple[ParameterSpec, ...], optimizer: str) -> tuple[ParameterSpec, ...]:
    if optimizer != "cmaes":
        return tuple(replace(item, stage="global") for item in parameters)
    return tuple(
        replace(item, stage="refine")
        for item in parameters
        if item.kind in {"float", "int"}
    )
