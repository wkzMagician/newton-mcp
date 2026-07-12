# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Fixed Random baseline and TPE-to-CMA two-stage orchestration."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

from ca_framework.scene import Scene, SceneExecutorLocal

from .model import OptimizationPlan
from .patch import apply_parameter_patch
from .runner import OptimizationRunner, TaskMetrics


def run_two_stage(
    base_scene: Scene,
    plan: OptimizationPlan,
    output_dir: str | Path,
    *,
    global_trials: int,
    refine_trials: int,
    random_baseline_trials: int | None = None,
    executor: SceneExecutorLocal | None = None,
    task_metrics: TaskMetrics | None = None,
) -> dict[str, Any]:
    """Run equal-format Random/TPE global studies and CMA continuous refinement."""
    root = Path(output_dir).resolve()
    global_parameters = tuple(item for item in plan.parameters if item.stage == "global")
    refine_parameters = tuple(item for item in plan.parameters if item.stage == "refine")
    tpe_plan = replace(
        plan,
        name=f"{plan.name}-tpe",
        study_name=f"{plan.study_name or plan.name}-tpe",
        optimizer="tpe",
        parameters=global_parameters,
        trials=global_trials,
        fidelity=replace(plan.fidelity, render_top_k=0),
    )
    tpe = OptimizationRunner(root / "tpe", executor=executor).run(
        base_scene, tpe_plan, task_metrics=task_metrics
    )
    best = tpe.get("best_feasible_trial")
    refined_base = (
        apply_parameter_patch(base_scene, best["parameters"]) if best is not None else base_scene
    )
    cma = None
    if refine_parameters and best is not None:
        cma_plan = replace(
            plan,
            name=f"{plan.name}-cmaes",
            study_name=f"{plan.study_name or plan.name}-cmaes",
            optimizer="cmaes",
            parameters=refine_parameters,
            trials=refine_trials,
        )
        cma = OptimizationRunner(root / "cmaes", executor=executor).run(
            refined_base, cma_plan, task_metrics=task_metrics
        )
    baseline = None
    if random_baseline_trials:
        random_plan = replace(
            plan,
            name=f"{plan.name}-random",
            study_name=f"{plan.study_name or plan.name}-random",
            optimizer="random",
            parameters=global_parameters,
            trials=random_baseline_trials,
            fidelity=replace(plan.fidelity, render_top_k=0),
        )
        baseline = OptimizationRunner(root / "random", executor=executor).run(
            base_scene, random_plan, task_metrics=task_metrics
        )
    return {"tpe": tpe, "cmaes": cma, "random": baseline}
