# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Run the six-scene, three-sampler, five-seed optimizer evaluation matrix."""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Any

from ca_framework.optimization import FidelitySettings, OptimizationPlan, OptimizationRunner, ParameterSpec
from evaluation.canonical import canonical_scenes


BENCHMARK_SCENES = {
    "domino": "02_domino_wave",
    "cloth": "05_hanging_cloth",
    "fluid": "08_liquid_pour",
    "rigid_cloth": "04_cloth_ball",
    "rigid_fluid": "09_liquid_splash",
    "cloth_fluid": "11_liquid_cloth_membrane",
}
OPTIMIZERS = ("random", "tpe", "cmaes")
SEEDS = tuple(range(5))


def benchmark_plan(
    scene_key: str,
    optimizer: str,
    seed: int,
    *,
    trials: int,
    max_wall_time_sec: float,
) -> OptimizationPlan:
    """Build one equal-budget benchmark plan for a representative scene."""
    parameters = tuple(
        ParameterSpec(
            spec.name,
            spec.path,
            spec.scope,
            spec.kind,
            spec.lower,
            spec.upper,
            choices=spec.choices,
            transform=spec.transform,
            stage="refine" if optimizer == "cmaes" else "global",
        )
        for spec in _scene_parameters(scene_key)
        if optimizer != "cmaes" or spec.kind == "float"
    )
    return OptimizationPlan(
        name=f"benchmark-{scene_key}-{optimizer}-{seed}",
        optimizer=optimizer,
        parameters=parameters,
        seed=seed,
        trials=trials,
        max_wall_time_sec=max_wall_time_sec,
        fidelity=FidelitySettings(render_top_k=0),
        robustness_top_k=3,
        robustness_samples=3,
    )


def run_matrix(
    output_dir: Path,
    *,
    scene_keys: tuple[str, ...],
    optimizers: tuple[str, ...],
    seeds: tuple[int, ...],
    trials: int,
    max_wall_time_sec: float,
) -> dict[str, Any]:
    """Run studies serially for deterministic, comparable sampler evidence."""
    scenes = canonical_scenes()
    studies = []
    for scene_key in scene_keys:
        scene = scenes[BENCHMARK_SCENES[scene_key]]
        for optimizer in optimizers:
            for seed in seeds:
                directory = output_dir / scene_key / optimizer / f"seed_{seed}"
                started = time.monotonic()
                summary = OptimizationRunner(directory).run(
                    scene,
                    benchmark_plan(
                        scene_key,
                        optimizer,
                        seed,
                        trials=trials,
                        max_wall_time_sec=max_wall_time_sec,
                    ),
                )
                studies.append(
                    {
                        "scene": scene_key,
                        "optimizer": optimizer,
                        "seed": seed,
                        "wall_time_sec": time.monotonic() - started,
                        "summary": summary,
                    }
                )
    report = _aggregate(studies)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "benchmark_summary.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    return report


def _scene_parameters(scene_key: str) -> tuple[ParameterSpec, ...]:
    common_solver = ParameterSpec("substeps", "settings.substeps", "solver", "int", 2, 12)
    values = {
        "domino": (
            ParameterSpec(
                "friction", "objects.domino_0.physical_material.friction_dynamic", "material", "float", 0.1, 0.9
            ),
            ParameterSpec(
                "restitution", "objects.domino_0.physical_material.restitution", "material", "float", 0.01, 0.8
            ),
        ),
        "cloth": (
            ParameterSpec("damping", "objects.cloth.damping", "material", "float", 0.1, 8.0, transform="log"),
            ParameterSpec(
                "stretch", "objects.cloth.stretch_stiffness", "material", "float", 1.0e3, 1.0e5, transform="log"
            ),
        ),
        "fluid": (
            ParameterSpec("viscosity", "objects.water.viscosity", "material", "float", 1.0e-4, 0.1, transform="log"),
            ParameterSpec("flip", "objects.water.flip_ratio", "material", "float", 0.1, 0.99, transform="logit"),
        ),
        "rigid_cloth": (
            ParameterSpec("damping", "objects.cloth.damping", "material", "float", 0.1, 8.0, transform="log"),
            ParameterSpec(
                "compliance", "settings.rigid.contact_compliance", "solver", "float", 1.0e-9, 1.0e-3, transform="log"
            ),
        ),
        "rigid_fluid": (
            ParameterSpec("viscosity", "objects.water.viscosity", "material", "float", 1.0e-4, 0.1, transform="log"),
            ParameterSpec("relaxation", "settings.coupling.relaxation", "coupling", "float", 0.1, 0.99, transform="logit"),
        ),
        "cloth_fluid": (
            ParameterSpec("drag", "settings.coupling.cloth_fluid_drag", "coupling", "float", 0.01, 4.0, transform="log"),
            ParameterSpec(
                "permeability", "settings.coupling.cloth_permeability", "coupling", "float", 1.0e-6, 0.5, transform="log"
            ),
        ),
    }
    return (*values[scene_key], common_solver)


def _aggregate(studies: list[dict[str, Any]]) -> dict[str, Any]:
    groups = {}
    for study in studies:
        key = f"{study['scene']}:{study['optimizer']}"
        groups.setdefault(key, []).append(study)
    aggregate = {}
    for key, values in groups.items():
        objectives = [
            item["summary"]["best_feasible_trial"]["objective"]
            for item in values
            if item["summary"]["best_feasible_trial"] is not None
        ]
        feasibility = [item["summary"]["feasibility_rate"] for item in values]
        threshold_times = [
            item["summary"]["time_to_objective_threshold_sec"]
            for item in values
            if item["summary"]["time_to_objective_threshold_sec"] is not None
        ]
        robust_objectives = [
            item["summary"]["most_robust_trial"].get("robust_objective")
            for item in values
            if item["summary"]["most_robust_trial"] is not None
            and item["summary"]["most_robust_trial"].get("robust_objective") is not None
        ]
        aggregate[key] = {
            "seed_count": len(values),
            "best_objective_median": statistics.median(objectives) if objectives else None,
            "best_objective_quartiles": _quartiles(objectives),
            "feasibility_rate_median": statistics.median(feasibility),
            "time_to_objective_threshold_sec_median": (
                statistics.median(threshold_times) if threshold_times else None
            ),
            "timeout_rate": statistics.mean(item["summary"]["timeout_rate"] for item in values),
            "physics_failure_rate": statistics.mean(
                item["summary"]["physics_failure_rate"] for item in values
            ),
            "robust_objective_median": (
                statistics.median(robust_objectives) if robust_objectives else None
            ),
            "wall_time_sec_median": statistics.median(item["wall_time_sec"] for item in values),
        }
    return {"studies": studies, "aggregate": aggregate}


def _quartiles(values: list[float]) -> list[float] | None:
    if not values:
        return None
    if len(values) == 1:
        return [values[0], values[0], values[0]]
    return statistics.quantiles(values, n=4, method="inclusive")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--scene", choices=tuple(BENCHMARK_SCENES), action="append")
    parser.add_argument("--optimizer", choices=OPTIMIZERS, action="append")
    parser.add_argument("--seed", type=int, action="append")
    parser.add_argument("--trials", type=int, default=30)
    parser.add_argument("--max-wall-time-sec", type=float, default=3600.0)
    args = parser.parse_args()
    run_matrix(
        args.output_dir,
        scene_keys=tuple(args.scene or BENCHMARK_SCENES),
        optimizers=tuple(args.optimizer or OPTIMIZERS),
        seeds=tuple(args.seed or SEEDS),
        trials=args.trials,
        max_wall_time_sec=args.max_wall_time_sec,
    )


if __name__ == "__main__":
    main()
