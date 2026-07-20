# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Repeat selected baseline and finalist scenes for uncertainty estimates."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import warp as wp

from ca_framework.metrics import MetricPipeline, ObjectiveSettings, TaskSpec
from ca_framework.scene import Scene, SceneExecutorLocal

METHODS = ("mcp", "random", "tpe", "cmaes")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True, help="final-matrix directory")
    parser.add_argument("--case", action="append", required=True, dest="cases")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    return parser.parse_args()


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _method_scene(results_root: Path, case_name: str, method: str, bundle: Path) -> Path:
    if method == "mcp":
        return bundle / "scene.json"
    summary = _read(
        results_root / "optimization" / "optimizer-suite" / case_name / method / "optimization_summary.json"
    )
    selected = summary.get("median_feasible_trial")
    if selected is None:
        raise ValueError(f"No median feasible trial for {case_name}/{method}")
    return Path(str(selected["result_dir"])) / "scene.json"


def summarize_repeats(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize feasible raw objectives without hiding failed repeats."""
    values = [float(row["raw_objective"]) for row in rows if row["feasible"]]
    return {
        "repeat_count": len(rows),
        "feasible_count": len(values),
        "feasibility_rate": len(values) / len(rows) if rows else 0.0,
        "raw_objective_mean": float(np.mean(values)) if values else None,
        "raw_objective_std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0 if values else None,
        "raw_objective_median": float(np.median(values)) if values else None,
        "raw_objectives": values,
    }


def _run_case(results_root: Path, output_root: Path, case_name: str, repeats: int) -> dict[str, Any]:
    bundle = results_root / "optimization" / "objectives" / case_name / "optimization-objective"
    task = TaskSpec.from_dict(_read(bundle / "task_spec.json"))
    settings = ObjectiveSettings(**_read(bundle / "objective_settings.json"))
    pipeline = MetricPipeline(settings)
    case_result: dict[str, Any] = {"case": case_name, "repeat_count": repeats, "methods": {}}
    for method in METHODS:
        scene_path = _method_scene(results_root, case_name, method, bundle)
        scene = Scene.from_dict(_read(scene_path))
        rows = []
        for repeat in range(repeats):
            repeat_dir = output_root / case_name / method / f"repeat_{repeat:02d}"
            repeat_dir.mkdir(parents=True, exist_ok=False)
            started = time.perf_counter()
            simulation = SceneExecutorLocal().simulate(scene, capture_cache=False)
            runtime = time.perf_counter() - started
            task_losses = task.evaluate(simulation)
            objective = pipeline.evaluate(simulation, runtime_sec=runtime, task_metrics=task_losses)
            (repeat_dir / "scene.json").write_text(json.dumps(scene.to_dict(), indent=2) + "\n", encoding="utf-8")
            (repeat_dir / "simulation_metrics.json").write_text(
                json.dumps(simulation["metrics"], indent=2) + "\n", encoding="utf-8"
            )
            (repeat_dir / "metrics.json").write_text(json.dumps(objective.to_dict(), indent=2) + "\n", encoding="utf-8")
            rows.append(
                {
                    "repeat": repeat,
                    "feasible": objective.feasible,
                    "raw_objective": objective.metrics.get("raw_objective"),
                    "runtime_sec": runtime,
                    "failure_reason": objective.failure_reason,
                    "result_dir": str(repeat_dir),
                }
            )
        case_result["methods"][method] = {**summarize_repeats(rows), "repeats": rows}
    return case_result


def main() -> None:
    """Run selected robustness repeats serially and write a durable summary."""
    args = _parse_args()
    if args.repeats < 2:
        raise ValueError("--repeats must be at least 2 for an uncertainty estimate")
    wp.set_device(args.device)
    results_root = args.results_root.resolve()
    output_root = args.output_dir.resolve()
    output_root.mkdir(parents=True, exist_ok=False)
    rows = []
    for case_name in args.cases:
        row = _run_case(results_root, output_root, case_name, args.repeats)
        rows.append(row)
        (output_root / "summary.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
        print(f"{case_name}: completed", flush=True)


if __name__ == "__main__":
    main()
