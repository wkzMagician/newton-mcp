# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Run Random, TPE, and CMA-ES on Agent-authored MCP base scenes."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from ca_framework.metrics import MetricPipeline, ObjectiveSettings, TaskSpec
from ca_framework.optimization import OptimizationPlan, run_optimizer_suite
from ca_framework.scene import Scene, SceneExecutorLocal
from evaluation.objective_contract import validate_objective_settings, validate_task_semantics, validate_task_spec


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mcp-run-dir", type=Path, required=True, help="Completed MCP generation run directory")
    parser.add_argument("--objective-dir", type=Path, help="Objective-Agent outputs; defaults below --mcp-run-dir")
    parser.add_argument("--case", action="append", dest="cases")
    parser.add_argument(
        "--optimizer",
        action="append",
        choices=("random", "tpe", "cmaes"),
        dest="optimizers",
        help="Run only the selected optimizer; repeat for multiple methods.",
    )
    parser.add_argument("--output-dir", type=Path, help="New directory for optimizer studies")
    parser.add_argument("--trials", type=int, help="Override the Agent-authored per-method trial budget")
    parser.add_argument("--render-top-k", type=int, default=1, help="Final feasible candidates to render per method")
    parser.add_argument(
        "--skip-f1",
        action="store_true",
        help="Skip the diagnostic-only low-fidelity prepass; full-fidelity F2 scoring is unchanged.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse completed cases already present in --output-dir.",
    )
    return parser.parse_args()


def _write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _objective_bundle(root: Path, case_name: str) -> Path:
    bundle = root / case_name / "optimization-objective"
    required = (
        "scene.json",
        "optimization_plan.json",
        "task_spec.json",
        "objective_settings.json",
        "base_metrics.json",
    )
    if not bundle.is_dir() or not all((bundle / name).is_file() for name in required):
        raise FileNotFoundError(f"Complete Agent-authored objective bundle is missing for {case_name}: {bundle}")
    return bundle


def _completed_output(output_dir: Path, optimizers: tuple[str, ...] = ("random", "tpe", "cmaes")) -> bool:
    for method in optimizers:
        summary_path = output_dir / method / "optimization_summary.json"
        if not summary_path.is_file():
            return False
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        finalists = summary.get("finalist_artifacts", [])
        if not finalists or not (Path(str(finalists[0]["output_dir"])) / "animation.mp4").is_file():
            return False
    return True


def _run_case(
    case_name: str,
    *,
    bundle: Path,
    output_dir: Path,
    trials: int | None,
    render_top_k: int,
    skip_f1: bool,
    optimizers: tuple[str, ...],
) -> dict[str, Any]:
    scene = Scene.from_dict(json.loads((bundle / "scene.json").read_text(encoding="utf-8")))
    plan = OptimizationPlan.from_dict(json.loads((bundle / "optimization_plan.json").read_text(encoding="utf-8")))
    task_spec = TaskSpec.from_dict(json.loads((bundle / "task_spec.json").read_text(encoding="utf-8")))
    base_metrics = json.loads((bundle / "base_metrics.json").read_text(encoding="utf-8"))
    objective_settings_data = json.loads((bundle / "objective_settings.json").read_text(encoding="utf-8"))
    errors = [
        *validate_task_spec(task_spec, base_metrics),
        *validate_task_semantics(task_spec, base_metrics),
        *validate_objective_settings(task_spec, objective_settings_data, base_metrics),
    ]
    if errors:
        raise ValueError("Invalid Agent-authored task_spec: " + "; ".join(errors))
    objective_settings = ObjectiveSettings(**objective_settings_data)
    fidelity = replace(plan.fidelity, render_top_k=render_top_k)
    if skip_f1:
        fidelity = replace(fidelity, short_duration_fraction=1.0, low_resolution_scale=1.0)
    plan = replace(plan, fidelity=fidelity)
    summaries = run_optimizer_suite(
        scene,
        plan,
        output_dir,
        optimizers=optimizers,
        trials=trials,
        executor=SceneExecutorLocal(),
        metrics=MetricPipeline(objective_settings),
        task_metrics=task_spec,
    )
    return {
        "case": case_name,
        "status": "completed",
        "objective_bundle": str(bundle),
        "base_scene": str(bundle / "scene.json"),
        "task_spec": task_spec.to_dict(),
        "objective_settings": asdict(objective_settings),
        "methods": summaries,
    }


def _completed_case_row(
    case_name: str,
    *,
    bundle: Path,
    output_dir: Path,
    optimizers: tuple[str, ...],
) -> dict[str, Any]:
    """Rebuild a suite row from completed method artifacts without rerunning."""
    task_spec = TaskSpec.from_dict(json.loads((bundle / "task_spec.json").read_text(encoding="utf-8")))
    objective_settings = ObjectiveSettings(
        **json.loads((bundle / "objective_settings.json").read_text(encoding="utf-8"))
    )
    methods = {
        method: json.loads((output_dir / method / "optimization_summary.json").read_text(encoding="utf-8"))
        for method in optimizers
    }
    return {
        "case": case_name,
        "status": "completed",
        "objective_bundle": str(bundle),
        "base_scene": str(bundle / "scene.json"),
        "task_spec": task_spec.to_dict(),
        "objective_settings": asdict(objective_settings),
        "methods": methods,
        "resumed": True,
    }


def main() -> None:
    """Run every selected optimizer suite serially on immutable MCP scenes."""
    args = _parse_args()
    if args.trials is not None and args.trials <= 0:
        raise ValueError("--trials must be positive")
    if args.render_top_k < 0:
        raise ValueError("--render-top-k cannot be negative")
    mcp_run_dir = args.mcp_run_dir.resolve()
    objective_root = (args.objective_dir or mcp_run_dir / "optimization-objectives").resolve()
    output_root = (args.output_dir or mcp_run_dir / "optimizer-suite").resolve()
    output_root.mkdir(parents=True, exist_ok=args.resume)
    cases = args.cases or sorted(path.name for path in objective_root.iterdir() if path.is_dir())
    optimizers = tuple(dict.fromkeys(args.optimizers or ("random", "tpe", "cmaes")))
    rows: list[dict[str, Any]] = []
    for case_name in cases:
        if args.resume and _completed_output(output_root / case_name, optimizers):
            row = _completed_case_row(
                case_name,
                bundle=_objective_bundle(objective_root, case_name),
                output_dir=output_root / case_name,
                optimizers=optimizers,
            )
        else:
            if (output_root / case_name).exists() and not args.resume:
                raise FileExistsError(
                    f"Incomplete optimizer output exists for {case_name}; move it aside before resuming: "
                    f"{output_root / case_name}"
                )
            try:
                row = _run_case(
                    case_name,
                    bundle=_objective_bundle(objective_root, case_name),
                    output_dir=output_root / case_name,
                    trials=args.trials,
                    render_top_k=args.render_top_k,
                    skip_f1=args.skip_f1,
                    optimizers=optimizers,
                )
            except Exception as error:
                row = {"case": case_name, "status": "failed", "error": f"{type(error).__name__}: {error}"}
        rows.append(row)
        _write_summary(output_root / "summary.json", rows)
        print(f"{case_name}: {row['status']}", flush=True)
    if any(row["status"] != "completed" for row in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
