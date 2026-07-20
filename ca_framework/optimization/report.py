# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Study-independent optimization history and candidate reporting."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def build_report(
    output_dir: str | Path,
    robust_scores: dict[int, float] | None = None,
    *,
    acceptable_objective: float = 100.0,
) -> dict[str, Any]:
    """Build required final selections and histories from durable trials."""
    trials = []
    terminal_results = []
    for path in sorted(Path(output_dir).glob("trial_*")):
        if not (path / "result.json").exists():
            continue
        result = json.loads((path / "result.json").read_text(encoding="utf-8"))
        terminal_results.append(result)
        if not (path / "metrics.json").exists() or "objective" not in result:
            continue
        metrics = json.loads((path / "metrics.json").read_text(encoding="utf-8"))
        parameters = json.loads((path / "parameters.json").read_text(encoding="utf-8"))
        trials.append(
            {
                "number": int(path.name.removeprefix("trial_")),
                "objective": float(result["objective"]),
                "feasible": bool(result["feasible"]),
                "runtime_sec": float(metrics.get("runtime_sec", 0.0)),
                "parameters": parameters,
                "result_dir": str(path),
            }
        )
    feasible = [item for item in trials if item["feasible"]]
    best = min(feasible, key=lambda item: item["objective"], default=None)
    median = None
    if feasible:
        ordered_feasible = sorted(feasible, key=lambda item: (item["objective"], item["number"]))
        median = ordered_feasible[(len(ordered_feasible) - 1) // 2]
    acceptable = [item for item in feasible if item["objective"] <= acceptable_objective]
    fastest = min(acceptable, key=lambda item: item["runtime_sec"], default=None)
    robust_scores = robust_scores or {}
    robust = min(
        (item for item in feasible if item["number"] in robust_scores),
        key=lambda item: robust_scores[item["number"]],
        default=best,
    )
    if robust is not None and robust["number"] in robust_scores:
        robust = {**robust, "robust_objective": robust_scores[robust["number"]]}
    best_so_far = []
    current = float("inf")
    feasible_count = 0
    violation_history = []
    cumulative_runtime = 0.0
    time_to_threshold = None
    for index, item in enumerate(trials, start=1):
        cumulative_runtime += item["runtime_sec"]
        if item["feasible"]:
            current = min(current, item["objective"])
            feasible_count += 1
            if time_to_threshold is None and item["objective"] <= acceptable_objective:
                time_to_threshold = cumulative_runtime
        best_so_far.append(None if not np.isfinite(current) else current)
        violation_history.append(0.0 if item["feasible"] else max(0.0, item["objective"] - 1000.0))
    report = {
        "best_feasible_trial": best,
        "median_feasible_trial": median,
        "fastest_acceptable_trial": fastest,
        "most_robust_trial": robust,
        "optimization_history": best_so_far,
        "constraint_violation_history": violation_history,
        "feasibility_rate": feasible_count / len(trials) if trials else 0.0,
        "timeout_rate": (
            sum(item.get("status") == "timeout" for item in terminal_results) / len(terminal_results)
            if terminal_results
            else 0.0
        ),
        "physics_failure_rate": (
            sum(
                any(
                    token in str(item.get("failure_reason", ""))
                    for token in ("physics_valid", "cloth.", "fluid.", "coupling.", "penetration")
                )
                for item in terminal_results
            )
            / len(terminal_results)
            if terminal_results
            else 0.0
        ),
        "time_to_objective_threshold_sec": time_to_threshold,
        "acceptable_objective": acceptable_objective,
        "parameter_importance": _parameter_importance(feasible),
        "runtime_breakdown": {
            "total_trial_runtime_sec": sum(item["runtime_sec"] for item in trials),
            "median_trial_runtime_sec": float(np.median([item["runtime_sec"] for item in trials])) if trials else 0.0,
        },
    }
    _write_history_svg(Path(output_dir) / "best_so_far.svg", best_so_far)
    return report


def _parameter_importance(trials: list[dict[str, Any]]) -> dict[str, float]:
    if len(trials) < 2:
        return {}
    objectives = np.asarray([item["objective"] for item in trials], dtype=float)
    importance = {}
    common = set.intersection(*(set(item["parameters"]) for item in trials)) if trials else set()
    for name in sorted(common):
        values = [item["parameters"][name] for item in trials]
        if not all(type(value) in {int, float} for value in values):
            continue
        numeric = np.asarray(values, dtype=float)
        if np.std(numeric) <= 1.0e-12 or np.std(objectives) <= 1.0e-12:
            importance[name] = 0.0
        else:
            importance[name] = abs(float(np.corrcoef(numeric, objectives)[0, 1]))
    total = sum(importance.values())
    return {name: value / total for name, value in importance.items()} if total > 0.0 else importance


def _write_history_svg(path: Path, history: list[float | None]) -> None:
    """Write a dependency-free best-so-far curve suitable for reports."""
    width, height, margin = 640, 360, 48
    points = [(index, value) for index, value in enumerate(history) if value is not None]
    if points:
        values = [float(value) for _, value in points]
        low, high = min(values), max(values)
        span = max(high - low, 1.0e-12)
        x_span = max(len(history) - 1, 1)
        coordinates = " ".join(
            f"{margin + index / x_span * (width - 2 * margin):.2f},"
            f"{height - margin - (float(value) - low) / span * (height - 2 * margin):.2f}"
            for index, value in points
        )
        line = f'<polyline points="{coordinates}" fill="none" stroke="#2563eb" stroke-width="3"/>'
    else:
        line = ""
    svg = (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">\n'
        '<rect width="100%" height="100%" fill="white"/>\n'
        f'<path d="M {margin} {margin} V {height - margin} H {width - margin}" '
        'fill="none" stroke="#475569"/>\n'
        f'{line}\n'
        f'<text x="{width / 2}" y="{height - 10}" text-anchor="middle">Trial</text>\n'
        '<text x="16" y="180" text-anchor="middle" transform="rotate(-90 16 180)">Best feasible objective</text>\n'
        "</svg>\n"
    )
    path.write_text(svg, encoding="utf-8")
