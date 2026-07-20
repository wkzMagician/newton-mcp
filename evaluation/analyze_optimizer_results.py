# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Summarize completed Agent-authored optimizer comparisons without scene rules."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from ca_framework.metrics import MetricPipeline, ObjectiveSettings, TaskSpec


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, required=True, help="final-matrix directory")
    parser.add_argument("--output", type=Path, required=True, help="JSON analysis output")
    parser.add_argument("--table-output", type=Path, help="Optional flat 12-by-5 CSV table")
    parser.add_argument("--markdown-output", type=Path, help="Optional human-readable Markdown table")
    parser.add_argument(
        "--comparison-dir",
        action="append",
        type=Path,
        dest="comparison_dirs",
        help=(
            "Completed optimizer-comparison directory to include. Repeat to build an "
            "analysis from an explicit experiment set; paths relative to --results-root are supported."
        ),
    )
    return parser.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _usage_totals(events_path: Path) -> dict[str, int]:
    """Sum agent-reported token categories from Codex JSONL events."""
    totals = {"input_tokens": 0, "cached_input_tokens": 0, "output_tokens": 0, "reasoning_output_tokens": 0}
    if not events_path.is_file():
        return totals
    for line in events_path.read_text(encoding="utf-8").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") != "turn.completed":
            continue
        usage = event.get("usage", {})
        if not isinstance(usage, Mapping):
            continue
        for key in totals:
            value = usage.get(key, 0)
            if type(value) in {int, float}:
                totals[key] += int(value)
    return totals


def _video_metadata(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    command = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,width,height,r_frame_rate,nb_frames,duration",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(command, text=True, capture_output=True, check=False)
    if result.returncode != 0:
        return {"path": str(path), "readable": False}
    streams = json.loads(result.stdout).get("streams", [])
    return {"path": str(path), "readable": bool(streams), "stream": streams[0] if streams else None}


def _baseline(bundle: Path) -> dict[str, Any] | None:
    required = ("base_metrics.json", "task_spec.json", "objective_settings.json")
    if not all((bundle / name).is_file() for name in required):
        return None
    return _score_metrics(_read_json(bundle / "base_metrics.json"), bundle)


def _score_metrics(metrics: Mapping[str, Any], bundle: Path) -> dict[str, Any]:
    """Score one simulated trajectory with the Agent-authored task contract."""
    task = TaskSpec.from_dict(_read_json(bundle / "task_spec.json"))
    settings = ObjectiveSettings(**_read_json(bundle / "objective_settings.json"))
    losses = task.evaluate({"metrics": metrics})
    result = MetricPipeline(settings).evaluate({"metrics": metrics}, task_metrics=losses)
    return {
        "objective": result.objective,
        "raw_objective": result.metrics.get("raw_objective"),
        "feasible": result.feasible,
        "failure_reason": result.failure_reason,
        "task_losses": losses,
        "physics": metrics.get("physics", {}),
        "simulation_device": metrics.get("simulation_device"),
    }


def _trial_metrics(trial: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Read the persisted metrics for one selected optimizer trial."""
    if trial is None:
        return None
    result_dir = Path(str(trial.get("result_dir", "")))
    path = result_dir / "metrics.json"
    if not path.is_file():
        return None
    payload = _read_json(path)
    values = payload.get("metrics", {})
    return {
        "raw_objective": values.get("raw_objective"),
        "task_loss": values.get("task_loss"),
        "physics_loss": values.get("physics_loss"),
        "result_dir": str(result_dir),
    }


def _generation_method(root: Path, case_name: str, variant: str) -> dict[str, Any]:
    """Summarize one scene-generation method from the canonical matrix."""
    matrix_path = root / "summary.json"
    rows = json.loads(matrix_path.read_text(encoding="utf-8")) if matrix_path.is_file() else []
    row = next(
        (item for item in rows if item.get("case") == case_name and item.get("variant") == variant),
        None,
    )
    case_dir = root / variant / case_name
    if row is None:
        return {"status": "missing", "representative_video": None}
    output_dir = Path(str(row.get("output_dir", "")))
    return {
        "status": row.get("status"),
        "elapsed_seconds": row.get("elapsed_seconds"),
        "token_usage": _usage_totals(case_dir / "codex-events.jsonl"),
        "representative_video": _video_metadata(output_dir / "animation.mp4"),
        "physics_valid": row.get("physics_valid"),
    }


def _comparison_summaries(results_root: Path, comparison_dirs: Iterable[Path] | None = None) -> Iterable[Path]:
    if comparison_dirs is not None:
        for directory in comparison_dirs:
            path = directory if directory.is_absolute() else results_root / directory
            summary = path / "summary.json"
            if not summary.is_file():
                raise FileNotFoundError(f"Completed comparison summary is missing: {summary}")
            yield summary
        return
    for path in sorted(results_root.glob("optimizer-comparison-*/summary.json")):
        if "-pilot-" not in path.parent.name:
            yield path


def _scene_agent_events(root: Path, bundle: Path, case_name: str) -> Path:
    """Locate the exact scene-authoring Agent event stream when provenance exists."""
    provenance_path = bundle / "provenance.json"
    if provenance_path.is_file():
        try:
            source_scene = Path(str(_read_json(provenance_path)["source_scene"]))
            return source_scene.parents[3] / case_name / "codex-events.jsonl"
        except (KeyError, OSError, TypeError, ValueError):
            pass
    return root / "mcp" / case_name / "codex-events.jsonl"


def _case_row(root: Path, comparison_summary: Path, row: Mapping[str, Any]) -> dict[str, Any]:
    case_name = str(row["case"])
    bundle = Path(str(row["objective_bundle"]))
    mcp_events = _scene_agent_events(root, bundle, case_name)
    objective_events = bundle.parent / "codex-events.jsonl"
    methods: dict[str, Any] = {}
    baseline = _baseline(bundle)
    for method, summary in row.get("methods", {}).items():
        finalist = summary.get("finalist_artifacts", [])
        video = None
        representative_score = None
        if finalist:
            finalist_dir = Path(str(finalist[0]["output_dir"]))
            video = _video_metadata(finalist_dir / "animation.mp4")
            if (finalist_dir / "metrics.json").is_file():
                representative_score = _score_metrics(_read_json(finalist_dir / "metrics.json"), bundle)
        median = summary.get("median_feasible_trial")
        median_metrics = _trial_metrics(median)
        methods[method] = {
            "trial_count": summary.get("trial_count"),
            "feasibility_rate": summary.get("feasibility_rate"),
            "physics_failure_rate": summary.get("physics_failure_rate"),
            "wall_time_sec": summary.get("wall_time_sec"),
            "runtime_breakdown": summary.get("runtime_breakdown"),
            "best_feasible_trial": summary.get("best_feasible_trial"),
            "median_feasible_trial": median,
            "median_metrics": median_metrics,
            "representative_video": video,
            "search_median_improvement_fraction": (
                None
                if baseline is None
                or median_metrics is None
                or baseline["raw_objective"] in {None, 0.0}
                or median_metrics["raw_objective"] is None
                else 1.0 - float(median_metrics["raw_objective"]) / float(baseline["raw_objective"])
            ),
            "representative_score": representative_score,
            "representative_improvement_fraction": (
                None
                if baseline is None
                or representative_score is None
                or baseline["raw_objective"] in {None, 0.0}
                or representative_score["raw_objective"] is None
                else 1.0 - float(representative_score["raw_objective"]) / float(baseline["raw_objective"])
            ),
        }
    return {
        "comparison_dir": str(comparison_summary.parent),
        "case": case_name,
        "objective_bundle": str(bundle),
        "baseline": baseline,
        "generation_methods": {
            "direct-newton": _generation_method(root, case_name, "direct-newton"),
            "mcp": {
                **_generation_method(root, case_name, "mcp"),
                "objective": None if baseline is None else baseline["objective"],
                "raw_objective": None if baseline is None else baseline["raw_objective"],
                "feasible": None if baseline is None else baseline["feasible"],
            },
        },
        "token_usage": {
            "mcp_scene_agent": _usage_totals(mcp_events),
            "task_spec_agent": _usage_totals(objective_events),
        },
        "methods": methods,
    }


def build_analysis(results_root: Path, comparison_dirs: Iterable[Path] | None = None) -> dict[str, Any]:
    """Build a machine-readable summary from completed comparison summaries."""
    rows = []
    for summary_path in _comparison_summaries(results_root, comparison_dirs):
        for row in _read_json(summary_path):
            if row.get("status") == "completed":
                rows.append(_case_row(results_root, summary_path, row))
    return {"results_root": str(results_root), "comparisons": rows}


def _flat_rows(analysis: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Flatten analysis into one row per case and displayed method."""
    rows: list[dict[str, Any]] = []
    for case in analysis.get("comparisons", []):
        case_name = case["case"]
        for method, values in case.get("generation_methods", {}).items():
            video = values.get("representative_video") or {}
            rows.append(
                {
                    "case": case_name,
                    "method": method,
                    "status": values.get("status"),
                    "feasible": values.get("feasible"),
                    "raw_objective": values.get("raw_objective"),
                    "search_raw_objective": None,
                    "median_improvement_fraction": None,
                    "feasibility_rate": None,
                    "elapsed_seconds": values.get("elapsed_seconds"),
                    "video": video.get("path"),
                    "video_readable": video.get("readable"),
                }
            )
        for method, values in case.get("methods", {}).items():
            video = values.get("representative_video") or {}
            median_metrics = values.get("median_metrics") or {}
            representative_score = values.get("representative_score") or {}
            rows.append(
                {
                    "case": case_name,
                    "method": method,
                    "status": "completed",
                    "feasible": (values.get("median_feasible_trial") or {}).get("feasible"),
                    "raw_objective": representative_score.get("raw_objective"),
                    "search_raw_objective": median_metrics.get("raw_objective"),
                    "median_improvement_fraction": values.get("representative_improvement_fraction"),
                    "feasibility_rate": values.get("feasibility_rate"),
                    "elapsed_seconds": values.get("wall_time_sec"),
                    "video": video.get("path"),
                    "video_readable": video.get("readable"),
                }
            )
    return rows


def _write_table(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else [])
        if rows:
            writer.writeheader()
            writer.writerows(rows)


def _write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    columns = ("case", "method", "status", "feasible", "raw_objective", "median_improvement_fraction", "video_readable")
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        values = []
        for column in columns:
            value = row.get(column)
            if isinstance(value, float):
                value = f"{value:.6g}"
            values.append("" if value is None else str(value))
        lines.append("| " + " | ".join(values) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    """Write an analysis JSON file for all completed optimizer comparisons."""
    args = _parse_args()
    analysis = build_analysis(args.results_root.resolve(), args.comparison_dirs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(analysis, indent=2) + "\n", encoding="utf-8")
    rows = _flat_rows(analysis)
    if args.table_output is not None:
        _write_table(args.table_output, rows)
    if args.markdown_output is not None:
        _write_markdown(args.markdown_output, rows)


if __name__ == "__main__":
    main()
