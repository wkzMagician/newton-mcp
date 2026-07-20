# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Ask an MCP Agent to define an optimization objective for an existing scene."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import time
from pathlib import Path
from typing import Any

from ca_framework.metrics import TaskSpec
from ca_framework.scene import Scene, SceneStore
from evaluation.agent_prompts import AGENT_PROMPTS
from evaluation.objective_contract import (
    metric_paths,
    physics_metric_semantics,
    validate_objective_settings,
    validate_task_semantics,
    validate_task_spec,
)
from evaluation.run_agent_experiments import _copy_objective_bundle, _prepare_codex_home, _register_mcp


def _prompt(
    case_name: str,
    plan_name: str,
    *,
    repair_errors: list[str] | None = None,
    pilot_available: bool = False,
) -> str:
    repair = ""
    if repair_errors:
        repair = f"""
The prior contract was rejected for these exact reasons:
{chr(10).join(f"- {error}" for error in repair_errors)}
Replace the existing plan with `overwrite=true` and correct every issue.
"""
    pilot = ""
    if pilot_available:
        pilot = """
A prior generic pilot is available in `pilot_evidence.json`, with any representative videos named
`pilot_<method>.mp4`. Read the evidence and inspect every available pilot video. Treat it as empirical
feedback about the prior search space: if feasible medians worsened or every candidate failed, revise
the causal parameter choices or bounds instead of merely increasing the trial count. Do not overfit a
single best trial and do not encode case-specific pilot values as framework constraints.
"""
    return f"""Use ca-scene MCP tools to define an optimization objective for an existing animation scene.
Do not alter the scene, do not create a new scene, do not run the animation, and do not start an optimization job.

Requested animation semantics:
{AGENT_PROMPTS[case_name]}

The existing scene is named `{case_name}`. Before creating a plan, you MUST read the local read-only files `base_metric_paths.json` and `physics_metric_semantics.json`; this file access is explicitly permitted and is required. The metric-path file is the exhaustive list of numeric paths you may use. A task_spec path must be copied exactly from that list, including its `metrics.` prefix. The physics-semantics file documents the exact baseline measurement used by each framework-owned physical feasibility gate. Never probe or infer metric paths through plan validation.

Visual evidence is available as the read-only `baseline.mp4` plus `baseline_start.png`, `baseline_middle.png`, and `baseline_end.png`. You MUST inspect the video and all three images with the available visual tools before creating a plan. State in your final message which visible failure or remaining quality gap you found and how every selected parameter can affect it. If the baseline has no clear failure, say so and optimize a genuine measurable quality margin without contradicting the requested animation. Do not infer visual success from scene JSON or scalar metrics alone.
{pilot}
{repair}

Requirements:
1. Inspect the baseline video and its start, middle, and end images, then call get_capabilities, get_scene_schema, get_scene, and validate_scene for `{case_name}`.
2. Create exactly one plan named `{plan_name}`. Choose one supported optimizer and the smallest trial count you judge sufficient to find a single feasible quality improvement; do not create a method comparison or parameter grid.
3. Select two to four relevant bounded physical or solver parameters. When the visual evidence identifies excessive separation or spatial misalignment as a cause of failure, include the corresponding bounded task-object transform component and choose its direction and bounds yourself from that evidence; do not compensate for a diagnosed geometry problem only by increasing force or velocity. Do not optimize the camera, duration, or visual materials. Parameter paths are dot-separated; use zero-based numeric path components to address list values, for example an emitter velocity component. Read `get_capabilities` for the exact path contract.
   Search bounds must follow the visible causal diagnosis. In particular, when an object starts visibly centered or aligned, a transform search interval must straddle the current value rather than move in only one arbitrary direction. Exclude the current value only when the video shows a clear directional misalignment and the interval covers the diagnosed corrected region. When all current parameter values lie inside the bounds, include that exact baseline parameter mapping in `initial_parameters` as a control candidate.
4. Supply a non-empty task_spec that quantitatively represents the requested visible outcome. Every task_spec path must be copied exactly from `base_metric_paths.json`; do not invent metrics. In `objective_settings.metric_thresholds` and `metric_weights`, keys must be task_spec metric *names* (or omit them); never use a `metrics.*` path as a key. Supply objective_settings that make unstable, penetrating, or otherwise invalid physics infeasible.
   If `base_metrics.json` reports `physics.valid=true`, the complete objective_settings hard gates MUST accept that baseline as feasible. Contact simulations have small run-to-run variation, so do not place a hard gate directly on the observed baseline value: for each applicable nonzero upper limit allow at least 10 percent above the baseline, and for each nonzero lower limit allow at least 10 percent below it. Put desired improvements beyond that valid baseline in task_spec rather than silently relying on DTO defaults.
   Preserve every requested spatial and interaction semantic. Do not optimize an easy-to-change proxy that contradicts the prompt merely to create non-zero loss. In particular, minimize pair distance only when the requested outcome actually requires those objects to meet or overlap, and maximize it only when the request requires separation. If the baseline already visibly satisfies the requested arrangement, optimize a genuine quality defect such as penetration, deformation, settling, containment, or propagation rather than inventing a contradictory layout goal.
   Treat metric validity flags in `base_metrics.json` as semantic gates: for example, do not use a fluid's `relative_mass_change` when its `mass_conservation_applicable` flag is false.
   When task semantics involve containment, compare the available measurements and select the one that directly reflects the visible requested outcome; do not use total final material mass as a containment proxy without justification.
   Match each loss mode to the requested direction: use `minimum` only when larger values are better, `maximum` only when smaller values are better, and `target` only when the request specifies a desired numeric level. Before choosing tolerances or generic feasibility gates, compare against the baseline values in `base_metrics.json`. Do not impose a limit orders of magnitude tighter than a valid baseline unless a selected optimization parameter can plausibly control that quantity. The task_spec must distinguish the baseline: at least one selected task loss must be non-zero on `base_metrics.json`. If the baseline already meets a coarse task condition, choose a meaningful, agent-justified quality measure of that requested outcome instead of encoding a condition that is already fully satisfied.
   When the request says maximize or minimize an outcome, keep the loss informative across the full useful range: use the physically ideal bound as the target (for example, 1.0 for a fraction to maximize or 0.0 for an error to minimize), not a coarse pass threshold that makes substantially better candidates receive the same zero loss.
   Do not put object IDs, object pairs, or any other scene-specific condition into `objective_settings`; in particular, leave `minimum_coupling_contact_duration` empty. Express requested contacts only as task_spec metrics.
   Resource, memory, runtime, and `metrics.physics.*` validity fields are never valid task outcomes; physics validity belongs to framework-owned hard constraints.
5. Call validate_optimization_plan and repair it if needed. Its active_parameters result must list every enabled parameter path you supplied; otherwise repair the plan and validate again.
6. Stop after validation. Do not call start_optimization, get_optimization_job, get_optimization_trial, apply_optimization_trial, preview_scene, or run_scene.
"""


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mcp-run-dir", type=Path, required=True, help="Completed MCP generation run directory")
    parser.add_argument("--case", choices=sorted(AGENT_PROMPTS), action="append", dest="cases")
    parser.add_argument("--output-dir", type=Path, help="New directory for objective-Agent workspaces")
    parser.add_argument("--pilot-dir", type=Path, help="Optional prior optimizer suite used as repair evidence")
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument("--reasoning", default="low")
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--auth-file", type=Path, default=Path.home() / ".codex" / "auth.json")
    parser.add_argument("--no-auth-copy", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse validated objective bundles already present in --output-dir.",
    )
    return parser.parse_args()


def _write_summary(path: Path, rows: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _completed_cases(run_dir: Path) -> dict[str, dict[str, Any]]:
    summary_path = run_dir / "summary.json"
    if not summary_path.is_file():
        summary_path = run_dir.parent / "summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"MCP generation summary is missing: {run_dir / 'summary.json'}")
    rows = json.loads(summary_path.read_text(encoding="utf-8"))
    return {
        str(row["case"]): row for row in rows if row.get("status") == "passed" and row.get("variant", "mcp") == "mcp"
    }


def _contract_errors(scene_store: Path, case_name: str, metrics: dict[str, Any]) -> list[str]:
    plan_dir = scene_store / ".optimizations" / "plans" / f"objective-{case_name}"
    task_path = plan_dir / "task_spec.json"
    objective_path = plan_dir / "objective_settings.json"
    if not task_path.is_file() or not objective_path.is_file():
        return ["objective plan must include task_spec.json and objective_settings.json"]
    try:
        task_spec = TaskSpec.from_dict(json.loads(task_path.read_text(encoding="utf-8")))
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        return [f"invalid task_spec: {type(error).__name__}: {error}"]
    try:
        objective_settings = json.loads(objective_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        return [f"invalid objective_settings: {type(error).__name__}: {error}"]
    return [
        *validate_task_spec(task_spec, metrics),
        *validate_task_semantics(task_spec, metrics),
        *validate_objective_settings(task_spec, objective_settings, metrics),
    ]


def _bundle_errors(bundle: Path, metrics: dict[str, Any]) -> list[str]:
    required = (
        "scene.json",
        "optimization_plan.json",
        "task_spec.json",
        "objective_settings.json",
        "base_metrics.json",
    )
    missing = [name for name in required if not (bundle / name).is_file()]
    if missing:
        return [f"objective bundle is missing: {', '.join(missing)}"]
    try:
        task_spec = TaskSpec.from_dict(json.loads((bundle / "task_spec.json").read_text(encoding="utf-8")))
        objective_settings = json.loads((bundle / "objective_settings.json").read_text(encoding="utf-8"))
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        return [f"invalid objective bundle: {type(error).__name__}: {error}"]
    return [
        *validate_task_spec(task_spec, metrics),
        *validate_task_semantics(task_spec, metrics),
        *validate_objective_settings(task_spec, objective_settings, metrics),
    ]


def _invoke_agent(
    command: list[str], prompt: str, events_path: Path, environment: dict[str, str], timeout: float
) -> int | None:
    with events_path.open("a", encoding="utf-8") as events:
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=events,
            stderr=subprocess.STDOUT,
            text=True,
            env=environment,
            start_new_session=True,
        )
        try:
            process.communicate(prompt, timeout=timeout)
            return process.returncode
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=10.0)
            return None


def _run_case(
    case_name: str,
    *,
    source_scene: Path,
    source_metrics: Path,
    source_video: Path,
    repo: Path,
    output_root: Path,
    auth_source: Path | None,
    model: str,
    reasoning: str,
    timeout: float,
    pilot_dir: Path | None,
) -> dict[str, Any]:
    case_root = output_root / case_name
    workspace = case_root / "workspace"
    scene_store = case_root / "scene-store"
    codex_home = case_root / "codex-home"
    workspace.mkdir(parents=True)
    scene_store.mkdir()
    scene = Scene.from_dict(json.loads(source_scene.read_text(encoding="utf-8")))
    if scene.name != case_name:
        raise ValueError(f"Expected scene {case_name!r}, got {scene.name!r}")
    SceneStore(scene_store).save(scene)
    shutil.copy2(source_scene, workspace / "base_scene.json")
    if source_metrics.is_file():
        shutil.copy2(source_metrics, workspace / "base_metrics.json")
        metrics = json.loads(source_metrics.read_text(encoding="utf-8"))
    else:
        (workspace / "base_metrics.json").write_text("{}\n", encoding="utf-8")
        metrics = {}
    _write_visual_evidence(source_video, workspace, scene.settings.duration)
    pilot_available = _write_pilot_evidence(pilot_dir, case_name, workspace)
    (workspace / "base_metric_paths.json").write_text(
        json.dumps(metric_paths(metrics), indent=2) + "\n", encoding="utf-8"
    )
    (workspace / "physics_metric_semantics.json").write_text(
        json.dumps(physics_metric_semantics(metrics), indent=2) + "\n", encoding="utf-8"
    )
    plan_name = f"objective-{case_name}"
    prompt = _prompt(case_name, plan_name, pilot_available=pilot_available)
    (case_root / "prompt.txt").write_text(prompt, encoding="utf-8")
    _prepare_codex_home(codex_home, auth_source)
    warp_cache = case_root / "warp-cache"
    warp_cache.mkdir()
    _register_mcp(repo, codex_home, scene_store, warp_cache=warp_cache)
    environment = {**os.environ, "CODEX_HOME": str(codex_home), "WARP_CACHE_ROOT": str(warp_cache)}
    command = [
        "codex",
        "exec",
        "--ephemeral",
        "--skip-git-repo-check",
        "--dangerously-bypass-approvals-and-sandbox",
        "--json",
        "--output-last-message",
        str(case_root / "final-message.txt"),
        "--cd",
        str(workspace),
    ]
    if model:
        command.extend(("--model", model))
    if reasoning:
        command.extend(("-c", f'model_reasoning_effort="{reasoning}"'))
    started = time.monotonic()
    events_path = case_root / "codex-events.jsonl"
    exit_code = _invoke_agent(command, prompt, events_path, environment, timeout)
    errors = (
        _contract_errors(scene_store, case_name, metrics)
        if exit_code == 0
        else ["objective Agent did not exit successfully"]
    )
    repair_exit_code: int | None = None
    if errors:
        repair_prompt = _prompt(
            case_name,
            plan_name,
            repair_errors=errors,
            pilot_available=pilot_available,
        )
        repair_exit_code = _invoke_agent(command, repair_prompt, events_path, environment, timeout)
        errors = _contract_errors(scene_store, case_name, metrics) if repair_exit_code == 0 else errors

    objective_dir = _copy_objective_bundle(scene_store, case_root, case_name) if not errors else None
    if objective_dir is not None:
        shutil.copy2(workspace / "base_metrics.json", objective_dir / "base_metrics.json")
        (objective_dir / "provenance.json").write_text(
            json.dumps(
                {
                    "source_scene": str(source_scene),
                    "source_metrics": str(source_metrics),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    return {
        "case": case_name,
        "status": "passed" if not errors and objective_dir is not None else "failed",
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "codex_exit_code": exit_code,
        "repair_codex_exit_code": repair_exit_code,
        "validation_errors": errors,
        "objective_dir": str(objective_dir) if objective_dir is not None else None,
        "source_scene": str(source_scene),
        "source_video": str(source_video),
    }


def _write_visual_evidence(source_video: Path, workspace: Path, duration: float) -> None:
    """Copy a baseline video and three time-spaced stills for visual objective design."""
    if not source_video.is_file():
        raise FileNotFoundError(f"Baseline video is missing: {source_video}")
    video = workspace / "baseline.mp4"
    shutil.copy2(source_video, video)
    timestamps = {
        "baseline_start.png": 0.0,
        "baseline_middle.png": duration * 0.5,
    }
    for filename, timestamp in timestamps.items():
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-ss",
                f"{timestamp:.6f}",
                "-i",
                str(video),
                "-frames:v",
                "1",
                str(workspace / filename),
            ],
            check=True,
        )
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-sseof",
            "-0.1",
            "-i",
            str(video),
            "-frames:v",
            "1",
            str(workspace / "baseline_end.png"),
        ],
        check=True,
    )


def _write_pilot_evidence(pilot_root: Path | None, case_name: str, workspace: Path) -> bool:
    """Copy generic prior-trial outcomes and representative videos for Objective Agent repair."""
    if pilot_root is None:
        return False
    case_root = pilot_root / case_name
    if not case_root.is_dir():
        return False
    evidence: dict[str, Any] = {"case": case_name, "methods": {}}
    for method in ("random", "tpe", "cmaes"):
        method_root = case_root / method
        trials = []
        for result_path in sorted(method_root.glob("trial_*/result.json")):
            trial_root = result_path.parent
            result = json.loads(result_path.read_text(encoding="utf-8"))
            metrics_path = trial_root / "metrics.json"
            metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.is_file() else {}
            trials.append(
                {
                    "trial": trial_root.name,
                    "parameters": json.loads((trial_root / "parameters.json").read_text(encoding="utf-8")),
                    "feasible": result.get("feasible"),
                    "objective": result.get("objective"),
                    "failure_reason": result.get("failure_reason"),
                    "raw_objective": metrics.get("metrics", {}).get("raw_objective"),
                    "constraint_values": metrics.get("constraints", {}),
                    "runtime_sec": metrics.get("runtime_sec"),
                }
            )
        summary_path = method_root / "optimization_summary.json"
        summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else {}
        evidence["methods"][method] = {
            "feasibility_rate": summary.get("feasibility_rate"),
            "best_feasible_trial": summary.get("best_feasible_trial"),
            "median_feasible_trial": summary.get("median_feasible_trial"),
            "trials": trials,
        }
        finalists = summary.get("finalist_artifacts", [])
        if finalists:
            video = Path(str(finalists[0]["output_dir"])) / "animation.mp4"
            if video.is_file():
                shutil.copy2(video, workspace / f"pilot_{method}.mp4")
    if not any(item["trials"] for item in evidence["methods"].values()):
        return False
    (workspace / "pilot_evidence.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
    return True


def main() -> None:
    """Collect Agent-authored objective contracts for completed MCP scenes."""
    args = _parse_args()
    repo = Path(__file__).resolve().parents[1]
    mcp_run_dir = args.mcp_run_dir.resolve()
    output_root = (args.output_dir or mcp_run_dir / "optimization-objectives").resolve()
    pilot_dir = args.pilot_dir.resolve() if args.pilot_dir is not None else None
    output_root.mkdir(parents=True, exist_ok=args.resume)
    completed = _completed_cases(mcp_run_dir)
    cases = args.cases or sorted(completed)
    auth_source = None if args.no_auth_copy else args.auth_file.expanduser().resolve()
    rows: list[dict[str, Any]] = []
    for case_name in cases:
        output_dir = mcp_run_dir / case_name / "workspace" / "result"
        source_scene = output_dir / "scene.json"
        source_metrics = output_dir / "metrics.json"
        if case_name not in completed or not source_scene.is_file():
            rows.append({"case": case_name, "status": "skipped", "reason": "MCP generation is not completed"})
        elif args.resume and not _bundle_errors(
            output_root / case_name / "optimization-objective",
            json.loads(source_metrics.read_text(encoding="utf-8")),
        ):
            rows.append(
                {
                    "case": case_name,
                    "status": "passed",
                    "resumed": True,
                    "objective_dir": str(output_root / case_name / "optimization-objective"),
                    "source_scene": str(source_scene),
                    "source_video": str(output_dir / "animation.mp4"),
                }
            )
        else:
            if (output_root / case_name).exists():
                raise FileExistsError(
                    f"Incomplete objective workspace exists for {case_name}; move it aside before resuming: "
                    f"{output_root / case_name}"
                )
            rows.append(
                _run_case(
                    case_name,
                    source_scene=source_scene,
                    source_metrics=source_metrics,
                    source_video=output_dir / "animation.mp4",
                    repo=repo,
                    output_root=output_root,
                    auth_source=auth_source,
                    model=args.model,
                    reasoning=args.reasoning,
                    timeout=args.timeout,
                    pilot_dir=pilot_dir,
                )
            )
        _write_summary(output_root / "summary.json", rows)
    if any(row["status"] != "passed" for row in rows):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
