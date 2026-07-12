# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Run isolated prompt-to-scene experiments through ``codex exec``."""

from __future__ import annotations

import argparse
import base64
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from html import escape
from pathlib import Path
from threading import Lock
from typing import Any

from evaluation.agent_prompts import AGENT_PROMPTS, build_agent_prompt

REQUIRED_FILES = ("animation.mp4", "scene.json", "program.py", "metrics.json", "diagnostics.jsonl")
_ACTIVE_PROCESSES: set[int] = set()
_ACTIVE_LOCK = Lock()


def _copy_agent_skills(repo: Path, workspace: Path) -> None:
    destination = workspace / ".agents" / "skills"
    destination.mkdir(parents=True)
    for source in sorted((repo / "knowledge" / "skills").glob("*")):
        if source.is_dir():
            shutil.copytree(source, destination / source.name)


def _prepare_codex_home(codex_home: Path, auth_source: Path | None) -> None:
    codex_home.mkdir(parents=True)
    if auth_source is not None:
        if not auth_source.is_file():
            raise FileNotFoundError(f"Codex authentication file not found: {auth_source}")
        shutil.copy2(auth_source, codex_home / "auth.json")


def _register_mcp(
    repo: Path, codex_home: Path, scene_store: Path, *, allow_over_budget: bool = False, warp_cache: Path | None = None
) -> None:
    environment = {**os.environ, "CODEX_HOME": str(codex_home)}
    command = [
        "codex",
        "mcp",
        "add",
        "ca-scene",
        "--env",
        f"CA_SCENE_WORKSPACE={scene_store}",
    ]
    command.extend(("--env", f"CA_SCENE_ALLOW_OVER_BUDGET={int(allow_over_budget)}"))
    if warp_cache is not None:
        command.extend(("--env", f"WARP_CACHE_ROOT={warp_cache}"))
    for name in ("DISPLAY", "XAUTHORITY", "WAYLAND_DISPLAY", "WARP_CACHE_ROOT", "UV_CACHE_DIR"):
        if value := environment.get(name):
            command.extend(("--env", f"{name}={value}"))
    command.extend(
        (
            "--",
            "uv",
            "run",
            "--project",
            str(repo / "mcp_server"),
            "ca-scene-mcp",
        )
    )
    subprocess.run(command, check=True, env=environment)


def _artifact_status(output_dir: Path) -> dict[str, bool]:
    status = {
        name: (output_dir / name).is_file() and (name == "diagnostics.jsonl" or (output_dir / name).stat().st_size > 0)
        for name in REQUIRED_FILES
    }
    status["cache"] = (output_dir / "cache").is_dir() and any((output_dir / "cache").iterdir())
    return status


def _valid_video(path: Path) -> bool:
    """Require a non-empty MP4 with a readable, plausible video stream."""
    if not path.is_file() or path.stat().st_size == 0:
        return False
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=nb_frames,duration",
            "-of",
            "json",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if probe.returncode != 0:
        return False
    try:
        stream = json.loads(probe.stdout)["streams"][0]
        return float(stream.get("duration") or 0) > 0 and int(stream.get("nb_frames") or 0) > 0
    except (KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError):
        return False


def _bundle_status(output_dir: Path) -> tuple[str | None, bool | None]:
    manifest_path = output_dir / "manifest.json"
    metrics_path = output_dir / "metrics.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.is_file() else {}
    return manifest.get("status"), metrics.get("physics", {}).get("valid")


def _resume_cached_render(case_name: str, case_root: Path) -> dict[str, Any] | None:
    """Resume render-only work when simulation cache and valid physics are complete."""
    output_dir = case_root / "workspace" / "result"
    cache_manifest = output_dir / "cache" / "manifest.json"
    scene_path = output_dir / "scene.json"
    if not cache_manifest.is_file() or not scene_path.is_file():
        return None
    try:
        cache_complete = json.loads(cache_manifest.read_text(encoding="utf-8")).get("complete", False)
        _, physics_valid = _bundle_status(output_dir)
    except (OSError, json.JSONDecodeError):
        return None
    if not cache_complete or physics_valid is not True:
        return None

    # Keep Warp initialization out of ordinary runner startup and dry runs.
    from ca_framework.scene import Scene, SceneExecutorLocal  # noqa: PLC0415

    started = time.monotonic()
    scene = Scene.from_dict(json.loads(scene_path.read_text(encoding="utf-8")))
    executor = SceneExecutorLocal()
    result: dict[str, Any] = {}
    passed = False
    artifacts: dict[str, bool] = {}
    video_valid = False
    job_status: str | None = None
    # OpenGL context/readback creation can fail transiently. Retrying render is
    # cheap compared with discarding a complete simulation cache.
    for _attempt in range(3):
        try:
            result = executor.resume_render(scene, output_dir=output_dir)
        except Exception as error:
            result = {"status": "render_failed", "error": f"{type(error).__name__}: {error}"}
        artifacts = _artifact_status(output_dir)
        video_valid = _valid_video(output_dir / "animation.mp4")
        job_status, physics_valid = _bundle_status(output_dir)
        passed = (
            result.get("status") == "completed"
            and all(artifacts.values())
            and video_valid
            and job_status == "completed"
            and physics_valid is True
        )
        if passed:
            break
    return {
        "case": case_name,
        "status": "passed" if passed else "failed",
        "workspace": str(case_root / "workspace"),
        "output_dir": str(output_dir),
        "elapsed_seconds": round(time.monotonic() - started, 3),
        "artifacts": artifacts,
        "video_valid": video_valid,
        "job_status": job_status,
        "physics_valid": physics_valid,
        "resumed_render": True,
    }


def run_case(
    case_name: str,
    *,
    repo: Path,
    run_root: Path,
    auth_source: Path | None,
    model: str | None,
    reasoning: str | None,
    timeout: float,
    dry_run: bool,
    allow_over_budget: bool = False,
) -> dict[str, Any]:
    """Run one experiment and return its machine-readable summary."""
    case_root = run_root / case_name
    workspace = case_root / "workspace"
    scene_store = case_root / "scene-store"
    output_dir = workspace / "result"
    codex_home = case_root / "codex-home"
    for directory in (workspace, scene_store, output_dir):
        directory.mkdir(parents=True, exist_ok=False)
    _copy_agent_skills(repo, workspace)

    prompt = build_agent_prompt(case_name, str(output_dir))
    (case_root / "prompt.txt").write_text(prompt, encoding="utf-8")
    summary: dict[str, Any] = {
        "case": case_name,
        "status": "dry-run" if dry_run else "running",
        "workspace": str(workspace),
        "output_dir": str(output_dir),
        "prompt_file": str(case_root / "prompt.txt"),
    }
    if dry_run:
        return summary

    _prepare_codex_home(codex_home, auth_source)
    warp_cache = case_root / "warp-cache"
    warp_cache.mkdir()
    _register_mcp(repo, codex_home, scene_store, allow_over_budget=allow_over_budget, warp_cache=warp_cache)
    environment = {**os.environ, "CODEX_HOME": str(codex_home)}
    environment["WARP_CACHE_ROOT"] = str(warp_cache)
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
    with (case_root / "codex-events.jsonl").open("w", encoding="utf-8") as events:
        process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=events,
            stderr=subprocess.STDOUT,
            text=True,
            env=environment,
            start_new_session=True,
        )
        with _ACTIVE_LOCK:
            _ACTIVE_PROCESSES.add(process.pid)
        try:
            process.communicate(prompt, timeout=timeout)
            summary["codex_exit_code"] = process.returncode
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=10.0)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            summary["codex_exit_code"] = None
            summary["timed_out"] = True
        finally:
            with _ACTIVE_LOCK:
                _ACTIVE_PROCESSES.discard(process.pid)

    artifacts = _artifact_status(output_dir)
    video_valid = _valid_video(output_dir / "animation.mp4")
    job_status, physics_valid = _bundle_status(output_dir)
    summary.update(
        {
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "artifacts": artifacts,
            "video_valid": video_valid,
            "job_status": job_status,
            "physics_valid": physics_valid,
            "status": "passed"
            if summary.get("codex_exit_code") == 0
            and all(artifacts.values())
            and video_valid
            and job_status == "completed"
            and physics_valid is True
            else "failed",
        }
    )
    return summary


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=sorted(AGENT_PROMPTS), action="append", dest="cases")
    parser.add_argument("--run-dir", type=Path, help="New directory that will contain this evaluation run")
    parser.add_argument(
        "--model", default="gpt-5.6-terra", help="Optional Codex model override (default: gpt-5.6-terra)"
    )
    parser.add_argument(
        "--reasoning", default="low", help="Optional reasoning effort (e.g. low, medium, high; default: low)"
    )
    parser.add_argument("--timeout", type=float, default=1200.0, help="Timeout per case in seconds")
    parser.add_argument("--auth-file", type=Path, default=Path.home() / ".codex" / "auth.json")
    parser.add_argument("--no-auth-copy", action="store_true", help="Use OPENAI_API_KEY instead of copying auth.json")
    parser.add_argument("--dry-run", action="store_true", help="Prepare prompts/workspaces without invoking Codex")
    parser.add_argument("--jobs", type=int, default=2, help="Maximum concurrent Codex processes")
    parser.add_argument("--resource-capacity", type=int, default=2, help="Maximum aggregate case resource weight")
    parser.add_argument("--allow-over-budget", action="store_true", help="Allow MCP hard resource limits with warnings")
    parser.add_argument("--resume", type=Path, help="Resume a prior run directory")
    return parser.parse_args()


def _case_weight(case_name: str) -> int:
    return 2 if any(token in case_name for token in ("cloth", "liquid", "floating")) else 1


def _write_summary(path: Path, results: list[dict[str, Any] | None]) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps([item for item in results if item is not None], indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def _write_report(run_root: Path, results: list[dict[str, Any] | None]) -> None:
    rows = []
    for item in results:
        if item is None:
            continue
        case = escape(str(item["case"]))
        status = escape(str(item.get("status")))
        final_path = run_root / str(item["case"]) / "final-message.txt"
        final = escape(final_path.read_text(encoding="utf-8") if final_path.is_file() else "")
        rejected = " <strong>rejected</strong>" if item.get("physics_valid") is False else ""
        rows.append(
            f"<tr><td>{case}</td><td>{status}{rejected}</td><td>{item.get('elapsed_seconds', '')}</td><td><pre>{final}</pre></td></tr>"
        )
    (run_root / "report.html").write_text(
        "<!doctype html><meta charset=utf-8><title>Agent workflow report</title>"
        "<h1>Agent workflow report</h1><table><tr><th>Case</th><th>Status</th><th>Seconds</th><th>Agent final</th></tr>"
        + "".join(rows)
        + "</table>",
        encoding="utf-8",
    )
    videos = [
        run_root / str(item["case"]) / "workspace" / "result" / "animation.mp4" for item in results if item is not None
    ]
    source = next((path for path in videos if _valid_video(path)), None)
    if source is not None:
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-v",
                "error",
                "-ss",
                "0",
                "-i",
                str(source),
                "-frames:v",
                "1",
                "-vf",
                "scale=1280:-1",
                str(run_root / "video-overview.png"),
            ],
            check=False,
        )
    elif not (run_root / "video-overview.png").exists():
        # A valid transparent PNG keeps dry/interrupted runs self-contained.
        (run_root / "video-overview.png").write_bytes(
            base64.b64decode(
                "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mNk+M/wHwAF/gL+Xw4WAAAAAElFTkSuQmCC"
            )
        )


def _terminate_active_processes() -> None:
    """Terminate every isolated Codex process group still owned by this runner."""
    with _ACTIVE_LOCK:
        pids = list(_ACTIVE_PROCESSES)
    for pid in pids:
        try:
            os.killpg(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass


def main() -> None:
    """Run the selected experiments serially and write ``summary.json``."""
    args = _parse_args()
    if args.jobs < 1 or args.resource_capacity < 1:
        raise ValueError("--jobs and --resource-capacity must be positive")
    repo = Path(__file__).resolve().parents[1]
    run_root = (
        args.resume
        or args.run_dir
        or repo / "evaluation" / "results" / "agent-workflow" / time.strftime("%Y%m%d-%H%M%S")
    )
    run_root = run_root.resolve()
    run_root.mkdir(parents=True, exist_ok=bool(args.resume))
    cases = args.cases or list(AGENT_PROMPTS)
    auth_source = None if args.no_auth_copy else args.auth_file.expanduser().resolve()

    previous = {}
    summary_path = run_root / "summary.json"
    if args.resume and summary_path.is_file():
        previous = {item["case"]: item for item in json.loads(summary_path.read_text(encoding="utf-8"))}
    results: list[dict[str, Any] | None] = [None] * len(cases)
    pending = []
    for index, case_name in enumerate(cases):
        if previous.get(case_name, {}).get("status") == "passed":
            results[index] = previous[case_name]
        else:
            case_root = run_root / case_name
            resumed = _resume_cached_render(case_name, case_root) if args.resume and case_root.exists() else None
            if resumed is not None and resumed["status"] == "passed":
                results[index] = resumed
                continue
            if case_root.exists():
                shutil.rmtree(case_root)
            pending.append((index, case_name))
    _write_summary(summary_path, results)
    _write_report(run_root, results)
    running: dict[Future[dict[str, Any]], tuple[int, str, int]] = {}
    try:
        with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
            while pending or running:
                used = sum(weight for _, _, weight in running.values())
                for item in list(pending):
                    index, case_name = item
                    weight = _case_weight(case_name)
                    if len(running) >= max(1, args.jobs) or used + weight > max(1, args.resource_capacity):
                        continue
                    print(f"Running {case_name} -> {run_root / case_name}", flush=True)
                    future = pool.submit(
                        run_case,
                        case_name,
                        repo=repo,
                        run_root=run_root,
                        auth_source=auth_source,
                        model=args.model,
                        reasoning=args.reasoning,
                        timeout=args.timeout,
                        dry_run=args.dry_run,
                        allow_over_budget=args.allow_over_budget,
                    )
                    running[future] = (index, case_name, weight)
                    pending.remove(item)
                    used += weight
                if not running:
                    raise ValueError("resource-capacity is smaller than a selected case weight")
                done, _ = wait(running, return_when=FIRST_COMPLETED)
                for future in done:
                    index, case_name, _ = running.pop(future)
                    try:
                        result = future.result()
                    except Exception as error:
                        result = {"case": case_name, "status": "error", "error": f"{type(error).__name__}: {error}"}
                    results[index] = result
                    _write_summary(summary_path, results)
                    _write_report(run_root, results)
                    print(f"{case_name}: {result['status']}", flush=True)
    except KeyboardInterrupt:
        _terminate_active_processes()
        raise

    passed = sum(result is not None and result["status"] in {"passed", "dry-run"} for result in results)
    print(f"Summary: {passed}/{len(results)} successful; details: {run_root / 'summary.json'}")
    if passed != len(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
