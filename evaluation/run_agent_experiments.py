# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Run isolated prompt-to-scene experiments through ``codex exec``."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from evaluation.agent_prompts import AGENT_PROMPTS, build_agent_prompt

REQUIRED_FILES = ("animation.mp4", "scene.json", "program.py", "metrics.json", "diagnostics.jsonl")


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


def _register_mcp(repo: Path, codex_home: Path, scene_store: Path) -> None:
    environment = {**os.environ, "CODEX_HOME": str(codex_home)}
    command = [
        "codex",
        "mcp",
        "add",
        "ca-scene",
        "--env",
        f"CA_SCENE_WORKSPACE={scene_store}",
    ]
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


def _bundle_status(output_dir: Path) -> tuple[str | None, bool | None]:
    manifest_path = output_dir / "manifest.json"
    metrics_path = output_dir / "metrics.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8")) if manifest_path.is_file() else {}
    metrics = json.loads(metrics_path.read_text(encoding="utf-8")) if metrics_path.is_file() else {}
    return manifest.get("status"), metrics.get("physics", {}).get("valid")


def run_case(
    case_name: str,
    *,
    repo: Path,
    run_root: Path,
    auth_source: Path | None,
    model: str | None,
    timeout: float,
    dry_run: bool,
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
    _register_mcp(repo, codex_home, scene_store)
    environment = {**os.environ, "CODEX_HOME": str(codex_home)}
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

    artifacts = _artifact_status(output_dir)
    job_status, physics_valid = _bundle_status(output_dir)
    summary.update(
        {
            "elapsed_seconds": round(time.monotonic() - started, 3),
            "artifacts": artifacts,
            "job_status": job_status,
            "physics_valid": physics_valid,
            "status": "passed"
            if summary.get("codex_exit_code") == 0
            and all(artifacts.values())
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
    parser.add_argument("--model", help="Optional Codex model override")
    parser.add_argument("--timeout", type=float, default=600.0, help="Timeout per case in seconds")
    parser.add_argument("--auth-file", type=Path, default=Path.home() / ".codex" / "auth.json")
    parser.add_argument("--no-auth-copy", action="store_true", help="Use OPENAI_API_KEY instead of copying auth.json")
    parser.add_argument("--dry-run", action="store_true", help="Prepare prompts/workspaces without invoking Codex")
    return parser.parse_args()


def main() -> None:
    """Run the selected experiments serially and write ``summary.json``."""
    args = _parse_args()
    repo = Path(__file__).resolve().parents[1]
    run_root = args.run_dir or repo / "evaluation" / "results" / "agent-workflow" / time.strftime("%Y%m%d-%H%M%S")
    run_root = run_root.resolve()
    run_root.mkdir(parents=True, exist_ok=False)
    cases = args.cases or list(AGENT_PROMPTS)
    auth_source = None if args.no_auth_copy else args.auth_file.expanduser().resolve()

    results = []
    for case_name in cases:
        print(f"Running {case_name} -> {run_root / case_name}", flush=True)
        try:
            result = run_case(
                case_name,
                repo=repo,
                run_root=run_root,
                auth_source=auth_source,
                model=args.model,
                timeout=args.timeout,
                dry_run=args.dry_run,
            )
        except Exception as error:
            result = {"case": case_name, "status": "error", "error": f"{type(error).__name__}: {error}"}
        results.append(result)
        (run_root / "summary.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
        print(f"{case_name}: {result['status']}", flush=True)

    passed = sum(result["status"] in {"passed", "dry-run"} for result in results)
    print(f"Summary: {passed}/{len(results)} successful; details: {run_root / 'summary.json'}")
    if passed != len(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
