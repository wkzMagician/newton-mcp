# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Run the serial direct-Newton and MCP baseline benchmark."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any

from evaluation.agent_prompts import AGENT_PROMPTS
from evaluation.run_agent_experiments import run_case

VARIANTS: dict[str, tuple[bool, str]] = {
    "direct-newton": (False, "none"),
    "mcp": (True, "none"),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=sorted(AGENT_PROMPTS), action="append", dest="cases")
    parser.add_argument("--variant", choices=tuple(VARIANTS), action="append", dest="variants")
    parser.add_argument("--run-dir", type=Path, help="New directory for this complete matrix")
    parser.add_argument("--resume", type=Path, help="Rerun selected failed matrix entries in place")
    parser.add_argument("--model", default="gpt-5.4")
    parser.add_argument("--reasoning", default="low")
    parser.add_argument("--timeout", type=float, default=1200.0)
    parser.add_argument("--auth-file", type=Path, default=Path.home() / ".codex" / "auth.json")
    parser.add_argument("--no-auth-copy", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--allow-over-budget", action="store_true")
    return parser.parse_args()


def _write_summary(path: Path, results: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def main() -> None:
    """Execute all selected variants serially to keep one GPU owner at a time."""
    args = _parse_args()
    repo = Path(__file__).resolve().parents[1]
    if args.run_dir is not None and args.resume is not None:
        raise ValueError("--run-dir and --resume are mutually exclusive")
    run_root = (
        args.resume
        or args.run_dir
        or repo / "evaluation" / "results" / "agent-matrix" / time.strftime("%Y%m%d-%H%M%S")
    ).resolve()
    run_root.mkdir(parents=True, exist_ok=args.resume is not None)
    cases = args.cases or list(AGENT_PROMPTS)
    variants = args.variants or list(VARIANTS)
    auth_source = None if args.no_auth_copy else args.auth_file.expanduser().resolve()
    summary_path = run_root / "summary.json"
    previous = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.is_file() else []
    result_by_key = {(str(item.get("variant")), str(item.get("case"))): item for item in previous}

    for variant in variants:
        with_mcp, optimizer = VARIANTS[variant]
        for case_name in cases:
            print(f"Running {variant}/{case_name}", flush=True)
            case_root = run_root / variant / case_name
            if case_root.exists():
                shutil.rmtree(case_root)
            try:
                result = run_case(
                    case_name,
                    repo=repo,
                    run_root=run_root / variant,
                    auth_source=auth_source,
                    model=args.model,
                    reasoning=args.reasoning,
                    timeout=args.timeout,
                    dry_run=args.dry_run,
                    with_mcp=with_mcp,
                    optimizer=optimizer,
                    allow_over_budget=args.allow_over_budget,
                )
            except Exception as error:
                result = {"case": case_name, "status": "error", "error": f"{type(error).__name__}: {error}"}
            result["variant"] = variant
            result_by_key[(variant, case_name)] = result
            ordered = [
                result_by_key[(str(item.get("variant")), str(item.get("case")))]
                for item in previous
                if (str(item.get("variant")), str(item.get("case"))) in result_by_key
            ]
            known = {(str(item.get("variant")), str(item.get("case"))) for item in previous}
            ordered.extend(result_by_key[key] for key in result_by_key if key not in known)
            _write_summary(summary_path, ordered)
            print(f"{variant}/{case_name}: {result['status']}", flush=True)

    final_results = json.loads(summary_path.read_text(encoding="utf-8"))
    passed = sum(item["status"] in {"passed", "dry-run"} for item in final_results)
    print(f"Summary: {passed}/{len(final_results)} successful; details: {summary_path}")
    if any(
        item.get("variant") in variants and item.get("case") in cases and item["status"] not in {"passed", "dry-run"}
        for item in final_results
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
