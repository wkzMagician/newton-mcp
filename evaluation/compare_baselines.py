# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Compare pre/post-refactor canonical evidence against explicit tolerances."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def compare_baselines(before: Path, after: Path) -> dict[str, Any]:
    """Return per-scene trajectory and physical-metric regression evidence."""
    before_manifest = _read(before / "manifest.json")
    after_manifest = _read(after / "manifest.json")
    scenes = {}
    for name, baseline in before_manifest["scenes"].items():
        current = after_manifest["scenes"][name]
        baseline_metrics = _read(before / name / "metrics.json")
        current_metrics = _read(after / name / "metrics.json")
        checks = {
            "status_completed": current["status"] == "completed",
            "penetration": float(current_metrics.get("max_penetration", 0.0))
            <= max(0.05, float(baseline_metrics.get("max_penetration", 0.0)) + 0.02),
            "soft_penetration": float(current_metrics.get("max_soft_penetration", 0.0))
            <= max(0.1, float(baseline_metrics.get("max_soft_penetration", 0.0)) + 0.05),
        }
        for object_id, values in current_metrics.get("cloth", {}).items():
            reference = baseline_metrics.get("cloth", {}).get(object_id, {})
            checks[f"cloth.{object_id}.edge_ratio"] = float(
                values.get("max_edge_length_ratio", 1.0)
            ) <= max(2.0, 1.1 * float(reference.get("max_edge_length_ratio", 1.0)))
            checks[f"cloth.{object_id}.flips"] = int(values.get("flipped_triangle_count", 0)) == 0
        for object_id, values in current_metrics.get("fluid", {}).items():
            reference = baseline_metrics.get("fluid", {}).get(object_id, {})
            checks[f"fluid.{object_id}.mass"] = abs(float(values.get("relative_mass_change", 0.0))) <= max(
                0.1, abs(float(reference.get("relative_mass_change", 0.0))) + 0.05
            )
            current_divergence = float(values.get("peak_divergence", values.get("max_divergence", 0.0)))
            reference_divergence = float(
                reference.get("peak_divergence", reference.get("max_divergence", 0.0))
            )
            checks[f"fluid.{object_id}.divergence"] = current_divergence <= max(
                10.0, 1.25 * reference_divergence
            )
        scenes[name] = {
            "passed": all(checks.values()),
            "trajectory_hash_equal": baseline["trajectory_hash"] == current["trajectory_hash"],
            "checks": checks,
        }
    return {
        "before_schema_version": before_manifest["schema_version"],
        "after_schema_version": after_manifest["schema_version"],
        "passed": all(item["passed"] for item in scenes.values()),
        "scenes": scenes,
    }


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    comparison = compare_baselines(args.before, args.after)
    text = json.dumps(comparison, indent=2) + "\n"
    if args.output is not None:
        args.output.write_text(text, encoding="utf-8")
    print(text, end="")
    if not comparison["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
