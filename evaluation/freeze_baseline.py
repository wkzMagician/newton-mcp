# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Freeze deterministic canonical simulation evidence before major refactors."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import warp as wp

from ca_framework.scene import SCHEMA_VERSION, Scene, SceneExecutorLocal
from evaluation.canonical import canonical_scenes


def freeze_baseline(output_dir: Path) -> dict[str, object]:
    """Run every canonical scene and persist M0 comparison evidence."""
    return freeze_scenes(canonical_scenes(), output_dir)


def freeze_scenes(
    scenes: dict[str, Scene], output_dir: Path, *, source_commit: str | None = None
) -> dict[str, object]:
    """Run a named scene set and persist deterministic comparison evidence."""
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    executor = SceneExecutorLocal()
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "warp_version": wp.__version__,
        "device": str(wp.get_device()),
        "random_seed": 0,
        "source_commit": source_commit,
        "scenes": {},
    }
    for name, scene in scenes.items():
        directory = output_dir / name
        directory.mkdir(exist_ok=True)
        started = time.perf_counter()
        result = executor.simulate(scene, capture_cache=False)
        wall_time = time.perf_counter() - started
        (directory / "scene.json").write_text(json.dumps(scene.to_dict(), indent=2) + "\n", encoding="utf-8")
        (directory / "metrics.json").write_text(
            json.dumps(result["metrics"], indent=2) + "\n", encoding="utf-8"
        )
        (directory / "diagnostics.jsonl").write_text(
            "".join(json.dumps(item, sort_keys=True) + "\n" for item in result["diagnostics"]),
            encoding="utf-8",
        )
        evidence = {
            "status": result["status"],
            "scene_hash": result["scene_hash"],
            "trajectory_hash": result["trajectory_hash"],
            "wall_time_sec": wall_time,
        }
        (directory / "baseline.json").write_text(json.dumps(evidence, indent=2) + "\n", encoding="utf-8")
        manifest["scenes"][name] = evidence
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    freeze_baseline(args.output_dir)


if __name__ == "__main__":
    main()
