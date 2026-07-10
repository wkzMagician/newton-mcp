# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Run private canonical scenes and write their output bundles."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from ca_framework.scene import SceneExecutorLocal
from evaluation.canonical import canonical_scenes


def main() -> None:
    """Run one or all canonical scenes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scene", choices=sorted(canonical_scenes()))
    parser.add_argument("--output-dir", type=Path, default=Path("evaluation/results/canonical"))
    args = parser.parse_args()

    scenes = canonical_scenes()
    names = [args.scene] if args.scene else list(scenes)
    if args.scene is None:
        for name in names:
            output_dir = args.output_dir / name
            print(f"Running {name} -> {output_dir}", flush=True)
            subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "evaluation.run_canonical",
                    "--scene",
                    name,
                    "--output-dir",
                    str(args.output_dir),
                ],
                check=True,
            )
        return

    executor = SceneExecutorLocal()
    for name in names:
        output_dir = args.output_dir / name
        print(f"Running {name} -> {output_dir}")
        print(executor.run(scenes[name], output_dir=output_dir))


if __name__ == "__main__":
    main()
