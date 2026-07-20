# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for resumable optimizer-suite orchestration."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from evaluation.run_mcp_optimizer_suite import _completed_case_row, _completed_output


class TestCompletedOptimizerOutput(unittest.TestCase):
    def test_requires_a_video_for_every_method(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for method in ("random", "tpe", "cmaes"):
                finalist = root / method / "finalists" / "trial_00000"
                finalist.mkdir(parents=True)
                (root / method / "optimization_summary.json").write_text(
                    json.dumps({"finalist_artifacts": [{"output_dir": str(finalist)}]}),
                    encoding="utf-8",
                )
                (finalist / "animation.mp4").write_bytes(b"video")

            self.assertTrue(_completed_output(root))
            (root / "tpe" / "finalists" / "trial_00000" / "animation.mp4").unlink()
            self.assertFalse(_completed_output(root))
            self.assertTrue(_completed_output(root, ("random", "cmaes")))

    def test_rebuilds_summary_row_from_completed_method_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = root / "bundle"
            bundle.mkdir()
            (bundle / "scene.json").write_text("{}", encoding="utf-8")
            (bundle / "task_spec.json").write_text('{"name": "task", "metrics": []}', encoding="utf-8")
            (bundle / "objective_settings.json").write_text("{}", encoding="utf-8")
            output = root / "output"
            (output / "random").mkdir(parents=True)
            (output / "random" / "optimization_summary.json").write_text('{"trial_count": 3}', encoding="utf-8")

            row = _completed_case_row("case", bundle=bundle, output_dir=output, optimizers=("random",))

        self.assertEqual(row["methods"]["random"]["trial_count"], 3)
        self.assertTrue(row["resumed"])


if __name__ == "__main__":
    unittest.main()
