# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import sys
import unittest
from pathlib import Path
from unittest.mock import call, patch

from evaluation import run_canonical
from evaluation.canonical import canonical_scenes


class TestRunCanonical(unittest.TestCase):
    @patch("evaluation.run_canonical.subprocess.run")
    def test_full_run_isolates_each_scene_in_a_subprocess(self, run):
        output_dir = Path("custom-results")
        with patch.object(sys, "argv", ["run_canonical", "--output-dir", str(output_dir)]):
            run_canonical.main()

        self.assertEqual(
            run.call_args_list,
            [
                call(
                    [
                        sys.executable,
                        "-m",
                        "evaluation.run_canonical",
                        "--scene",
                        name,
                        "--output-dir",
                        str(output_dir),
                    ],
                    check=True,
                )
                for name in canonical_scenes()
            ],
        )

    @patch("evaluation.run_canonical.SceneExecutorLocal")
    def test_single_scene_runs_in_the_current_process(self, executor_type):
        output_dir = Path("custom-results")
        scene_name = "02_domino_wave"
        with patch.object(
            sys,
            "argv",
            ["run_canonical", "--scene", scene_name, "--output-dir", str(output_dir)],
        ):
            run_canonical.main()

        executor_type.return_value.run.assert_called_once_with(
            canonical_scenes()[scene_name],
            output_dir=output_dir / scene_name,
        )


if __name__ == "__main__":
    unittest.main()
