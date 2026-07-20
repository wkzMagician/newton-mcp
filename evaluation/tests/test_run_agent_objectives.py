# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for agent-facing visual objective inputs."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from evaluation.run_agent_objectives import (
    _bundle_errors,
    _prompt,
    _write_pilot_evidence,
    _write_visual_evidence,
)


class TestObjectiveVisualEvidence(unittest.TestCase):
    def test_resume_bundle_requires_all_contract_files(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            errors = _bundle_errors(Path(temporary), {})

        self.assertEqual(len(errors), 1)
        self.assertIn("optimization_plan.json", errors[0])

    def test_prompt_requires_baseline_video_and_visual_inspection(self) -> None:
        prompt = _prompt("08_liquid_pour", "objective-08_liquid_pour")

        self.assertIn("baseline.mp4", prompt)
        self.assertIn("baseline_start.png", prompt)
        self.assertIn("spatial misalignment", prompt)
        self.assertIn("do not compensate for a diagnosed geometry problem", prompt)
        self.assertIn("physically ideal bound", prompt)
        self.assertIn("Preserve every requested spatial", prompt)
        self.assertIn("metrics.physics.*", prompt)

    def test_visual_evidence_copies_video_and_extracts_three_frames(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            video = root / "animation.mp4"
            video.write_bytes(b"video")
            workspace = root / "workspace"
            workspace.mkdir()

            with patch("evaluation.run_agent_objectives.subprocess.run") as run:
                _write_visual_evidence(video, workspace, 5.0)

            self.assertEqual((workspace / "baseline.mp4").read_bytes(), b"video")
            self.assertEqual(run.call_count, 3)
            outputs = [Path(call.args[0][-1]).name for call in run.call_args_list]
            self.assertEqual(outputs, ["baseline_start.png", "baseline_middle.png", "baseline_end.png"])

    def test_pilot_evidence_includes_constraint_values_and_runtime(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trial = root / "04_cloth_ball" / "random" / "trial_00000"
            trial.mkdir(parents=True)
            (trial / "parameters.json").write_text('{"stiffness": 4.0}', encoding="utf-8")
            (trial / "result.json").write_text(
                '{"feasible": false, "objective": 1001.0, "failure_reason": "cloth.edge"}',
                encoding="utf-8",
            )
            (trial / "metrics.json").write_text(
                '{"metrics": {}, "constraints": {"cloth.edge": 2.5}, "runtime_sec": 3.0}',
                encoding="utf-8",
            )
            workspace = root / "workspace"
            workspace.mkdir()

            self.assertTrue(_write_pilot_evidence(root, "04_cloth_ball", workspace))
            evidence = json.loads((workspace / "pilot_evidence.json").read_text(encoding="utf-8"))

        trial_evidence = evidence["methods"]["random"]["trials"][0]
        self.assertEqual(trial_evidence["constraint_values"], {"cloth.edge": 2.5})
        self.assertEqual(trial_evidence["runtime_sec"], 3.0)
