# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for generic optimizer-result analysis."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from evaluation.analyze_optimizer_results import _baseline, _comparison_summaries, _scene_agent_events


class TestBaselineAnalysis(unittest.TestCase):
    def test_baseline_applies_physics_constraints_to_raw_metrics_bundle(self):
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary)
            (bundle / "base_metrics.json").write_text(
                json.dumps({"physics": {"valid": True}, "max_penetration": 1.0}), encoding="utf-8"
            )
            (bundle / "task_spec.json").write_text(
                json.dumps(
                    {
                        "name": "generic",
                        "metrics": [
                            {
                                "name": "penetration",
                                "path": "metrics.max_penetration",
                                "target": 0.0,
                                "tolerance": 1.0,
                                "mode": "maximum",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (bundle / "objective_settings.json").write_text(json.dumps({"max_penetration": 0.1}), encoding="utf-8")

            result = _baseline(bundle)

        self.assertFalse(result["feasible"])
        self.assertIn("penetration", result["failure_reason"])

    def test_baseline_exposes_uncompressed_objective_for_reporting(self):
        with tempfile.TemporaryDirectory() as temporary:
            bundle = Path(temporary)
            (bundle / "base_metrics.json").write_text(
                json.dumps({"physics": {"valid": True}, "score": 4.0}), encoding="utf-8"
            )
            (bundle / "task_spec.json").write_text(
                json.dumps(
                    {
                        "name": "generic",
                        "metrics": [
                            {
                                "name": "score",
                                "path": "metrics.score",
                                "target": 0.0,
                                "tolerance": 1.0,
                                "mode": "target",
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            (bundle / "objective_settings.json").write_text(
                json.dumps({"cost_weight": 0.0, "metric_weights": {"score": 1.0}}),
                encoding="utf-8",
            )

            result = _baseline(bundle)

        self.assertTrue(result["feasible"])
        self.assertEqual(result["raw_objective"], 3.5)
        self.assertLess(result["objective"], 1000.0)


class TestComparisonSelection(unittest.TestCase):
    def test_explicit_comparison_directories_exclude_other_completed_runs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selected = root / "optimizer-comparison-selected"
            other = root / "optimizer-comparison-other"
            selected.mkdir()
            other.mkdir()
            (selected / "summary.json").write_text("[]", encoding="utf-8")
            (other / "summary.json").write_text("[]", encoding="utf-8")

            summaries = list(_comparison_summaries(root, [Path("optimizer-comparison-selected")]))

        self.assertEqual(summaries, [selected / "summary.json"])


class TestSceneAgentProvenance(unittest.TestCase):
    def test_provenance_selects_the_scene_authoring_event_stream(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = root / "objective"
            source = root / "mcp-repaired" / "08_liquid_pour" / "workspace" / "result" / "scene.json"
            bundle.mkdir()
            (bundle / "provenance.json").write_text(json.dumps({"source_scene": str(source)}), encoding="utf-8")

            events = _scene_agent_events(root, bundle, "08_liquid_pour")

        self.assertEqual(events, root / "mcp-repaired" / "08_liquid_pour" / "codex-events.jsonl")


if __name__ == "__main__":
    unittest.main()
