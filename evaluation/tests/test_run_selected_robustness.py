# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for repeated finalist evaluation."""

from __future__ import annotations

import unittest

from evaluation.run_selected_robustness import summarize_repeats


class TestRobustnessSummary(unittest.TestCase):
    def test_reports_sample_standard_deviation_and_failures(self) -> None:
        summary = summarize_repeats(
            [
                {"feasible": True, "raw_objective": 1.0},
                {"feasible": True, "raw_objective": 3.0},
                {"feasible": False, "raw_objective": None},
            ]
        )

        self.assertEqual(summary["feasibility_rate"], 2 / 3)
        self.assertEqual(summary["raw_objective_mean"], 2.0)
        self.assertAlmostEqual(summary["raw_objective_std"], 2**0.5)


if __name__ == "__main__":
    unittest.main()
