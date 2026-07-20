# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest

from evaluation.plot_optimizer_analysis import _robustness_cases, _total_tokens


class TestPlotOptimizerAnalysis(unittest.TestCase):
    def test_robustness_cases_accepts_list_and_wrapped_payload(self) -> None:
        cases = [{"case": "03_ramp_bounce"}]
        self.assertEqual(_robustness_cases(cases), cases)
        self.assertEqual(_robustness_cases({"cases": cases}), cases)
        self.assertEqual(_robustness_cases({"comparisons": cases}), cases)

    def test_total_tokens_does_not_double_count_cached_input(self) -> None:
        usage = {"input_tokens": 100, "cached_input_tokens": 80, "output_tokens": 20}
        self.assertEqual(_total_tokens(usage), 120.0)


if __name__ == "__main__":
    unittest.main()
