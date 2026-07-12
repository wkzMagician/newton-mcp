# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for common telemetry and objective contracts."""

from __future__ import annotations

import unittest

import numpy as np

from ca_framework.coupling import CouplerRigidFluid, CouplingExchange
from ca_framework.metrics import MetricPipeline, TaskMetricSpec, TaskSpec
from ca_framework.scene import FrameTelemetry, SimulationTelemetry


class TestCouplingExchange(unittest.TestCase):
    def test_balanced_impulses_have_zero_error(self) -> None:
        exchange = CouplingExchange(
            pair_type="rigid-fluid",
            object_a="ball",
            object_b="water",
            linear_impulse_a=np.array([1.0, 0.0, 0.0]),
            linear_impulse_b=np.array([-1.0, 0.0, 0.0]),
            angular_impulse_a=np.zeros(3),
            angular_impulse_b=np.zeros(3),
            interface_residual=0.01,
            penetration=0.0,
        )
        self.assertEqual(exchange.impulse_balance_error, 0.0)
        self.assertEqual(exchange.angular_impulse_balance_error, 0.0)
        self.assertEqual(exchange.to_dict()["linear_impulse_a"], [1.0, 0.0, 0.0])

    def test_rigid_fluid_adapter_emits_balanced_exchange(self) -> None:
        class Solver:
            rigid_linear_impulse = {2: np.array([0.0, 3.0, 0.0])}
            rigid_angular_impulse = {2: np.array([0.0, 0.0, 1.0])}
            rigid_pressure_impulse = {2: np.array([0.0, 2.0, 0.0])}
            rigid_stabilization_impulse = {2: np.array([0.0, 1.0, 0.0])}
            impulse_clip_count = 1
            solid_body = np.full((2, 2, 2), -1)
            solid_velocity = np.zeros((2, 2, 2, 3))
            fluid = np.zeros((2, 2, 2), dtype=bool)
            grid_velocity = np.zeros((2, 2, 2, 3))

            @staticmethod
            def _shifted(values, axis, direction, fill):
                return np.full_like(values, fill)

        solver = Solver()
        exchanges = CouplerRigidFluid().collect("water", solver, {2: "ball"})
        self.assertEqual(len(exchanges), 1)
        self.assertEqual(exchanges[0].object_a, "ball")
        self.assertEqual(exchanges[0].impulse_balance_error, 0.0)
        diagnostics = CouplerRigidFluid.diagnostics("water", solver)
        self.assertEqual(diagnostics["impulse_clip_count"], 1)


class TestTelemetry(unittest.TestCase):
    def test_frame_records_are_json_serializable_values(self) -> None:
        telemetry = SimulationTelemetry([FrameTelemetry(frame=2, substep=3, time=0.25)])
        records = telemetry.to_records()
        self.assertEqual(records[0]["frame"], 2)
        self.assertEqual(records[0]["solver_stats"]["iterations"], 0)


class TestMetricPipeline(unittest.TestCase):
    def test_feasible_result_uses_normalized_robust_loss(self) -> None:
        result = MetricPipeline().evaluate(
            {
                "metrics": {
                    "max_penetration": 0.01,
                    "physics": {"valid": True},
                    "cloth": {
                        "sheet": {
                            "finite": True,
                            "max_edge_length_ratio": 1.1,
                            "min_triangle_area_ratio": 0.9,
                            "max_triangle_area_ratio": 1.1,
                            "flipped_triangle_count": 0,
                            "final_residual_speed": 0.2,
                        }
                    },
                    "fluid": {},
                }
            },
            runtime_sec=1.0,
            task_metrics={"target_error": 0.5},
        )
        self.assertTrue(result.feasible)
        self.assertEqual(result.status, "completed")
        self.assertGreater(result.objective, 0.0)
        self.assertLess(result.objective, 1000.0)

    def test_infeasible_result_dominates_quality_score(self) -> None:
        result = MetricPipeline().evaluate(
            {
                "metrics": {
                    "max_penetration": 0.2,
                    "physics": {"valid": False},
                    "cloth": {},
                    "fluid": {},
                }
            }
        )
        self.assertFalse(result.feasible)
        self.assertGreaterEqual(result.objective, 1000.0)
        self.assertIn("physics_valid", result.failure_reason)

    def test_nonfinite_metric_is_infeasible(self) -> None:
        result = MetricPipeline().evaluate(
            {
                "metrics": {
                    "max_penetration": float("nan"),
                    "physics": {"valid": True},
                    "cloth": {},
                    "fluid": {},
                }
            }
        )
        self.assertFalse(result.feasible)


class TestTaskSpec(unittest.TestCase):
    def test_path_metrics_support_target_and_bounds(self) -> None:
        spec = TaskSpec(
            "contact-task",
            (
                TaskMetricSpec("contact", "metrics.contacts", 1.0, mode="minimum"),
                TaskMetricSpec("penetration", "metrics.max_penetration", 0.01, mode="maximum"),
            ),
        )
        losses = spec.evaluate({"metrics": {"contacts": 2, "max_penetration": 0.03}})
        self.assertEqual(losses["contact"], 0.0)
        self.assertAlmostEqual(losses["penetration"], 0.02)


if __name__ == "__main__":
    unittest.main()
