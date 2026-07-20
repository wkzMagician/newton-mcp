# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for common telemetry and objective contracts."""

from __future__ import annotations

import math
import unittest
from types import SimpleNamespace

import numpy as np

from ca_framework.coupling import CouplerRigidFluid, CouplingExchange
from ca_framework.metrics import MetricPipeline, ObjectiveSettings, TaskMetricSpec, TaskSpec
from ca_framework.scene import FrameTelemetry, ObjectContainer, ObjectRigid, Scene, SimulationTelemetry, Transform
from ca_framework.scene.executor import SceneExecutorLocal


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


class TestContainerRetentionMetric(unittest.TestCase):
    def test_measures_fraction_inside_rotated_container(self) -> None:
        container = ObjectContainer(
            "receiver",
            inner_size=(2.0, 2.0, 2.0),
            transform=Transform(position=(1.0, 0.0, 0.0), rotation=(0.0, 0.0, 1.0, 0.0)),
        )
        positions = np.asarray(
            [
                [2.0, 0.0, 1.0],
                [0.0, 0.0, 1.0],
                [1.0, 0.0, -0.1],
                [1.0, -1.2, 1.0],
            ],
            dtype=np.float64,
        )

        fraction = SceneExecutorLocal._container_retention_fraction(positions, container)

        self.assertEqual(fraction, 0.5)

    def test_layout_metrics_expose_final_positions_and_pair_distances(self) -> None:
        scene = Scene(
            "layout",
            objects={
                "left": ObjectRigid("left"),
                "right": ObjectRigid("right"),
            },
        )
        metrics = SceneExecutorLocal._layout_metrics(
            scene,
            {
                "left": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                "right": [[0.0, 2.0, 0.0], [4.0, 3.0, 0.0]],
            },
        )

        self.assertEqual(metrics["final_body_position"]["left"]["x"], 1.0)
        self.assertEqual(metrics["body_displacement"]["left"], 1.0)
        self.assertEqual(metrics["final_body_pair_distance"]["left|right"], math.sqrt(18.0))
        self.assertEqual(metrics["final_body_pair_horizontal_distance"]["left|right"], math.sqrt(18.0))

    def test_orientation_metrics_measure_tilt_from_world_up(self) -> None:
        scene = Scene("tilt", objects={"box": ObjectRigid("box")})
        half_angle = math.pi / 4.0
        state = SimpleNamespace(
            body_q=SimpleNamespace(
                numpy=lambda: np.asarray([[0.0, 0.0, 0.0, 0.0, math.sin(half_angle), 0.0, math.cos(half_angle)]])
            ),
            body_qd=SimpleNamespace(numpy=lambda: np.asarray([[3.0, 4.0, 0.0, 0.0, 0.0, 2.0]])),
        )

        metrics = SceneExecutorLocal._orientation_metrics(
            scene,
            SimpleNamespace(body_indices={"box": 0}),
            state,
        )

        self.assertAlmostEqual(metrics["final_body_up_alignment"]["box"], 0.0, places=7)
        self.assertAlmostEqual(metrics["final_body_tilt_angle"]["box"], math.pi / 2.0, places=7)
        self.assertAlmostEqual(metrics["final_body_linear_speed"]["box"], 5.0, places=7)
        self.assertAlmostEqual(metrics["final_body_angular_speed"]["box"], 2.0, places=7)


class TestMetricPipeline(unittest.TestCase):
    def test_emitter_source_divergence_is_diagnostic_not_a_constraint(self) -> None:
        result = MetricPipeline(ObjectiveSettings(max_fluid_divergence=10.0)).evaluate(
            {
                "metrics": {
                    "physics": {"valid": True},
                    "fluid": {
                        "water": {
                            "finite": True,
                            "peak_divergence": 100.0,
                            "divergence_constraint_applicable": False,
                            "mass_conservation_applicable": False,
                        }
                    },
                }
            }
        )

        self.assertTrue(result.feasible)
        self.assertNotIn("fluid.water.divergence", result.constraints)
        self.assertNotIn("fluid.water.divergence", result.metrics)

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

    def test_feasible_objective_stays_below_infeasible_range_for_large_task_loss(self) -> None:
        settings = ObjectiveSettings(infeasible_base=100.0)
        feasible = MetricPipeline(settings).evaluate(
            {
                "metrics": {
                    "max_penetration": 0.0,
                    "physics": {"valid": True},
                    "fluid": {},
                    "cloth": {},
                }
            },
            task_metrics={"large_loss": 1.0e9},
        )
        infeasible = MetricPipeline(settings).evaluate(
            {
                "metrics": {
                    "max_penetration": 0.0,
                    "physics": {"valid": False},
                    "fluid": {},
                    "cloth": {},
                }
            }
        )

        self.assertTrue(feasible.feasible)
        self.assertLess(feasible.objective, settings.infeasible_base)
        self.assertGreaterEqual(infeasible.objective, settings.infeasible_base)
        self.assertLess(feasible.objective, infeasible.objective)

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

    def test_stability_metrics_are_part_of_physics_loss(self) -> None:
        result = MetricPipeline().evaluate(
            {
                "metrics": {
                    "max_penetration": 0.0,
                    "physics": {"valid": True},
                    "fluid": {},
                    "cloth": {
                        "sheet": {
                            "finite": True,
                            "max_edge_length_ratio": 1.5,
                            "min_triangle_area_ratio": 0.7,
                            "max_triangle_area_ratio": 1.3,
                            "flipped_triangle_count": 0,
                            "final_residual_speed": 0.0,
                        }
                    },
                    "coupling": {
                        "impulse_balance_error": 0.01,
                        "angular_impulse_balance_error": 0.01,
                        "interface_velocity_residual": 2.0,
                        "exchange_energy_error": 5.0,
                    },
                }
            }
        )
        self.assertTrue(result.feasible)
        self.assertGreater(result.metrics["physics_loss"], 0.0)
        self.assertIn("cloth.sheet.edge_distortion", result.metrics)
        self.assertIn("coupling.interface_velocity_residual", result.metrics)

    def test_rigid_cloth_penetration_and_contact_duration_are_hard_constraints(self) -> None:
        result = MetricPipeline().evaluate(
            {
                "metrics": {
                    "max_penetration": 0.0,
                    "physics": {"valid": True},
                    "fluid": {},
                    "cloth": {},
                    "coupling": {
                        "max_rigid_cloth_penetration": 0.03,
                        "contact_duration": {"rigid-cloth:cloth|ball": 0.2},
                    },
                }
            },
            coupling_metrics={
                "max_rigid_cloth_penetration": 0.03,
                "contact_duration": {"rigid-cloth:cloth|ball": 0.2},
            },
        )
        self.assertFalse(result.feasible)
        self.assertIn("coupling.rigid_cloth_penetration", result.failure_reason)

        duration_result = MetricPipeline(
            ObjectiveSettings(minimum_coupling_contact_duration={"rigid-cloth:cloth|ball": 0.5})
        ).evaluate(
            {
                "metrics": {
                    "max_penetration": 0.0,
                    "physics": {"valid": True},
                    "fluid": {},
                    "cloth": {},
                    "coupling": {
                        "max_rigid_cloth_penetration": 0.0,
                        "contact_duration": {"rigid-cloth:cloth|ball": 0.2},
                    },
                }
            }
        )
        self.assertFalse(duration_result.feasible)
        self.assertIn("coupling.contact_duration.rigid-cloth:cloth|ball", duration_result.failure_reason)

    def test_liquid_cloth_penetration_fraction_is_a_hard_constraint(self) -> None:
        result = MetricPipeline(
            ObjectiveSettings(max_liquid_cloth_penetration_fraction=0.1)
        ).evaluate(
            {
                "metrics": {
                    "max_penetration": 0.0,
                    "physics": {"valid": True},
                    "cloth": {},
                    "fluid": {
                        "water": {
                            "phase": "liquid",
                            "finite": True,
                            "particles": 100,
                            "peak_divergence": 0.0,
                            "relative_mass_change": 0.0,
                            "cloth_penetration_count": {"membrane": 12},
                        }
                    },
                }
            }
        )
        self.assertFalse(result.feasible)
        self.assertIn("fluid.water.cloth.membrane.penetration_fraction", result.failure_reason)


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

    def test_missing_event_metric_becomes_task_loss(self) -> None:
        spec = TaskSpec("contact-task", (TaskMetricSpec("contact", "metrics.contact_time", 1.0),))
        losses = spec.evaluate({"metrics": {}})
        self.assertEqual(losses["contact"], 10.0)


if __name__ == "__main__":
    unittest.main()
