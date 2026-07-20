# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for semantic validation of Agent-authored objective contracts."""

from __future__ import annotations

import unittest

from ca_framework.metrics import TaskMetricSpec, TaskSpec
from evaluation.objective_contract import (
    metric_paths,
    physics_metric_semantics,
    validate_objective_settings,
    validate_task_semantics,
    validate_task_spec,
)


class TestObjectiveContract(unittest.TestCase):
    def test_metric_paths_use_the_full_simulation_metrics_prefix(self):
        metrics = {"physics": {"valid": True}, "fluid": {"water": {"relative_mass_change": 0.02}}}

        self.assertEqual(metric_paths(metrics), ["metrics.fluid.water.relative_mass_change"])

    def test_physics_metric_semantics_prefers_peak_divergence(self):
        metrics = {"fluid": {"water": {"max_divergence": 2.0, "peak_divergence": 5.0}}}

        self.assertEqual(
            physics_metric_semantics(metrics),
            [
                {
                    "constraint": "fluid.water.divergence",
                    "path": "metrics.fluid.water.peak_divergence",
                    "baseline_value": 5.0,
                    "meaning": "maximum fluid divergence over the simulated trajectory",
                }
            ],
        )

    def test_physics_metric_semantics_omits_inapplicable_emitter_divergence(self):
        metrics = {
            "fluid": {
                "water": {
                    "peak_divergence": 100.0,
                    "divergence_constraint_applicable": False,
                }
            }
        }

        self.assertEqual(physics_metric_semantics(metrics), [])

    def test_task_spec_rejects_missing_or_unprefixed_metric_paths(self):
        metrics = {"fluid": {"water": {"relative_mass_change": 0.02}}}
        unprefixed = TaskSpec("bad", (TaskMetricSpec("mass", "fluid.water.relative_mass_change", 0.0),))
        missing = TaskSpec("bad", (TaskMetricSpec("mass", "metrics.fluid.water.mass", 0.0),))

        self.assertIn("must start", validate_task_spec(unprefixed, metrics)[0])
        self.assertIn("Unknown task metric path", validate_task_spec(missing, metrics)[0])

    def test_task_spec_accepts_a_real_baseline_metric_path(self):
        metrics = {"fluid": {"water": {"relative_mass_change": 0.02}}}
        task = TaskSpec("mass", (TaskMetricSpec("mass", "metrics.fluid.water.relative_mass_change", 0.0),))

        self.assertEqual(validate_task_spec(task, metrics), [])

    def test_objective_settings_reject_metric_paths_as_threshold_keys(self):
        task = TaskSpec("mass", (TaskMetricSpec("mass", "metrics.fluid.water.relative_mass_change", 0.0),))

        self.assertIn(
            "must name a task_spec metric",
            validate_objective_settings(task, {"metric_thresholds": {"metrics.fluid.water.relative_mass_change": 1.0}})[
                0
            ],
        )

    def test_objective_settings_reject_scene_pair_hard_constraint(self):
        task = TaskSpec("contact", (TaskMetricSpec("duration", "metrics.contacts", 1.0),))

        errors = validate_objective_settings(
            task,
            {"minimum_coupling_contact_duration": {"rigid-cloth:any|any": 0.5}},
        )

        self.assertTrue(any("scene-pair-specific" in error for error in errors))

    def test_semantic_validation_rejects_resource_metrics(self):
        task = TaskSpec("domino", (TaskMetricSpec("cost", "metrics.timings.simulation_seconds", 0.0),))

        errors = validate_task_semantics(task)

        self.assertTrue(any("resource/runtime" in error for error in errors))

    def test_semantic_validation_rejects_physics_validity_as_task_outcome(self):
        task = TaskSpec(
            "invalid",
            (TaskMetricSpec("failures", "metrics.physics.failure_count", 0.0),),
        )

        errors = validate_task_semantics(task)

        self.assertTrue(any("not the requested outcome" in error for error in errors))

    def test_objective_settings_reject_non_positive_explicit_task_scalars(self):
        task = TaskSpec("contact", (TaskMetricSpec("duration", "metrics.contacts", 1.0),))

        errors = validate_objective_settings(
            task,
            {"metric_weights": {"duration": 0.0}, "metric_thresholds": {"duration": 0.0}},
        )

        self.assertEqual(len([error for error in errors if "must be positive" in error]), 2)

    def test_objective_settings_must_accept_framework_valid_baseline(self):
        task = TaskSpec("splash", (TaskMetricSpec("extent", "metrics.extent", 1.0),))
        metrics = {
            "extent": 0.0,
            "physics": {"valid": True},
            "fluid": {},
            "cloth": {},
            "max_penetration": 0.0,
            "coupling": {"exchange_energy_error": 200.0},
        }

        errors = validate_objective_settings(
            task,
            {"max_exchange_energy_error": 100.0},
            metrics,
        )

        self.assertTrue(any("framework-valid baseline" in error for error in errors))

    def test_objective_settings_require_reproduction_margin_around_valid_baseline(self):
        task = TaskSpec("cloth", (TaskMetricSpec("sag", "metrics.sag", 1.0),))
        metrics = {
            "sag": 0.0,
            "physics": {"valid": True},
            "fluid": {},
            "cloth": {
                "sheet": {
                    "finite": True,
                    "max_edge_length_ratio": 1.2,
                    "min_triangle_area_ratio": 0.9,
                    "max_triangle_area_ratio": 1.4,
                    "flipped_triangles": 0,
                }
            },
            "max_penetration": 0.0,
            "coupling": {},
        }

        errors = validate_objective_settings(
            task,
            {
                "max_cloth_edge_ratio": 1.25,
                "min_cloth_area_ratio": 0.88,
                "max_cloth_area_ratio": 1.45,
            },
            metrics,
        )

        margin_errors = [error for error in errors if "reproduction margin" in error]
        self.assertEqual(len(margin_errors), 3)

    def test_fraction_at_natural_upper_bound_does_not_require_impossible_margin(self):
        task = TaskSpec("membrane", (TaskMetricSpec("sag", "metrics.sag", 1.0),))
        metrics = {
            "sag": 0.0,
            "physics": {"valid": True},
            "fluid": {
                "water": {
                    "phase": "liquid",
                    "particles": 10,
                    "cloth_penetration_count": {"membrane": 10},
                }
            },
            "cloth": {},
            "coupling": {},
        }

        errors = validate_objective_settings(
            task,
            {"max_liquid_cloth_penetration_fraction": 1.0},
            metrics,
        )

        self.assertFalse(any("penetration_fraction" in error for error in errors))

    def test_semantic_validation_rejects_inapplicable_mass_conservation_metric(self):
        metrics = {
            "fluid": {
                "water": {
                    "relative_mass_change": 2.0,
                    "mass_conservation_applicable": False,
                }
            }
        }
        task = TaskSpec(
            "pour",
            (TaskMetricSpec("mass", "metrics.fluid.water.relative_mass_change", 0.0),),
        )

        errors = validate_task_semantics(task, metrics)

        self.assertTrue(any("not applicable" in error for error in errors))

    def test_semantic_validation_rejects_task_spec_without_baseline_improvement_signal(self):
        task = TaskSpec(
            "already-satisfied",
            (TaskMetricSpec("stable", "metrics.kinematics.max_speed", 1.0, mode="maximum"),),
        )

        errors = validate_task_semantics(task, {"kinematics": {"max_speed": 0.1}})

        self.assertTrue(any("cannot measure an optimization improvement" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
