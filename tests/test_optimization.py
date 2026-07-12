# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for schema-v3 optimization parameter primitives."""

from __future__ import annotations

import unittest

from ca_framework.optimization import ParameterSpec, activate_parameters, apply_parameter_patch
from ca_framework.scene import ObjectFluid, ObjectRigid, Scene
from evaluation.benchmark_optimizers import BENCHMARK_SCENES, OPTIMIZERS, benchmark_plan
from evaluation.canonical import canonical_scenes


class TestParameterPatch(unittest.TestCase):
    def setUp(self) -> None:
        self.scene = Scene(name="patch-test", objects={"ball": ObjectRigid(id="ball", shape="sphere")})

    def test_patch_is_deeply_immutable(self) -> None:
        candidate = apply_parameter_patch(
            self.scene,
            {
                "objects.ball.physical_material.restitution": 0.6,
                "settings.substeps": 4,
            },
        )
        self.assertEqual(candidate.objects["ball"].physical_material.restitution, 0.6)
        self.assertEqual(candidate.settings.substeps, 4)
        self.assertEqual(self.scene.objects["ball"].physical_material.restitution, 0.0)
        self.assertEqual(self.scene.settings.substeps, 8)
        candidate.objects["ball"].transform.position = (1.0, 2.0, 3.0)
        self.assertEqual(self.scene.objects["ball"].transform.position, (0.0, 0.0, 0.0))

    def test_independent_candidates_do_not_share_state(self) -> None:
        first = apply_parameter_patch(self.scene, {"settings.coupling.relaxation": 0.2})
        second = apply_parameter_patch(self.scene, {"settings.coupling.relaxation": 0.9})
        self.assertEqual(first.settings.coupling.relaxation, 0.2)
        self.assertEqual(second.settings.coupling.relaxation, 0.9)
        self.assertEqual(self.scene.settings.coupling.relaxation, 0.7)

    def test_specs_validate_type_range_and_registration(self) -> None:
        spec = ParameterSpec(
            name="substeps",
            path="settings.substeps",
            scope="solver",
            kind="int",
            lower=1,
            upper=16,
        )
        candidate = apply_parameter_patch(self.scene, {spec.path: 12}, {spec.path: spec})
        self.assertEqual(candidate.settings.substeps, 12)
        with self.assertRaises(ValueError):
            apply_parameter_patch(self.scene, {spec.path: 0}, {spec.path: spec})
        with self.assertRaises(ValueError):
            apply_parameter_patch(self.scene, {"settings.fps": 30}, {spec.path: spec})

    def test_invalid_path_and_storage_type_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            apply_parameter_patch(self.scene, {"settings.missing": 1})
        with self.assertRaises(ValueError):
            apply_parameter_patch(self.scene, {"settings.substeps": 1.5})


class TestSchemaV3Migration(unittest.TestCase):
    def test_v2_solver_fields_migrate_to_both_solid_solvers(self) -> None:
        scene = Scene.from_dict(
            {
                "name": "legacy",
                "schema_version": 2,
                "settings": {"solver": "vbd", "solver_iterations": 7},
            }
        )
        self.assertEqual(scene.schema_version, 3)
        self.assertEqual(scene.settings.rigid.method, "vbd")
        self.assertEqual(scene.settings.cloth.method, "vbd")
        self.assertEqual(scene.settings.rigid.iterations, 7)
        self.assertEqual(scene.settings.cloth.iterations, 7)
        self.assertEqual(scene.settings.cloth.strain_limit_iterations, 5)
        serialized = scene.to_dict()
        self.assertNotIn("solver", serialized["settings"])
        self.assertNotIn("solver_iterations", serialized["settings"])


class TestParameterActivation(unittest.TestCase):
    def test_pure_rigid_scene_disables_cloth_and_fluid_settings(self) -> None:
        scene = Scene(name="rigid", objects={"ball": ObjectRigid(id="ball")})
        specs = (
            ParameterSpec("rigid", "settings.rigid.iterations", "solver", "int", 1, 20),
            ParameterSpec("cloth", "settings.cloth.iterations", "solver", "int", 1, 20),
            ParameterSpec("fluid", "settings.fluid.pressure_iterations", "solver", "int", 1, 100),
        )
        active = activate_parameters(scene, specs)
        self.assertEqual([item.name for item in active], ["rigid"])

    def test_tpe_space_is_capped_at_fifteen(self) -> None:
        scene = Scene(name="rigid", objects={"ball": ObjectRigid(id="ball")})
        specs = tuple(
            ParameterSpec(f"p{index}", "settings.substeps", "solver", "int", 1, 20)
            for index in range(20)
        )
        self.assertEqual(len(activate_parameters(scene, specs, optimizer="tpe")), 15)

    def test_smoke_drag_is_active_for_rigid_smoke_without_cloth(self) -> None:
        scene = Scene(
            name="rigid-smoke",
            objects={"body": ObjectRigid("body"), "smoke": ObjectFluid("smoke", phase="smoke")},
        )
        spec = ParameterSpec(
            "smoke_drag",
            "settings.coupling.smoke_drag_coefficient",
            "coupling",
            "float",
            0.0,
            2.0,
        )
        self.assertEqual(activate_parameters(scene, (spec,)), (spec,))


class TestBenchmarkMatrix(unittest.TestCase):
    def test_all_representative_scene_sampler_plans_patch_valid_paths(self) -> None:
        scenes = canonical_scenes()
        for scene_key, canonical_name in BENCHMARK_SCENES.items():
            for optimizer in OPTIMIZERS:
                plan = benchmark_plan(
                    scene_key, optimizer, 0, trials=2, max_wall_time_sec=10.0
                )
                values = {
                    spec.path: (spec.lower + spec.upper) / 2.0
                    for spec in plan.parameters
                    if spec.kind == "float"
                }
                values.update(
                    {
                        spec.path: int((spec.lower + spec.upper) // 2)
                        for spec in plan.parameters
                        if spec.kind == "int"
                    }
                )
                apply_parameter_patch(
                    scenes[canonical_name], values, {spec.path: spec for spec in plan.parameters}
                )


if __name__ == "__main__":
    unittest.main()
