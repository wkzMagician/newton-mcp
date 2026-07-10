# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import json
import tempfile
import time
import unittest
from pathlib import Path

from ca_framework.mcp import SceneTools
from ca_framework.scene import Scene, SceneExecutorLocal, SceneStore, validate_scene
from newton.solvers import SolverFluidAPIC, SolverFluidSmoke
from newton.viewer import ViewerFluidGL


class TestSceneFramework(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.tools = SceneTools(SceneStore(self.root / "scenes"), SceneExecutorLocal())
        self.tools.create_scene("demo")

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_create_edit_and_round_trip_scene(self):
        self.tools.add_object(
            "demo",
            {
                "id": "ball",
                "kind": "rigid",
                "shape": "sphere",
                "size": [0.5, 0.5, 0.5],
                "transform": {"position": [0.0, 0.0, 2.0]},
            },
        )
        self.tools.add_field(
            "demo",
            {"id": "wind", "kind": "uniform", "vector": [1.0, 0.0, 0.0], "object_ids": ["ball"]},
        )

        scene = self.tools.get_scene("demo")

        self.assertEqual(scene["objects"]["ball"]["shape"], "sphere")
        self.assertEqual(scene["fields"]["wind"]["object_ids"], ["ball"])
        self.assertEqual(SceneStore(self.root / "scenes").load("demo").name, "demo")

    def test_rejects_dangling_references(self):
        self.tools.add_object("demo", {"id": "a", "kind": "rigid"})
        with self.assertRaisesRegex(ValueError, "Unknown object id: missing"):
            self.tools.add_constraint(
                "demo",
                {"id": "link", "kind": "distance", "object_a": "a", "object_b": "missing"},
            )

    def test_rejects_removing_referenced_object(self):
        self.tools.add_object("demo", {"id": "a", "kind": "rigid"})
        self.tools.add_object("demo", {"id": "b", "kind": "rigid"})
        self.tools.add_constraint("demo", {"id": "link", "kind": "distance", "object_a": "a", "object_b": "b"})

        with self.assertRaisesRegex(ValueError, "constraint:link"):
            self.tools.remove_item("demo", "objects", "a")

    def test_rejects_kind_change_and_dangling_update(self):
        self.tools.add_object("demo", {"id": "a", "kind": "rigid"})
        self.tools.add_object("demo", {"id": "b", "kind": "rigid"})
        self.tools.add_constraint("demo", {"id": "link", "kind": "distance", "object_a": "a", "object_b": "b"})

        with self.assertRaisesRegex(ValueError, "kinds cannot be changed"):
            self.tools.update_item("demo", "objects", "a", {"kind": "cloth"})
        with self.assertRaisesRegex(ValueError, "Unknown object id: missing"):
            self.tools.update_item("demo", "constraints", "link", {"object_b": "missing"})

    def test_exports_json(self):
        output = self.root / "exports" / "demo.json"

        result = self.tools.export_scene("demo", str(output))

        self.assertEqual(result["status"], "completed")
        self.assertEqual(json.loads(output.read_text())["name"], "demo")

    def test_scene_name_cannot_escape_workspace(self):
        with self.assertRaises(ValueError):
            self.tools.create_scene("../outside")

    def test_migrates_v1_material_and_motion(self):
        scene = Scene.from_dict(
            {
                "name": "old",
                "schema_version": 1,
                "objects": {
                    "ball": {
                        "id": "ball",
                        "kind": "rigid",
                        "dynamic": False,
                        "material": {"density": 500.0, "friction": 0.2, "color": [1.0, 0.0, 0.0, 1.0]},
                    }
                },
            }
        )
        ball = scene.objects["ball"]
        self.assertEqual(scene.schema_version, 2)
        self.assertEqual(ball.motion, "static")
        self.assertEqual(ball.physical_material.friction_static, 0.2)
        self.assertEqual(ball.visual_material.color, [1.0, 0.0, 0.0, 1.0])

    def test_validates_fluid_solver_and_particle_budget(self):
        scene = Scene.from_dict(
            {
                "name": "liquid",
                "schema_version": 2,
                "objects": {
                    "water": {
                        "id": "water",
                        "kind": "fluid",
                        "phase": "liquid",
                        "size": [1.0, 1.0, 1.0],
                        "particle_spacing": 0.01,
                    }
                },
                "settings": {"solver": "smoke", "max_particles": 10},
            }
        )
        report = validate_scene(scene)
        self.assertFalse(report["valid"])
        self.assertEqual({item["code"] for item in report["diagnostics"]}, {"solver_mismatch", "particle_budget"})
        self.assertEqual(report["pipeline"], ["apic"])

    def test_transactional_patch_preview_and_program_export(self):
        self.tools.apply_scene_patch(
            "demo",
            {
                "objects": {
                    "smoke": {
                        "id": "smoke",
                        "kind": "fluid",
                        "phase": "smoke",
                        "size": [1.0, 1.0, 2.0],
                        "grid_resolution": [8, 8, 16],
                    }
                },
                "settings": {"duration": 0.1, "fps": 20},
            },
        )
        preview = self.tools.preview_scene("demo")
        self.assertEqual(preview["pipeline"], ["smoke"])
        self.assertTrue(preview["metrics"]["fluid"]["smoke"]["finite"])

        program = self.root / "exports" / "demo.py"
        scene_json = self.root / "exports" / "demo.json"
        result = self.tools.export_program("demo", str(program), str(scene_json))
        self.assertEqual(result["status"], "completed")
        self.assertNotIn("newton._src", program.read_text())
        self.assertEqual(json.loads(scene_json.read_text()), self.tools.get_scene("demo"))

    def test_async_job_reports_completion(self):
        self.tools.apply_scene_patch(
            "demo", {"settings": {"duration": 0.05}, "render": {"resolution": [16, 16], "fps": 10}}
        )
        job = self.tools.run_scene("demo", str(self.root / "demo.mp4"))
        for _ in range(100):
            result = self.tools.get_job(job["job_id"])
            if result["status"] not in {"queued", "running"}:
                break
            time.sleep(0.01)
        self.assertIn(result["status"], {"completed", "render_failed"})

    def test_public_fluid_api(self):
        self.assertTrue(SolverFluidAPIC.__name__.startswith("SolverFluid"))
        self.assertTrue(SolverFluidSmoke.__name__.startswith("SolverFluid"))
        self.assertEqual(ViewerFluidGL.__name__, "FluidViewerGL")


if __name__ == "__main__":
    unittest.main()
