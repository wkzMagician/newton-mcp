# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import ctypes
import json
import math
import runpy
import tempfile
import time
import unittest
from pathlib import Path
from threading import Event
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from ca_framework.mcp import SceneTools
from ca_framework.scene import Scene, SceneExecutorLocal, SceneStore, validate_scene
from ca_framework.scene.executor import _decode_gl_string, _physics_validation
from newton.solvers import SolverBase, SolverExercise5Fluid, SolverFluidAPIC, SolverFluidSmoke, SolverSemiImplicit
from newton.viewer import ViewerFluidGL


class TestSceneFramework(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.tools = SceneTools(SceneStore(self.root / "scenes"), SceneExecutorLocal())
        self.tools.create_scene("demo")

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_decode_gl_string_from_ctypes_pointer(self):
        value = ctypes.cast(ctypes.create_string_buffer(b"NVIDIA Corporation"), ctypes.POINTER(ctypes.c_ubyte))

        self.assertEqual(_decode_gl_string(value), "NVIDIA Corporation")

    def test_decode_gl_string_from_bytes_or_null(self):
        self.assertEqual(_decode_gl_string(b"OpenGL 4.6"), "OpenGL 4.6")
        self.assertEqual(_decode_gl_string(None), "unknown")

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

    def test_set_camera_persists_valid_fixed_view(self):
        camera = self.tools.set_camera(
            "demo",
            position=(3.0, -4.0, 2.0),
            target=(0.0, 0.0, 0.5),
            up=(0.0, 0.0, 1.0),
            field_of_view=50.0,
        )

        self.assertFalse(camera["auto_frame"])
        self.assertEqual(camera["position"], (3.0, -4.0, 2.0))
        stored = self.tools.get_scene("demo")["render"]["camera"]
        self.assertEqual(stored["position"], [3.0, -4.0, 2.0])
        self.assertEqual(stored["target"], [0.0, 0.0, 0.5])
        self.assertFalse(stored["auto_frame"])

    def test_set_camera_rejects_degenerate_view(self):
        with self.assertRaisesRegex(ValueError, "degenerate_camera"):
            self.tools.set_camera("demo", position=(1.0, 1.0, 1.0), target=(1.0, 1.0, 1.0))

    def test_physics_validation_rejects_instability_diagnostics(self):
        result = _physics_validation(
            [
                {"code": "pressure_nonconvergence", "frame": 4},
                {"code": "state_escape", "frame": 5},
                {"code": "render_fallback", "frame": 5},
            ]
        )

        self.assertFalse(result["valid"])
        self.assertEqual(result["failure_count"], 2)
        self.assertEqual(result["failure_codes"], ["pressure_nonconvergence", "state_escape"])

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

    def test_typed_updates_reject_kind_change_and_dangling_reference(self):
        self.tools.add_object("demo", {"id": "a", "kind": "rigid"})
        self.tools.add_object("demo", {"id": "b", "kind": "rigid"})
        self.tools.add_constraint("demo", {"id": "link", "kind": "distance", "object_a": "a", "object_b": "b"})

        with self.assertRaisesRegex(ValueError, "kinds cannot be changed"):
            self.tools.update_object("demo", "a", {"kind": "cloth"})
        with self.assertRaisesRegex(ValueError, "Unknown object id: missing"):
            self.tools.update_constraint("demo", "link", {"kind": "distance", "object_b": "missing"})

    def test_typed_updates_only_change_explicit_fields(self):
        self.tools.add_object("demo", {"id": "ball", "kind": "rigid", "shape": "sphere"})
        updated = self.tools.update_object("demo", "ball", {"kind": "rigid", "linear_velocity": [1.0, 2.0, 3.0]})

        self.assertEqual(updated["linear_velocity"], [1.0, 2.0, 3.0])
        self.assertEqual(updated["shape"], "sphere")

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
        output_dir = self.root / "demo-output"
        job = self.tools.run_scene("demo", str(output_dir))
        for _ in range(100):
            result = self.tools.get_job(job["job_id"])
            if result["status"] not in {"queued", "running"}:
                break
            time.sleep(0.01)
        self.assertIn(result["status"], {"completed", "render_failed"})
        self.assertTrue((output_dir / "scene.json").is_file())
        self.assertTrue((output_dir / "program.py").is_file())
        self.assertTrue((output_dir / "metrics.json").is_file())
        self.assertTrue((output_dir / "diagnostics.jsonl").is_file())
        self.assertTrue((output_dir / "cache" / "manifest.json").is_file())

    def test_public_fluid_api(self):
        self.assertTrue(SolverFluidAPIC.__name__.startswith("SolverFluid"))
        self.assertTrue(SolverFluidSmoke.__name__.startswith("SolverFluid"))
        self.assertEqual(ViewerFluidGL.__name__, "FluidViewerGL")
        self.assertTrue(issubclass(SolverFluidAPIC, SolverBase))
        self.assertFalse(issubclass(SolverFluidAPIC, SolverSemiImplicit))
        self.assertIs(SolverExercise5Fluid, SolverFluidSmoke)

    def test_formal_render_does_not_fall_back_when_opengl_fails(self):
        scene = Scene(name="gpu-only")
        cache = self.root / "cache"
        frames = cache / "frames"
        frames.mkdir(parents=True)
        (cache / "manifest.json").write_text(json.dumps({"frames": 1}))
        np.savez(frames / "000000.npz")

        with patch.object(
            SceneExecutorLocal,
            "_render_cache_frames_gl",
            side_effect=RuntimeError("no hardware OpenGL"),
        ):
            result = SceneExecutorLocal._encode_video(scene, self.root / "formal.mp4", cache)

        self.assertEqual(result["status"], "render_failed")
        self.assertEqual(result["diagnostics"][0]["code"], "opengl_render_failed")
        self.assertIn("no hardware OpenGL", result["diagnostics"][0]["message"])
        self.assertFalse((self.root / "formal.mp4").exists())

    def test_cpu_simulation_uses_opengl_host_staging(self):
        scene = Scene(name="cpu-gl")
        output = self.root / "frames"
        output.mkdir()
        (self.root / "manifest.json").write_text(json.dumps({"frames": 1}))
        metadata = {
            "simulation_device": "cpu",
            "render_backend": "opengl_hardware",
            "render_data_path": "host_staging",
            "render_equivalent": True,
        }
        with (
            patch("ca_framework.scene.executor.wp.get_device", return_value=SimpleNamespace(is_cuda=False)),
            patch.object(SceneExecutorLocal, "_render_cache_frames_gl", return_value=metadata) as render_gl,
        ):
            result = SceneExecutorLocal._render_cache_frames(scene, self.root, output)

        render_gl.assert_called_once()
        self.assertEqual(result, metadata)

    def test_compiler_resolves_cloth_selectors_and_container_transform(self):
        scene = Scene.from_dict(
            {
                "name": "compiled",
                "schema_version": 2,
                "objects": {
                    "cloth": {
                        "id": "cloth",
                        "kind": "cloth",
                        "resolution": [3, 2],
                        "pinned": [{"kind": "edge", "edge": "top"}],
                    },
                    "container": {
                        "id": "container",
                        "kind": "container",
                        "motion": "static",
                        "inner_size": [2.0, 2.0, 1.0],
                        "transform": {"position": [10.0, 0.0, 1.0], "scale": [2.0, 1.0, 1.0]},
                    },
                },
            }
        )
        compiled = SceneExecutorLocal().compiler.compile(scene)
        self.assertEqual(compiled.cloth_particle_indices["cloth"], list(range(6)))
        self.assertEqual(compiled.pinned_particles["cloth"], [3, 4, 5])
        self.assertAlmostEqual(compiled.container_colliders["container"][0]["position"][0], 10.0)

    def test_kinematic_keyframes_interpolate_collision_shape_scale(self):
        scene = Scene.from_dict(
            {
                "name": "scaled-kinematic",
                "objects": {
                    "box": {
                        "id": "box",
                        "kind": "rigid",
                        "motion": "kinematic",
                        "shape": "box",
                        "size": [1.0, 1.0, 1.0],
                    }
                },
                "actions": {
                    "move": {
                        "id": "move",
                        "kind": "transform",
                        "object_id": "box",
                        "keyframes": [
                            {"time": 0.0, "transform": {"scale": [1.0, 1.0, 1.0]}},
                            {"time": 1.0, "transform": {"scale": [2.0, 1.0, 1.0]}},
                        ],
                    }
                },
            }
        )
        executor = SceneExecutorLocal()
        compiled = executor.compiler.compile(scene)
        shape_index = compiled.shape_indices["box"][0]

        executor._apply_kinematics(scene, compiled, compiled.state_0, 0.5, 1.0 / 60.0)

        scale = compiled.model.shape_scale.numpy()[shape_index]
        self.assertAlmostEqual(float(scale[0]), float(compiled.initial_shape_scales[shape_index, 0] * 1.5))

    def test_validation_rejects_bad_quaternion_selector_and_times(self):
        scene = Scene.from_dict(
            {
                "name": "invalid",
                "schema_version": 2,
                "objects": {
                    "cloth": {
                        "id": "cloth",
                        "kind": "cloth",
                        "resolution": [2, 2],
                        "transform": {"rotation": [0.0, 0.0, 0.0, 2.0]},
                        "pinned": [{"kind": "indices", "indices": [4]}],
                    }
                },
                "actions": {
                    "move": {
                        "id": "move",
                        "kind": "transform",
                        "object_id": "cloth",
                        "keyframes": [],
                    }
                },
            }
        )
        codes = {item["code"] for item in validate_scene(scene)["diagnostics"]}
        self.assertTrue({"invalid_quaternion", "invalid_selector", "invalid_action_time"} <= codes)

    def test_rigid_simulation_uses_newton_contacts(self):
        scene = Scene.from_dict(
            {
                "name": "drop",
                "schema_version": 2,
                "objects": {
                    "ball": {
                        "id": "ball",
                        "kind": "rigid",
                        "shape": "sphere",
                        "size": [1.0, 1.0, 1.0],
                        "transform": {"position": [0.0, 0.0, 2.0]},
                    }
                },
                "settings": {"duration": 1.0, "fps": 30, "substeps": 4},
            }
        )
        result = SceneExecutorLocal().simulate(scene)
        final_height = result["trajectories"]["ball"][-1][2]
        self.assertGreater(result["metrics"]["contacts"], 0)
        self.assertGreater(final_height, 0.45)
        self.assertLess(final_height, 0.7)

    def test_smoke_emitters_create_finite_projected_density(self):
        scene = Scene.from_dict(
            {
                "name": "smoke",
                "schema_version": 2,
                "objects": {
                    "smoke": {
                        "id": "smoke",
                        "kind": "fluid",
                        "phase": "smoke",
                        "size": [1.0, 1.0, 1.0],
                        "grid_resolution": [8, 8, 8],
                        "emitters": [
                            {
                                "position": [0.0, 0.0, -0.25],
                                "size": [0.25, 0.25, 0.25],
                                "start_time": 0.0,
                                "end_time": 0.2,
                                "density": 0.8,
                                "velocity": [0.0, 0.0, 0.2],
                            }
                        ],
                    }
                },
                "settings": {"duration": 0.1, "fps": 20, "substeps": 1},
            }
        )
        stats = SceneExecutorLocal().simulate(scene)["metrics"]["fluid"]["smoke"]
        self.assertTrue(stats["finite"])
        self.assertGreater(stats["density_mass"], 0.0)
        self.assertGreater(stats["occupied_cells"], 0)

    def test_apic_particle_pool_projects_and_remains_finite(self):
        scene = Scene.from_dict(
            {
                "name": "liquid",
                "schema_version": 2,
                "objects": {
                    "water": {
                        "id": "water",
                        "kind": "fluid",
                        "phase": "liquid",
                        "size": [0.4, 0.4, 0.4],
                        "grid_resolution": [8, 8, 8],
                        "particle_spacing": 0.1,
                    }
                },
                "settings": {"duration": 0.05, "fps": 20, "substeps": 1, "max_particles": 1000},
            }
        )
        stats = SceneExecutorLocal().simulate(scene)["metrics"]["fluid"]["water"]
        self.assertLessEqual(stats["particles"], 1000)
        self.assertGreaterEqual(stats["min_particles_per_cell"], 4)
        self.assertLessEqual(stats["max_particles_per_cell"], 8)
        self.assertAlmostEqual(stats["fluid_mass"], 64.0, places=4)
        self.assertTrue(stats["finite"])
        self.assertTrue(math.isfinite(stats["max_divergence"]))

    def test_apic_builds_rigid_boundary_and_returns_pressure_impulse(self):
        scene = Scene.from_dict(
            {
                "name": "coupled-liquid",
                "schema_version": 2,
                "objects": {
                    "block": {
                        "id": "block",
                        "kind": "rigid",
                        "shape": "box",
                        "size": [0.2, 0.2, 0.2],
                        "transform": {"position": [0.0, 0.0, 0.0]},
                    },
                    "water": {
                        "id": "water",
                        "kind": "fluid",
                        "phase": "liquid",
                        "size": [0.8, 0.8, 0.8],
                        "grid_resolution": [10, 10, 10],
                        "particle_spacing": 0.15,
                    },
                },
                "settings": {"duration": 0.1, "fps": 20, "substeps": 1, "max_particles": 1000},
                "render": {"ground": False},
            }
        )
        result = SceneExecutorLocal().simulate(scene)
        impulse = result["metrics"]["fluid"]["water"]["rigid_linear_impulse"]["0"]
        self.assertGreater(impulse[2], 0.0)
        self.assertTrue(all(math.isfinite(value) for value in impulse))

    def test_cancellation_leaves_resumable_incomplete_cache(self):
        scene = Scene.from_dict(
            {
                "name": "cancelled",
                "schema_version": 2,
                "settings": {"duration": 0.1, "fps": 20},
                "render": {"resolution": [16, 16], "fps": 10},
            }
        )
        cancellation = Event()
        cancellation.set()
        output_dir = self.root / "cancelled"
        result = SceneExecutorLocal().run(scene, output_dir=output_dir, cancel_event=cancellation)
        self.assertEqual(result["status"], "cancelled")
        manifest = json.loads((output_dir / "cache" / "manifest.json").read_text())
        self.assertFalse(manifest["complete"])

    def test_restart_marks_incomplete_simulation_interrupted(self):
        jobs = self.root / "scenes" / ".jobs"
        jobs.mkdir(exist_ok=True)
        job_id = "interrupted-job"
        (jobs / f"{job_id}.json").write_text(
            json.dumps(
                {
                    "job_id": job_id,
                    "scene": "demo",
                    "status": "running",
                    "stage": "simulation",
                    "output_dir": str(self.root / "missing-output"),
                }
            )
        )
        restored = SceneTools(SceneStore(self.root / "scenes"), SceneExecutorLocal())
        self.assertEqual(restored.get_job(job_id)["status"], "interrupted")

    def test_exported_program_matches_direct_execution(self):
        scene = Scene.from_dict(
            {
                "name": "equivalent",
                "schema_version": 2,
                "objects": {
                    "ball": {
                        "id": "ball",
                        "kind": "rigid",
                        "shape": "sphere",
                        "size": [0.4, 0.4, 0.4],
                        "transform": {"position": [0.0, 0.0, 1.0]},
                    }
                },
                "settings": {"duration": 0.1, "fps": 20, "substeps": 2},
            }
        )
        program = self.root / "equivalent.py"
        executor = SceneExecutorLocal()
        executor.compiler.export_program(scene, program)
        embedded = Scene.from_dict(runpy.run_path(program)["SCENE_IR"])
        direct = executor.simulate(scene)
        exported = SceneExecutorLocal().simulate(embedded)
        self.assertEqual(direct["scene_hash"], exported["scene_hash"])
        self.assertEqual(direct["metrics"]["first_contact_time"], exported["metrics"]["first_contact_time"])
        self.assertTrue(np.allclose(direct["trajectories"]["ball"], exported["trajectories"]["ball"]))

    def test_triangle_contact_covers_sphere_box_and_plane(self):
        triangle = np.array([[-1.0, -1.0, 0.0], [1.0, -1.0, 0.0], [0.0, 1.0, 0.0]])
        identity = np.array([0.0, 0.0, 0.0, 1.0])
        sphere, sphere_weights = SceneExecutorLocal._triangle_shape_correction(
            triangle, "sphere", np.array([0.0, 0.0, 0.1]), np.full(3, 0.4), identity, 0.01
        )
        box, box_weights = SceneExecutorLocal._triangle_shape_correction(
            triangle, "box", np.array([0.0, 0.0, 0.0]), np.full(3, 0.2), identity, 0.01
        )
        plane, plane_weights = SceneExecutorLocal._triangle_shape_correction(
            triangle - np.array([0.0, 0.0, 0.005]),
            "plane",
            np.zeros(3),
            np.ones(3),
            identity,
            0.01,
        )
        for name, correction, weights in (
            ("sphere", sphere, sphere_weights),
            ("box", box, box_weights),
            ("plane", plane, plane_weights),
        ):
            self.assertIsNotNone(correction, name)
            self.assertAlmostEqual(float(weights.sum()), 1.0)

    def test_triangle_contact_moves_dynamic_rigid_body(self):
        scene = Scene.from_dict(
            {
                "name": "cloth_dynamic_response",
                "objects": {
                    "cloth": {
                        "id": "cloth",
                        "kind": "cloth",
                        "size": [1.0, 1.0],
                        "resolution": [3, 3],
                        "thickness": 0.02,
                        "transform": {"position": [-0.5, -0.5, 0.0]},
                        "pinned": [
                            {"kind": "uv-corners", "corners": ["bottom-left", "bottom-right", "top-left", "top-right"]}
                        ],
                    },
                    "ball": {
                        "id": "ball",
                        "kind": "rigid",
                        "shape": "sphere",
                        "size": [0.2, 0.2, 0.2],
                        "physical_material": {"density": 10.0},
                        "transform": {"position": [0.0, 0.0, 0.05]},
                    },
                },
                "settings": {"duration": 0.05, "fps": 20, "substeps": 2, "gravity": [0.0, 0.0, 0.0]},
                "render": {"ground": False},
            }
        )
        result = SceneExecutorLocal().simulate(scene, capture_cache=True)
        self.assertGreater(result["state_frames"][-1]["body_q"][0, 2], 0.05)


if __name__ == "__main__":
    unittest.main()
