# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest
from functools import lru_cache
from itertools import pairwise

import numpy as np

from ca_framework.scene import SceneExecutorLocal, validate_scene
from evaluation.canonical import canonical_scenes


@lru_cache(maxsize=10)
def _result(name: str):
    return SceneExecutorLocal().simulate(canonical_scenes()[name], capture_cache=True)


class TestCanonicalScenes(unittest.TestCase):
    def test_all_canonical_scenes_validate(self):
        scenes = canonical_scenes()
        self.assertEqual(len(scenes), 10)
        self.assertTrue(all(validate_scene(scene)["valid"] for scene in scenes.values()))

    def test_all_canonical_scenes_are_physically_valid(self):
        for name in canonical_scenes():
            with self.subTest(name=name):
                result = _result(name)
                self.assertEqual(result["status"], "completed", result["diagnostics"])
                self.assertTrue(result["metrics"]["physics"]["valid"])
                for fluid in result["metrics"]["fluid"].values():
                    self.assertLess(fluid["peak_divergence"], 100.0)

    def test_01_rigid_objects_ground_and_collide(self):
        result = _result("01_rigid_contacts")
        self.assertGreaterEqual(len(result["metrics"]["contact_pairs"]), 1)
        self.assertTrue(all(samples[-1][2] > 0.2 for samples in result["trajectories"].values()))
        self.assertLess(result["metrics"]["max_penetration"], 0.02)

    def test_02_domino_contacts_propagate(self):
        times = _result("02_domino_wave")["metrics"]["first_contact_pair_times"]
        propagation = [times[f"domino_{index}|domino_{index + 1}"] for index in range(6)]
        self.assertTrue(all(left < right for left, right in pairwise(propagation)))

    def test_03_ball_descends_reverses_and_stops(self):
        frames = _result("03_ramp_bounce")["state_frames"]
        positions = np.asarray([frame["body_q"][0, 0] for frame in frames])
        velocities = np.asarray([frame["body_qd"][0, 0] for frame in frames])
        self.assertGreater(positions.max(), positions[0] + 1.0)
        self.assertGreater(velocities.max(), 0.5)
        self.assertLess(velocities.min(), -0.1)
        self.assertLess(abs(velocities[-1]), 0.05)

    def test_04_cloth_corners_hold_and_center_sags(self):
        scene = canonical_scenes()["04_cloth_ball"]
        result = _result("04_cloth_ball")
        cloth = result["state_frames"][-1]["cloth_q"]
        width, height = scene.objects["cloth"].resolution
        corner_indices = [0, width - 1, (height - 1) * width, height * width - 1]
        center_index = (height // 2) * width + width // 2
        corners = cloth[corner_indices, 2]
        self.assertTrue(np.allclose(corners, 1.2, atol=1.0e-5))
        center_minimum = min(frame["cloth_q"][center_index, 2] for frame in result["state_frames"])
        self.assertLess(center_minimum, corners.mean() - 0.15)
        self.assertGreater(result["metrics"]["soft_contacts"], 0)
        self.assertIn(["ball", "cloth"], result["metrics"]["soft_contact_pairs"])
        self.assertLess(result["metrics"]["max_soft_penetration"], 0.1)
        quality = result["metrics"]["cloth"]["cloth"]
        self.assertLessEqual(quality["max_edge_length_ratio"], 1.151)
        self.assertEqual(quality["flipped_triangle_count"], 0)

    def test_05_hanging_cloth_descends_and_swings(self):
        frames = _result("05_hanging_cloth")["state_frames"]
        lower_z = np.asarray([frame["cloth_q"][:11, 2].mean() for frame in frames])
        lower_y = np.asarray([frame["cloth_q"][:11, 1].mean() for frame in frames])
        top = frames[-1]["cloth_q"][-11:]
        self.assertTrue(np.allclose(top[:, 2], 2.5, atol=1.0e-5))
        self.assertLess(lower_z[-1], lower_z[0])
        self.assertGreater(np.ptp(lower_y), 0.02)
        self.assertLess(np.ptp(lower_y[len(lower_y) // 2 :]), np.ptp(lower_y[: len(lower_y) // 2]))

    def test_06_cloth_contacts_blocks_and_folds(self):
        result = _result("06_cloth_blocks")
        cloth = result["state_frames"][-1]["cloth_q"]
        self.assertGreater(result["metrics"]["soft_contacts"], 0)
        self.assertEqual(np.linalg.matrix_rank(cloth - cloth.mean(axis=0)), 3)
        self.assertGreater(float(cloth[:, 2].std()), 0.1)
        quality = result["metrics"]["cloth"]["cloth"]
        self.assertLessEqual(result["metrics"]["max_soft_penetration"], 0.025)
        self.assertLessEqual(quality["max_edge_length_ratio"], 1.201)
        self.assertLess(quality["max_speed"], 0.2)
        self.assertEqual(quality["flipped_triangle_count"], 0)
        self.assertEqual(
            result["metrics"]["soft_contact_pairs"],
            [["cloth", "left"], ["cloth", "right"]],
        )
        first_contact = result["metrics"]["first_contact_time"]
        for frame_index, frame in enumerate(result["state_frames"]):
            time_value = (frame_index + 1) / canonical_scenes()["06_cloth_blocks"].settings.fps
            if time_value >= first_contact:
                break
            speed = float(np.linalg.norm(frame["cloth_qd"], axis=1).max())
            self.assertLessEqual(speed, 9.81 * time_value + 0.3)

    def test_07_smoke_is_finite_expands_and_crosses_partitions(self):
        result = _result("07_smoke_partitions")
        densities = [frame["smoke_density"] for frame in result["state_frames"]]
        occupied_start = np.count_nonzero(densities[0] > 1.0e-5)
        occupied_end = np.count_nonzero(densities[-1] > 1.0e-5)
        occupied_x = np.flatnonzero(densities[-1].sum(axis=(1, 2)) > 1.0e-3)
        self.assertTrue(result["metrics"]["fluid"]["smoke"]["finite"])
        self.assertGreater(occupied_end, occupied_start)
        self.assertGreater(occupied_x.max() - occupied_x.min(), 8)

    def test_08_liquid_transfers_to_right_container(self):
        frame = _result("08_liquid_pour")["state_frames"][-1]
        particles = frame["water_particles"]
        masses = frame["water_masses"]
        right = particles[:, 0] > 0.15
        inside = right & (particles[:, 0] < 1.15) & (np.abs(particles[:, 1]) < 0.45)
        self.assertGreater(masses[right].sum() / masses.sum(), 0.5)
        self.assertGreater(masses[inside].sum() / masses.sum(), 0.5)

    def test_09_ball_sinks_and_creates_splash(self):
        scene = canonical_scenes()["09_liquid_splash"]
        result = _result("09_liquid_splash")
        initial_surface = scene.objects["water"].transform.position[2] + scene.objects["water"].size[2] * 0.5
        splash_height = max(frame["water_particles"][:, 2].max() for frame in result["state_frames"])
        self.assertLess(result["trajectories"]["ball"][-1][2], 0.2)
        self.assertGreater(splash_height, initial_surface + 0.05)

    def test_10_blocks_float_wobble_and_collide(self):
        result = _result("10_floating_blocks")
        frames = result["state_frames"]
        surface = frames[-1]["water_particles"][:, 2].max()
        block_heights = np.asarray([result["trajectories"][f"block_{index}"][-1][2] for index in range(3)])
        rotations = np.asarray([frame["body_q"][:, 3:6] for frame in frames])
        self.assertTrue(np.all(np.abs(block_heights - surface) < 0.35))
        self.assertGreater(float(np.max(np.linalg.norm(rotations, axis=2))), 0.05)
        self.assertIn(["block_0", "block_1"], result["metrics"]["contact_pairs"])
        self.assertGreater(result["metrics"]["first_active_contact_pair_times"]["block_0|block_1"], 0.0)


if __name__ == "__main__":
    unittest.main()
