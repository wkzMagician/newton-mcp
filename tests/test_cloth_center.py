# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for the public center-defined cloth transform contract."""

from __future__ import annotations

import math
import unittest

from ca_framework.scene.compiler import SceneCompilerNewton
from ca_framework.scene.model import ObjectCloth, Scene, Transform


class TestClothCenterContract(unittest.TestCase):
    def test_newton_grid_origin_is_offset_from_the_public_sheet_center(self):
        cloth = ObjectCloth(
            id="sheet",
            size=(2.0, 4.0),
            transform=Transform(
                position=(3.0, 4.0, 5.0),
                rotation=(0.0, 0.0, math.sqrt(0.5), math.sqrt(0.5)),
                scale=(2.0, 0.5, 1.0),
            ),
        )

        for actual, expected in zip(SceneCompilerNewton._cloth_grid_origin(cloth), (4.0, 2.0, 5.0), strict=True):
            self.assertAlmostEqual(actual, expected)

    def test_compiled_vertices_are_centered_at_the_public_position(self):
        cloth = ObjectCloth(
            id="sheet",
            size=(2.0, 4.0),
            resolution=(3, 5),
            transform=Transform(position=(3.0, 4.0, 5.0)),
        )
        compiled = SceneCompilerNewton().compile(Scene(name="centered-cloth", objects={"sheet": cloth}))
        center = compiled.initial_particle_q[compiled.cloth_particle_indices["sheet"]].mean(axis=0)

        for actual, expected in zip(center, cloth.transform.position, strict=True):
            self.assertAlmostEqual(float(actual), expected, places=6)


if __name__ == "__main__":
    unittest.main()
