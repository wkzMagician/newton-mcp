# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import unittest
from unittest.mock import Mock, call

import numpy as np

from newton.viewer import RendererFluidScreenSpace


class TestRendererFluidScreenSpace(unittest.TestCase):
    def test_depth_smoothing_repeats_separable_bilateral_filter(self):
        renderer = RendererFluidScreenSpace.__new__(RendererFluidScreenSpace)
        renderer.depth_smoothing_iterations = 8
        renderer._depth_tex = "depth"
        renderer._depth_ping_tex = "depth_ping"
        renderer._blur_depth = Mock()
        texel = np.asarray([1.0 / 1280.0, 1.0 / 720.0], dtype=np.float32)

        renderer._smooth_depth(texel)

        expected_iteration = [
            call("depth", "depth_ping", texel),
            call("depth_ping", "depth", texel, horizontal=False),
        ]
        self.assertEqual(renderer._blur_depth.call_args_list, expected_iteration * 8)


if __name__ == "__main__":
    unittest.main()
