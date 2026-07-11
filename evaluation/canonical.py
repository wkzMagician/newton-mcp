# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Private canonical scenes used only for end-to-end evaluation."""

from __future__ import annotations

import math

from ca_framework.scene import Scene


def _quat_y(angle: float) -> list[float]:
    return [0.0, math.sin(angle * 0.5), 0.0, math.cos(angle * 0.5)]


def _quat_x(angle: float) -> list[float]:
    return [math.sin(angle * 0.5), 0.0, 0.0, math.cos(angle * 0.5)]


def canonical_scenes() -> dict[str, Scene]:
    """Return fresh instances of the ten canonical acceptance scenes."""
    scenes = {
        "01_rigid_contacts": {
            "objects": {
                "box": {
                    "id": "box",
                    "kind": "rigid",
                    "shape": "box",
                    "size": [0.6, 0.6, 0.6],
                    "transform": {"position": [-0.45, 0.0, 2.0]},
                },
                "sphere": {
                    "id": "sphere",
                    "kind": "rigid",
                    "shape": "sphere",
                    "size": [0.6, 0.6, 0.6],
                    "transform": {"position": [0.0, 0.0, 3.0]},
                },
                "capsule": {
                    "id": "capsule",
                    "kind": "rigid",
                    "shape": "capsule",
                    "size": [0.5, 0.5, 1.0],
                    "transform": {"position": [0.45, 0.0, 2.2]},
                },
            },
            "settings": {"duration": 2.0, "fps": 30, "substeps": 4},
            "render": {"camera": {"position": [2.8, -4.2, 2.8], "target": [0.0, 0.0, 1.2]}},
        },
        "02_domino_wave": {
            "objects": {
                **{
                    f"domino_{index}": {
                        "id": f"domino_{index}",
                        "kind": "rigid",
                        "shape": "box",
                        "size": [0.12, 0.5, 0.8],
                        "transform": {"position": [-1.0 + index * 0.45, 0.0, 0.4]},
                        **(
                            {"transform": {"position": [-1.0, 0.0, 0.4], "rotation": _quat_y(0.22)}}
                            if index == 0
                            else {}
                        ),
                    }
                    for index in range(7)
                }
            },
            "actions": {
                "push": {
                    "id": "push",
                    "kind": "impulse",
                    "object_id": "domino_0",
                    "impulse": [8.0, 0.0, 0.0],
                    "time": 0.0,
                    "point": [0.0, 0.0, 0.35],
                }
            },
            "settings": {"duration": 2.0, "fps": 40, "substeps": 8},
            "render": {"camera": {"position": [2.8, -4.0, 2.2], "target": [0.35, 0.0, 0.4]}},
        },
        "03_ramp_bounce": {
            "objects": {
                "ramp": {
                    "id": "ramp",
                    "kind": "rigid",
                    "motion": "static",
                    "shape": "box",
                    "size": [3.0, 1.0, 0.15],
                    "transform": {"position": [0.0, 0.0, 0.8], "rotation": _quat_y(0.25)},
                    "physical_material": {"friction_static": 0.1, "friction_dynamic": 0.08},
                },
                "ball": {
                    "id": "ball",
                    "kind": "rigid",
                    "shape": "sphere",
                    "size": [0.35, 0.35, 0.35],
                    "transform": {"position": [-1.0, 0.0, 1.35]},
                    "physical_material": {"restitution": 0.75, "friction_static": 0.08, "friction_dynamic": 0.05},
                },
                "stop": {
                    "id": "stop",
                    "kind": "rigid",
                    "motion": "static",
                    "shape": "box",
                    "size": [0.12, 1.0, 0.8],
                    "transform": {"position": [1.25, 0.0, 0.65]},
                },
            },
            "settings": {"duration": 2.5, "fps": 40, "substeps": 5},
            "render": {"camera": {"position": [2.8, -3.8, 2.3], "target": [0.1, 0.0, 0.75]}},
        },
        "04_cloth_ball": {
            "objects": {
                "cloth": {
                    "id": "cloth",
                    "kind": "cloth",
                    "size": [2.0, 2.0],
                    "resolution": [13, 13],
                    "surface_density": 5.0,
                    "thickness": 0.025,
                    "stretch_stiffness": 200.0,
                    "damping": 2.0,
                    "transform": {"position": [-1.0, -1.0, 1.2]},
                    "pinned": [
                        {"kind": "uv-corners", "corners": ["bottom-left", "bottom-right", "top-left", "top-right"]}
                    ],
                },
                "ball": {
                    "id": "ball",
                    "kind": "rigid",
                    "shape": "sphere",
                    "size": [0.5, 0.5, 0.5],
                    "transform": {"position": [0.0, 0.0, 2.2]},
                },
            },
            "settings": {"duration": 1.8, "fps": 30, "substeps": 12},
            "render": {"camera": {"position": [2.5, -3.5, 2.5], "target": [0.0, 0.0, 1.0]}},
        },
        "05_hanging_cloth": {
            "objects": {
                "cloth": {
                    "id": "cloth",
                    "kind": "cloth",
                    "size": [1.5, 2.0],
                    "resolution": [11, 15],
                    "surface_density": 5.0,
                    "thickness": 0.02,
                    "stretch_stiffness": 100.0,
                    "bend_stiffness": 5.0,
                    "damping": 2.0,
                    "transform": {"position": [-0.75, 0.0, 0.5], "rotation": _quat_x(math.pi * 0.5)},
                    "pinned": [{"kind": "edge", "edge": "top"}],
                }
            },
            "actions": {
                "kick": {
                    "id": "kick",
                    "kind": "force",
                    "object_id": "cloth",
                    "force": [0.0, 20.0, 0.0],
                    "start_time": 0.0,
                    "end_time": 0.15,
                }
            },
            "settings": {"duration": 3.0, "fps": 30, "substeps": 8},
            "render": {
                "ground": False,
                "camera": {
                    "position": [2.5, -3.8, 2.1],
                    "target": [0.0, 0.0, 1.5],
                    "up": [0.0, 0.0, 1.0],
                    "auto_frame": False,
                },
            },
        },
        "06_cloth_blocks": {
            "objects": {
                "cloth": {
                    "id": "cloth",
                    "kind": "cloth",
                    "size": [2.4, 1.6],
                    "resolution": [15, 11],
                    "surface_density": 5.0,
                    "thickness": 0.025,
                    "transform": {"position": [-1.2, -0.8, 1.8]},
                },
                "left": {
                    "id": "left",
                    "kind": "rigid",
                    "motion": "static",
                    "shape": "box",
                    "size": [0.7, 0.8, 0.8],
                    "transform": {"position": [-0.55, 0.0, 0.4]},
                },
                "right": {
                    "id": "right",
                    "kind": "rigid",
                    "motion": "static",
                    "shape": "box",
                    "size": [0.7, 0.8, 1.1],
                    "transform": {"position": [0.55, 0.0, 0.55]},
                },
            },
            "settings": {"duration": 2.0, "fps": 30, "substeps": 5},
            "render": {"camera": {"position": [2.8, -4.0, 2.5], "target": [0.0, 0.0, 0.8]}},
        },
        "07_smoke_partitions": {
            "objects": {
                "smoke": {
                    "id": "smoke",
                    "kind": "fluid",
                    "phase": "smoke",
                    "size": [2.0, 1.0, 2.0],
                    "grid_resolution": [20, 10, 20],
                    "transform": {"position": [0.0, 0.0, 1.0]},
                    "dissipation": 0.08,
                    "emitters": [
                        {
                            "position": [-0.75, 0.0, 0.2],
                            "size": [0.2, 0.3, 0.2],
                            "start_time": 0.0,
                            "end_time": 1.2,
                            "density": 1.0,
                            "velocity": [1.0, 0.0, 0.8],
                        }
                    ],
                },
                "wall_a": {
                    "id": "wall_a",
                    "kind": "rigid",
                    "motion": "static",
                    "shape": "box",
                    "size": [0.08, 1.0, 1.2],
                    "transform": {"position": [-0.25, 0.0, 0.6]},
                },
                "wall_b": {
                    "id": "wall_b",
                    "kind": "rigid",
                    "motion": "static",
                    "shape": "box",
                    "size": [0.08, 1.0, 1.2],
                    "transform": {"position": [0.35, 0.0, 1.4]},
                },
            },
            "settings": {"duration": 2.5, "fps": 20, "substeps": 1},
            "render": {
                "ground": False,
                "camera": {
                    "position": [2.2, -3.3, 2.2],
                    "target": [0.0, 0.0, 1.0],
                    "up": [0.0, 0.0, 1.0],
                    "auto_frame": False,
                },
            },
        },
        "08_liquid_pour": {
            "objects": {
                "left": {
                    "id": "left",
                    "kind": "container",
                    "motion": "static",
                    "inner_size": [0.9, 0.8, 0.9],
                    "wall_thickness": 0.06,
                    "transform": {"position": [-0.5, 0.0, 0.8], "rotation": _quat_y(0.7)},
                },
                "right": {
                    "id": "right",
                    "kind": "container",
                    "motion": "static",
                    "inner_size": [1.0, 0.9, 0.45],
                    "wall_thickness": 0.06,
                    "transform": {"position": [0.65, 0.0, 0.0]},
                },
                "water": {
                    "id": "water",
                    "kind": "fluid",
                    "phase": "liquid",
                    "size": [0.2, 0.4, 0.2],
                    "grid_resolution": [24, 12, 16],
                    "particle_spacing": 0.12,
                    "transform": {"position": [0.2, 0.0, 1.15]},
                    "emitters": [
                        {
                            "position": [0.2, 0.0, 1.15],
                            "size": [0.2, 0.4, 0.2],
                            "start_time": 0.0,
                            "end_time": 0.6,
                            "velocity": [0.8, 0.0, -0.3],
                        }
                    ],
                },
            },
            "settings": {"duration": 3.0, "fps": 20, "substeps": 2, "max_particles": 10000},
            "render": {
                "ground": False,
                "camera": {
                    "position": [2.1, -3.2, 2.2],
                    "target": [0.1, 0.0, 0.65],
                    "up": [0.0, 0.0, 1.0],
                    "auto_frame": False,
                },
            },
        },
        "09_liquid_splash": {
            "objects": {
                "pool": {
                    "id": "pool",
                    "kind": "container",
                    "motion": "static",
                    "inner_size": [1.6, 1.0, 1.0],
                    "wall_thickness": 0.06,
                },
                "water": {
                    "id": "water",
                    "kind": "fluid",
                    "phase": "liquid",
                    "size": [1.35, 0.75, 0.45],
                    "grid_resolution": [20, 12, 14],
                    "particle_spacing": 0.13,
                    "transform": {"position": [0.0, 0.0, 0.3]},
                },
                "ball": {
                    "id": "ball",
                    "kind": "rigid",
                    "shape": "sphere",
                    "size": [0.28, 0.28, 0.28],
                    "transform": {"position": [0.0, 0.0, 1.5]},
                    "physical_material": {"density": 7800.0},
                },
            },
            "settings": {"duration": 1.5, "fps": 25, "substeps": 2, "max_particles": 12000},
            "render": {
                "ground": False,
                "camera": {
                    "position": [2.2, -3.4, 2.0],
                    "target": [0.0, 0.0, 0.55],
                    "up": [0.0, 0.0, 1.0],
                    "auto_frame": False,
                },
            },
        },
        "10_floating_blocks": {
            "objects": {
                "pool": {
                    "id": "pool",
                    "kind": "container",
                    "motion": "static",
                    "inner_size": [2.0, 1.2, 1.0],
                    "wall_thickness": 0.06,
                },
                "water": {
                    "id": "water",
                    "kind": "fluid",
                    "phase": "liquid",
                    "size": [1.75, 0.95, 0.6],
                    "grid_resolution": [22, 14, 14],
                    "particle_spacing": 0.14,
                    "transform": {"position": [0.0, 0.0, 0.38]},
                },
                **{
                    f"block_{index}": {
                        "id": f"block_{index}",
                        "kind": "rigid",
                        "shape": "box",
                        "size": [0.35, 0.28, 0.22],
                        "transform": {"position": [-0.45 + index * 0.45, 0.0, 0.75]},
                        "physical_material": {"density": 600.0},
                    }
                    for index in range(3)
                },
            },
            "settings": {"duration": 2.0, "fps": 20, "substeps": 2, "max_particles": 15000},
            "render": {
                "ground": False,
                "camera": {
                    "position": [2.5, -3.8, 2.1],
                    "target": [0.0, 0.0, 0.55],
                    "up": [0.0, 0.0, 1.0],
                    "auto_frame": False,
                },
            },
        },
    }
    return {name: Scene.from_dict({"name": name, "schema_version": 2, **payload}) for name, payload in scenes.items()}


__all__ = ["canonical_scenes"]
