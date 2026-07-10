# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Deterministic scene compilation and reproducible program export."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .model import ObjectCloth, ObjectContainer, Scene, VertexSelector
from .validation import select_pipeline, validate_scene


@dataclass(slots=True)
class CompiledScene:
    """Validated execution description consumed by local runners."""

    scene: Scene
    pipeline: list[str]
    container_colliders: dict[str, list[dict[str, Any]]]
    cloth_particle_indices: dict[str, list[int]]
    pinned_particles: dict[str, list[int]]


class SceneCompilerNewton:
    """Compile scene IR into a deterministic Newton execution description."""

    def compile(self, scene: Scene) -> CompiledScene:
        """Validate and compile a scene, raising with structured diagnostics."""
        report = validate_scene(scene)
        if not report["valid"]:
            raise ValueError(json.dumps({"error": "scene_validation_failed", **report}, sort_keys=True))
        containers = {
            item.id: self._container_shapes(item)
            for item in scene.objects.values()
            if isinstance(item, ObjectContainer)
        }
        cloth_indices: dict[str, list[int]] = {}
        pinned: dict[str, list[int]] = {}
        particle_start = 0
        for item in scene.objects.values():
            if not isinstance(item, ObjectCloth):
                continue
            count = item.resolution[0] * item.resolution[1]
            cloth_indices[item.id] = list(range(particle_start, particle_start + count))
            local_pins = {
                index for selector in item.pinned for index in self._selector_indices(selector, item.resolution)
            }
            pinned[item.id] = [particle_start + index for index in sorted(local_pins)]
            particle_start += count
        return CompiledScene(
            scene=scene,
            pipeline=select_pipeline(scene),
            container_colliders=containers,
            cloth_particle_indices=cloth_indices,
            pinned_particles=pinned,
        )

    def export_program(self, scene: Scene, output: str | Path) -> Path:
        """Write a standalone Python program embedding the exact scene IR."""
        self.compile(scene)
        output = Path(output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(scene.to_dict(), sort_keys=True, indent=2)
        program = f'''# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Generated reproducible Newton scene program."""

import json

from ca_framework import SceneExecutorLocal
from ca_framework.scene import Scene

SCENE_IR = json.loads(r\'''{payload}\''')

if __name__ == "__main__":
    result = SceneExecutorLocal().simulate(Scene.from_dict(SCENE_IR))
    print(json.dumps(result, sort_keys=True))
'''
        output.write_text(program, encoding="utf-8")
        return output

    @staticmethod
    def _container_shapes(container: ObjectContainer) -> list[dict[str, Any]]:
        x, y, z = container.inner_size
        t = container.wall_thickness
        local = [
            {"kind": "box", "size": (x + 2 * t, y + 2 * t, t), "position": (0.0, 0.0, -t / 2)},
            {"kind": "box", "size": (t, y + 2 * t, z), "position": (-(x + t) / 2, 0.0, z / 2)},
            {"kind": "box", "size": (t, y + 2 * t, z), "position": ((x + t) / 2, 0.0, z / 2)},
            {"kind": "box", "size": (x, t, z), "position": (0.0, -(y + t) / 2, z / 2)},
            {"kind": "box", "size": (x, t, z), "position": (0.0, (y + t) / 2, z / 2)},
        ]
        result = []
        for shape in local:
            scaled_position = tuple(shape["position"][axis] * container.transform.scale[axis] for axis in range(3))
            rotated_position = SceneCompilerNewton._rotate_vector(scaled_position, container.transform.rotation)
            result.append(
                {
                    **shape,
                    "size": tuple(shape["size"][axis] * container.transform.scale[axis] for axis in range(3)),
                    "position": tuple(container.transform.position[axis] + rotated_position[axis] for axis in range(3)),
                    "rotation": container.transform.rotation,
                }
            )
        return result

    @staticmethod
    def _selector_indices(selector: VertexSelector, resolution: tuple[int, int]) -> list[int]:
        """Resolve a selector to stable row-major Newton particle indices."""
        width, height = resolution
        if selector.kind == "indices":
            return list(selector.indices)
        if selector.kind == "edge":
            if selector.edge == "bottom":
                return list(range(width))
            if selector.edge == "top":
                return list(range((height - 1) * width, height * width))
            if selector.edge == "left":
                return [row * width for row in range(height)]
            return [row * width + width - 1 for row in range(height)]
        corners = {
            "bottom-left": 0,
            "bottom-right": width - 1,
            "top-left": (height - 1) * width,
            "top-right": height * width - 1,
        }
        return [corners[name] for name in selector.corners]

    @staticmethod
    def _rotate_vector(vector: tuple[float, float, float], quaternion: tuple[float, float, float, float]):
        """Rotate a vector by an ``(x, y, z, w)`` unit quaternion."""
        x, y, z, w = quaternion
        vx, vy, vz = vector
        tx, ty, tz = 2.0 * (y * vz - z * vy), 2.0 * (z * vx - x * vz), 2.0 * (x * vy - y * vx)
        return (
            vx + w * tx + y * tz - z * ty,
            vy + w * ty + z * tx - x * tz,
            vz + w * tz + x * ty - y * tx,
        )
