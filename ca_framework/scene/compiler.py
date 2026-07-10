# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Deterministic scene compilation and reproducible program export."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .model import ObjectContainer, Scene
from .validation import select_pipeline, validate_scene


@dataclass(slots=True)
class CompiledScene:
    """Validated execution description consumed by local runners."""

    scene: Scene
    pipeline: list[str]
    container_colliders: dict[str, list[dict[str, Any]]]


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
        return CompiledScene(scene=scene, pipeline=select_pipeline(scene), container_colliders=containers)

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
        return [
            {"kind": "box", "size": (x + 2 * t, y + 2 * t, t), "position": (0.0, 0.0, -t / 2)},
            {"kind": "box", "size": (t, y + 2 * t, z), "position": (-(x + t) / 2, 0.0, z / 2)},
            {"kind": "box", "size": (t, y + 2 * t, z), "position": ((x + t) / 2, 0.0, z / 2)},
            {"kind": "box", "size": (x, t, z), "position": (0.0, -(y + t) / 2, z / 2)},
            {"kind": "box", "size": (x, t, z), "position": (0.0, (y + t) / 2, z / 2)},
        ]
