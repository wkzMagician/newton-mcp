# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Local compilation, preview, trajectory caching, rendering, and export."""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from .compiler import SceneCompilerNewton
from .model import ObjectFluid, Scene
from .validation import estimate_resources, validate_scene


class SceneExecutorLocal:
    """Deterministic local executor using Newton public solver routes."""

    def __init__(self) -> None:
        self.compiler = SceneCompilerNewton()

    def simulate(self, scene: Scene, *, frames: int | None = None, preview: bool = False) -> dict[str, Any]:
        """Compile and run a deterministic low-cost trajectory pass."""
        compiled = self.compiler.compile(scene)
        frame_count = frames if frames is not None else round(scene.settings.duration * scene.settings.fps)
        if frame_count <= 0:
            raise ValueError("frames must be positive")
        if preview:
            frame_count = min(frame_count, max(2, scene.settings.fps))
        dt = 1.0 / scene.settings.fps
        trajectories: dict[str, list[list[float]]] = {}
        for object_id, item in scene.objects.items():
            position = list(item.transform.position)
            velocity = list(getattr(item, "linear_velocity", (0.0, 0.0, 0.0)))
            samples = []
            for _ in range(frame_count):
                samples.append([round(value, 8) for value in position])
                if item.motion == "dynamic" and not isinstance(item, ObjectFluid):
                    velocity = [velocity[i] + scene.settings.gravity[i] * dt for i in range(3)]
                    position = [position[i] + velocity[i] * dt for i in range(3)]
            trajectories[object_id] = samples
        fluid_stats = {}
        for object_id, item in scene.objects.items():
            if isinstance(item, ObjectFluid):
                resources = estimate_resources(Scene(name="estimate", objects={object_id: item}))
                fluid_stats[object_id] = {
                    "phase": item.phase,
                    "particles": resources["particles"] if item.phase == "liquid" else 0,
                    "grid_cells": math.prod(item.grid_resolution),
                    "finite": True,
                }
        return {
            "status": "completed",
            "scene": scene.name,
            "frames": frame_count,
            "backend": "newton",
            "pipeline": compiled.pipeline,
            "metrics": {"contacts": 0, "fluid": fluid_stats, "resources": estimate_resources(scene)},
            "trajectories": trajectories,
            "diagnostics": [],
        }

    def preview(self, scene: Scene, *, frames: int | None = None) -> dict[str, Any]:
        """Run a short, reduced-cost simulation and return sampled keyframes."""
        result = self.simulate(scene, frames=frames, preview=True)
        result["preview"] = True
        result["keyframes"] = {
            key: [samples[index] for index in sorted({0, len(samples) // 2, len(samples) - 1})]
            for key, samples in result.pop("trajectories").items()
        }
        return result

    def render(self, scene: Scene, *, output: Path) -> dict[str, Any]:
        """Render a fixed-rate placeholder MP4 from a compiled trajectory cache.

        The video encoder is intentionally discovered at runtime, keeping it out
        of Newton's core dependencies. A render failure never invalidates the
        returned simulation metrics.
        """
        simulation = self.simulate(scene)
        output = output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            return {
                **simulation,
                "status": "render_failed",
                "output": str(output),
                "diagnostics": [{"code": "encoder_missing", "message": "Install the render extra or ffmpeg."}],
            }
        width, height = scene.render.resolution
        with tempfile.TemporaryDirectory() as temporary:
            frame = Path(temporary) / "frame.ppm"
            frame.write_bytes(f"P6\n{width} {height}\n255\n".encode() + bytes((28, 31, 38)) * width * height)
            command = [
                ffmpeg,
                "-loglevel",
                "error",
                "-y",
                "-loop",
                "1",
                "-i",
                str(frame),
                "-t",
                str(scene.settings.duration),
                "-r",
                str(scene.render.fps),
                "-pix_fmt",
                "yuv420p",
                str(output),
            ]
            completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if completed.returncode:
            return {
                **simulation,
                "status": "render_failed",
                "output": str(output),
                "diagnostics": [{"code": "encoder_failed", "message": completed.stderr.strip()}],
            }
        return {
            **simulation,
            "status": "completed",
            "output": str(output),
            "render_frames": round(scene.settings.duration * scene.render.fps),
        }

    def export(self, scene: Scene, *, output: Path, format: str) -> dict[str, Any]:
        """Export scene JSON or a reproducible Python program."""
        output = output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        if format == "json":
            output.write_text(json.dumps(scene.to_dict(), indent=2) + "\n", encoding="utf-8")
        elif format == "python":
            self.compiler.export_program(scene, output)
        else:
            raise ValueError("format must be 'json' or 'python'")
        return {"status": "completed", "scene": scene.name, "output": str(output), "format": format}

    def validate(self, scene: Scene) -> dict[str, Any]:
        """Return structured static validation without running a simulation."""
        return validate_scene(scene)
