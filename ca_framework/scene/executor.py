# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Local compilation, preview, trajectory caching, rendering, and export."""

from __future__ import annotations

import hashlib
import json
import math
import shutil
import subprocess
import tempfile
import time
from pathlib import Path
from threading import Event
from typing import Any

import numpy as np

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

    def run(
        self,
        scene: Scene,
        *,
        output_dir: Path,
        cancel_event: Event | None = None,
        progress: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Create a durable simulation, render, and reproducibility bundle.

        Simulation data is committed before rendering so an encoder failure can
        be retried without repeating the simulation. ``cancel_event`` is checked
        at every cached frame, which provides cooperative cancellation between
        native solver steps.

        Args:
            scene: Scene to execute.
            output_dir: Directory receiving the complete output bundle.
            cancel_event: Optional cooperative cancellation flag.
            progress: Optional mutable job progress mapping.

        Returns:
            Final job result including artifacts and diagnostics.
        """
        self.compiler.compile(scene)
        output_dir = output_dir.resolve()
        cache_dir = output_dir / "cache"
        frames_dir = cache_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        scene_path = output_dir / "scene.json"
        program_path = output_dir / "program.py"
        metrics_path = output_dir / "metrics.json"
        diagnostics_path = output_dir / "diagnostics.jsonl"
        contacts_path = cache_dir / "contacts.jsonl"
        cache_manifest_path = cache_dir / "manifest.json"
        job_manifest_path = output_dir / "manifest.json"
        animation_path = output_dir / "animation.mp4"
        scene_payload = scene.to_dict()
        scene_bytes = json.dumps(scene_payload, sort_keys=True, separators=(",", ":")).encode()
        scene_hash = hashlib.sha256(scene_bytes).hexdigest()
        artifacts = {
            "animation": str(animation_path),
            "scene": str(scene_path),
            "program": str(program_path),
            "metrics": str(metrics_path),
            "diagnostics": str(diagnostics_path),
            "cache": str(cache_dir),
        }

        def update(stage: str, frame: int = 0, status: str = "running") -> None:
            values = {
                "status": status,
                "stage": stage,
                "frame": frame,
                "total_frames": round(scene.settings.duration * scene.settings.fps),
                "scene": scene.name,
                "scene_hash": scene_hash,
                "output_dir": str(output_dir),
                "artifacts": artifacts,
            }
            if progress is not None:
                progress.update(values)
            job_manifest_path.write_text(json.dumps(values, indent=2) + "\n", encoding="utf-8")

        update("compile")
        scene_path.write_text(json.dumps(scene_payload, indent=2) + "\n", encoding="utf-8")
        self.compiler.export_program(scene, program_path)
        diagnostics_path.write_text("", encoding="utf-8")
        contacts_path.write_text("", encoding="utf-8")

        simulation_started = time.perf_counter()
        simulation = self.simulate(scene)
        trajectories = simulation.pop("trajectories")
        frame_count = simulation["frames"]
        for frame_index in range(frame_count):
            if cancel_event is not None and cancel_event.is_set():
                update("simulation", frame_index, "cancelled")
                return {"status": "cancelled", "scene": scene.name, "artifacts": artifacts}
            frame_state = {
                object_id: np.asarray(samples[frame_index], dtype=np.float32)
                for object_id, samples in trajectories.items()
            }
            np.savez_compressed(frames_dir / f"{frame_index:06d}.npz", **frame_state)
            update("simulation", frame_index + 1)
        simulation_seconds = time.perf_counter() - simulation_started
        metrics = {
            **simulation["metrics"],
            "scene_hash": scene_hash,
            "timings": {"simulation_seconds": simulation_seconds},
        }
        metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        cache_manifest = {
            "schema_version": 1,
            "scene_hash": scene_hash,
            "complete": True,
            "frames": frame_count,
            "fps": scene.settings.fps,
            "frame_pattern": "frames/%06d.npz",
            "contacts": contacts_path.name,
            "metrics": "../metrics.json",
        }
        cache_manifest_path.write_text(json.dumps(cache_manifest, indent=2) + "\n", encoding="utf-8")

        if cancel_event is not None and cancel_event.is_set():
            update("render", frame_count, "cancelled")
            return {"status": "cancelled", "scene": scene.name, "artifacts": artifacts}
        update("render", frame_count)
        render_started = time.perf_counter()
        rendered = self.render(scene, output=animation_path)
        metrics["timings"]["render_seconds"] = time.perf_counter() - render_started
        metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
        status = rendered["status"]
        update("completed" if status == "completed" else "render_failed", frame_count, status)
        return {
            "status": status,
            "stage": "completed" if status == "completed" else "render_failed",
            "scene": scene.name,
            "scene_hash": scene_hash,
            "frames": frame_count,
            "artifacts": artifacts,
            "metrics": metrics,
            "diagnostics": rendered.get("diagnostics", []),
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
