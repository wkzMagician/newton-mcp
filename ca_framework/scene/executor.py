# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Basic local executor and export boundary for Newton integration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .model import Scene


class SceneExecutorLocal:
    """Local executor with JSON export and explicit backend capability errors.

    This initial executor deliberately keeps Newton compilation behind one boundary.
    A Newton runtime can replace or extend this class without changing MCP tools.
    """

    def simulate(self, scene: Scene, *, frames: int | None = None) -> dict[str, Any]:
        """Validate a simulation request pending Newton runtime compilation."""
        frame_count = frames if frames is not None else round(scene.settings.duration * scene.settings.fps)
        if frame_count <= 0:
            raise ValueError("frames must be positive")
        return {
            "status": "ready",
            "scene": scene.name,
            "frames": frame_count,
            "backend": "newton",
            "message": "Scene validated; Newton runtime compilation is the next implementation stage.",
        }

    def render(self, scene: Scene, *, output: Path) -> dict[str, Any]:
        """Report the intended render output pending Newton viewer integration."""
        return {
            "status": "ready",
            "scene": scene.name,
            "output": str(output.resolve()),
            "backend": "newton-viewer",
            "message": "Scene validated; Newton viewer integration is the next implementation stage.",
        }

    def export(self, scene: Scene, *, output: Path, format: str) -> dict[str, Any]:
        """Export the backend-neutral scene as JSON."""
        if format != "json":
            raise ValueError("The initial executor supports only JSON export")
        output = output.resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(scene.to_dict(), indent=2) + "\n", encoding="utf-8")
        return {"status": "completed", "scene": scene.name, "output": str(output), "format": format}
