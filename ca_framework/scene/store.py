# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Persistent scene storage."""

from __future__ import annotations

import json
from pathlib import Path

from .model import Scene


class SceneStore:
    """Store scenes as human-readable JSON files in one workspace."""

    def __init__(self, root: str | Path):
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def create(self, name: str, *, overwrite: bool = False) -> Scene:
        """Create and save an empty scene."""
        scene = Scene(name=_validate_name(name))
        path = self.path_for(scene.name)
        if path.exists() and not overwrite:
            raise FileExistsError(f"Scene already exists: {scene.name}")
        self.save(scene)
        return scene

    def load(self, name: str) -> Scene:
        """Load a scene by name."""
        with self.path_for(name).open(encoding="utf-8") as stream:
            return Scene.from_dict(json.load(stream))

    def save(self, scene: Scene) -> Path:
        """Atomically save a scene and return its path."""
        path = self.path_for(scene.name)
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(scene.to_dict(), indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)
        return path

    def list(self) -> list[str]:
        """Return scene names in lexical order."""
        return [path.stem for path in sorted(self.root.glob("*.json"))]

    def delete(self, name: str) -> None:
        """Delete a scene."""
        self.path_for(name).unlink()

    def path_for(self, name: str) -> Path:
        """Return the path for a validated scene name."""
        return self.root / f"{_validate_name(name)}.json"


def _validate_name(name: str) -> str:
    if not name or any(
        character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for character in name
    ):
        raise ValueError("Scene names may contain only letters, digits, '-' and '_'")
    return name
