# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""FastMCP adapter for the animation scene tools."""

from __future__ import annotations

import argparse
import os
from typing import Any, Literal

from mcp.server.fastmcp import FastMCP

from ca_framework.mcp import SceneTools
from ca_framework.scene import SceneExecutorLocal, SceneStore

mcp = FastMCP(
    "Newton Scene Editor",
    instructions="Create and edit backend-neutral animation scenes, then simulate, render, or export them with Newton.",
    json_response=True,
)


def _tools() -> SceneTools:
    workspace = os.environ.get("CA_SCENE_WORKSPACE", ".ca-scenes")
    return SceneTools(SceneStore(workspace), SceneExecutorLocal())


@mcp.tool()
def create_scene(name: str, overwrite: bool = False) -> dict[str, Any]:
    """Create an empty animation scene."""
    return _tools().create_scene(name, overwrite)


@mcp.tool()
def list_scenes() -> list[str]:
    """List all animation scenes in the workspace."""
    return _tools().list_scenes()


@mcp.tool()
def get_scene(name: str) -> dict[str, Any]:
    """Read a complete animation scene."""
    return _tools().get_scene(name)


@mcp.tool()
def add_object(scene_name: str, spec: dict[str, Any]) -> dict[str, Any]:
    """Add an object; spec.kind is rigid, cloth, or fluid."""
    return _tools().add_object(scene_name, spec)


@mcp.tool()
def add_constraint(scene_name: str, spec: dict[str, Any]) -> dict[str, Any]:
    """Add a constraint; spec.kind is fixed-point or distance."""
    return _tools().add_constraint(scene_name, spec)


@mcp.tool()
def add_field(scene_name: str, spec: dict[str, Any]) -> dict[str, Any]:
    """Add a force or acceleration field; spec.kind is uniform or radial."""
    return _tools().add_field(scene_name, spec)


@mcp.tool()
def update_item(
    scene_name: str,
    collection: Literal["objects", "constraints", "fields"],
    item_id: str,
    patch: dict[str, Any],
) -> dict[str, Any]:
    """Patch one scene item without changing its id or kind."""
    return _tools().update_item(scene_name, collection, item_id, patch)


@mcp.tool()
def remove_item(scene_name: str, collection: Literal["objects", "constraints", "fields"], item_id: str) -> None:
    """Remove one scene item when it has no remaining references."""
    _tools().remove_item(scene_name, collection, item_id)


@mcp.tool()
def simulate_scene(scene_name: str, frames: int | None = None) -> dict[str, Any]:
    """Validate and submit a scene to the configured Newton executor."""
    return _tools().simulate_scene(scene_name, frames)


@mcp.tool()
def render_scene(scene_name: str, output: str) -> dict[str, Any]:
    """Validate and submit a scene to the configured Newton renderer."""
    return _tools().render_scene(scene_name, output)


@mcp.tool()
def export_scene(scene_name: str, output: str, format: str = "json") -> dict[str, Any]:
    """Export a scene; the initial executor supports JSON."""
    return _tools().export_scene(scene_name, output, format)


def main() -> None:
    """Run the server over stdio or Streamable HTTP."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transport", choices=("stdio", "streamable-http"), default="stdio")
    args = parser.parse_args()
    mcp.run(transport=args.transport)


if __name__ == "__main__":
    main()
