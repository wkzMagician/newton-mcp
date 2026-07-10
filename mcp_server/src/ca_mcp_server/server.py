# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""FastMCP adapter for the animation scene tools."""

from __future__ import annotations

import argparse
import os
from functools import lru_cache
from typing import Any, Literal

from mcp.server.fastmcp import FastMCP

from ca_framework.mcp import SceneTools
from ca_framework.scene import SceneExecutorLocal, SceneStore

from .dto import (
    ActionPatchDTO,
    ActionSpecDTO,
    ConstraintPatchDTO,
    ConstraintSpecDTO,
    FieldPatchDTO,
    FieldSpecDTO,
    ObjectPatchDTO,
    ObjectSpecDTO,
    SceneDTO,
    ScenePatchDTO,
)

mcp = FastMCP(
    "Newton Scene Editor",
    instructions="Create and edit backend-neutral animation scenes, then simulate, render, or export them with Newton.",
    json_response=True,
)


@lru_cache(maxsize=1)
def _tools() -> SceneTools:
    workspace = os.environ.get("CA_SCENE_WORKSPACE", ".ca-scenes")
    return SceneTools(SceneStore(workspace), SceneExecutorLocal())


@mcp.tool()
def get_capabilities() -> dict[str, Any]:
    """Describe supported objects, actions, solvers, and outputs."""
    return _tools().get_capabilities()


@mcp.tool()
def get_scene_schema() -> dict[str, Any]:
    """Return the discriminated JSON schema for scene IR version 2."""
    return SceneDTO.model_json_schema()


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
def apply_scene_patch(scene_name: str, patch: ScenePatchDTO) -> dict[str, Any]:
    """Transactionally apply a recursive merge patch to a scene."""
    return _tools().apply_scene_patch(scene_name, patch.model_dump(exclude_unset=True))


@mcp.tool()
def add_object(scene_name: str, spec: ObjectSpecDTO) -> dict[str, Any]:
    """Add a rigid body, cloth, smoke/liquid fluid, or compound container."""
    return _tools().add_object(scene_name, spec.model_dump())


@mcp.tool()
def add_constraint(scene_name: str, spec: ConstraintSpecDTO) -> dict[str, Any]:
    """Add a constraint; spec.kind is fixed-point or distance."""
    return _tools().add_constraint(scene_name, spec.model_dump())


@mcp.tool()
def add_field(scene_name: str, spec: FieldSpecDTO) -> dict[str, Any]:
    """Add a force or acceleration field; spec.kind is uniform or radial."""
    return _tools().add_field(scene_name, spec.model_dump())


@mcp.tool()
def add_action(scene_name: str, spec: ActionSpecDTO) -> dict[str, Any]:
    """Add a transform, impulse, force, or fluid emission action."""
    return _tools().add_action(scene_name, spec.model_dump())


@mcp.tool()
def update_object(scene_name: str, item_id: str, patch: ObjectPatchDTO) -> dict[str, Any]:
    """Update one object; patch.kind must match its existing object kind."""
    return _tools().update_object(scene_name, item_id, patch.model_dump(exclude_unset=True))


@mcp.tool()
def update_constraint(scene_name: str, item_id: str, patch: ConstraintPatchDTO) -> dict[str, Any]:
    """Update one constraint; patch.kind must match its existing constraint kind."""
    return _tools().update_constraint(scene_name, item_id, patch.model_dump(exclude_unset=True))


@mcp.tool()
def update_field(scene_name: str, item_id: str, patch: FieldPatchDTO) -> dict[str, Any]:
    """Update one field; patch.kind must match its existing field kind."""
    return _tools().update_field(scene_name, item_id, patch.model_dump(exclude_unset=True))


@mcp.tool()
def update_action(scene_name: str, item_id: str, patch: ActionPatchDTO) -> dict[str, Any]:
    """Update one action; patch.kind must match its existing action kind."""
    return _tools().update_action(scene_name, item_id, patch.model_dump(exclude_unset=True))


@mcp.tool()
def remove_item(
    scene_name: str, collection: Literal["objects", "constraints", "fields", "actions"], item_id: str
) -> None:
    """Remove one scene item when it has no remaining references."""
    _tools().remove_item(scene_name, collection, item_id)


@mcp.tool()
def simulate_scene(scene_name: str, frames: int | None = None) -> dict[str, Any]:
    """Validate and submit a scene to the configured Newton executor."""
    return _tools().simulate_scene(scene_name, frames)


@mcp.tool()
def validate_scene(scene_name: str) -> dict[str, Any]:
    """Return structured errors, suggestions, and resource estimates."""
    return _tools().validate_scene(scene_name)


@mcp.tool()
def preview_scene(scene_name: str, frames: int | None = None) -> dict[str, Any]:
    """Run a short low-cost preview and return keyframes and metrics."""
    return _tools().preview_scene(scene_name, frames)


@mcp.tool()
def run_scene(scene_name: str, output_dir: str) -> dict[str, Any]:
    """Submit a persistent simulation and rendering bundle asynchronously."""
    return _tools().run_scene(scene_name, output_dir)


@mcp.tool()
def get_job(job_id: str) -> dict[str, Any]:
    """Poll an asynchronous scene job."""
    return _tools().get_job(job_id)


@mcp.tool()
def cancel_job(job_id: str) -> dict[str, Any]:
    """Cancel a queued scene job."""
    return _tools().cancel_job(job_id)


@mcp.tool()
def render_scene(scene_name: str, output: str) -> dict[str, Any]:
    """Validate and submit a scene to the configured Newton renderer."""
    return _tools().render_scene(scene_name, output)


@mcp.tool()
def export_scene(scene_name: str, output: str, format: str = "json") -> dict[str, Any]:
    """Export a scene; the initial executor supports JSON."""
    return _tools().export_scene(scene_name, output, format)


@mcp.tool()
def export_program(scene_name: str, program_output: str, scene_output: str) -> dict[str, Any]:
    """Export a standalone Python program and matching scene JSON."""
    return _tools().export_program(scene_name, program_output, scene_output)


def main() -> None:
    """Run the server over stdio or Streamable HTTP."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--transport", choices=("stdio", "streamable-http"), default="stdio")
    args = parser.parse_args()
    mcp.run(transport=args.transport)


if __name__ == "__main__":
    main()
