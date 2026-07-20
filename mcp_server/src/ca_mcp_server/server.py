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
from ca_framework.mcp.optimization_tools import OptimizationTools
from ca_framework.scene import SceneExecutorLocal, SceneStore

from .dto import (
    ActionPatchDTO,
    ActionSpecDTO,
    CameraLookAtDTO,
    ConstraintPatchDTO,
    ConstraintSpecDTO,
    FieldPatchDTO,
    FieldSpecDTO,
    ObjectiveSettingsDTO,
    ObjectPatchDTO,
    ObjectSpecDTO,
    OptimizationPlanDTO,
    SceneDTO,
    ScenePatchDTO,
    TaskSpecDTO,
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


@lru_cache(maxsize=1)
def _optimization_tools() -> OptimizationTools:
    tools = _tools()
    return OptimizationTools(tools.store, tools.executor)


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
def set_camera(scene_name: str, camera: CameraLookAtDTO) -> dict[str, Any]:
    """Set a fixed look-at camera; position and target use metres."""
    return _tools().set_camera(scene_name, **camera.model_dump())


@mcp.tool()
def apply_scene_patch(scene_name: str, patch: ScenePatchDTO) -> dict[str, Any]:
    """Transactionally apply a recursive merge patch to a scene."""
    return _tools().apply_scene_patch(scene_name, patch.model_dump(exclude_unset=True))


@mcp.tool()
def add_object(scene_name: str, spec: ObjectSpecDTO) -> dict[str, Any]:
    """Add a rigid body, cloth, smoke/liquid fluid, or compound container.

    For cloth, ``transform.position`` is the undeformed sheet center rather
    than a grid corner. Its ``size`` extends equally in local x/y around that
    point, then ``rotation`` is applied. Use ``preview_scene`` to verify
    placement.
    """
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
    """Update one object; ``patch.kind`` must match its existing object kind.

    A cloth transform position always denotes its undeformed sheet center.
    """
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
def create_optimization_plan(
    scene_name: str,
    plan: OptimizationPlanDTO,
    task_spec: TaskSpecDTO | None = None,
    objective_settings: ObjectiveSettingsDTO | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Create an optimization plan outside scene metadata."""
    return _optimization_tools().create_plan(
        scene_name,
        plan.model_dump(),
        task_spec=task_spec.model_dump() if task_spec is not None else None,
        objective_settings=objective_settings.model_dump() if objective_settings is not None else None,
        overwrite=overwrite,
    )


@mcp.tool()
def validate_optimization_plan(plan_name: str) -> dict[str, Any]:
    """Validate parameter paths, ranges, active count, and midpoint scene."""
    return _optimization_tools().validate_plan(plan_name)


@mcp.tool()
def start_optimization(plan_name: str, output_dir: str | None = None) -> dict[str, Any]:
    """Start an independent serialized Optuna ask/tell study."""
    return _optimization_tools().start(plan_name, output_dir)


@mcp.tool()
def get_optimization_job(job_id: str) -> dict[str, Any]:
    """Poll optimization progress and best trial."""
    return _optimization_tools().get_job(job_id)


@mcp.tool()
def list_optimization_trials(job_id: str) -> list[dict[str, Any]]:
    """List available trial results."""
    return _optimization_tools().list_trials(job_id)


@mcp.tool()
def get_optimization_trial(job_id: str, trial_number: int) -> dict[str, Any]:
    """Read one trial's parameters, metrics, and artifacts."""
    return _optimization_tools().get_trial(job_id, trial_number)


@mcp.tool()
def apply_optimization_trial(job_id: str, trial_number: int, scene_name: str) -> dict[str, Any]:
    """Apply a selected trial to a stored scene."""
    return _optimization_tools().apply_trial(job_id, trial_number, scene_name)


@mcp.tool()
def compare_optimization_trials(job_id: str, trial_numbers: list[int]) -> list[dict[str, Any]]:
    """Compare selected trials without modifying a scene."""
    return _optimization_tools().compare_trials(job_id, trial_numbers)


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
