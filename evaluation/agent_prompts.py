# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Natural-language prompt fixtures for black-box agent evaluation."""

from __future__ import annotations

AGENT_PROMPTS = {
    "01_rigid_contacts": """Create a five-second animation showing three freely falling rigid objects: a box, a sphere, and a capsule. Start them at visibly different horizontal positions and heights above a ground plane so gravity makes them fall, contact the ground, and interact naturally. Frame all three objects clearly throughout the animation.""",
    "02_domino_wave": """Create a five-second animation of seven upright dominoes arranged in one straight, evenly spaced row. Give the first domino an initial push toward the rest so that contact propagates sequentially along the row and all dominoes topple as a wave. Keep the complete row visible.""",
    "03_ramp_bounce": """Create a five-second animation in which a low-friction ball rolls or slides down an inclined ramp, bounces noticeably, and is finally stopped by a short rigid barrier at the lower end. The camera should clearly show the ramp, the ball's travel, its bounce, and the stop.""",
    "04_cloth_ball": """Create a five-second animation of a square horizontal cloth sheet held fixed at all four corners. Drop a dense rigid ball onto its center. The cloth must visibly sag and deform under the impact while its four corners remain fixed, without tearing or flipping. Keep the ball-cloth contact clearly visible.""",
    "05_hanging_cloth": """Create a five-second animation of a vertical rectangular cloth hanging from its entire top edge. Apply a brief sideways force near the beginning so the free lower portion swings and then gradually settles under gravity while the top edge stays fixed. Do not add a ground plane, and frame the whole cloth.""",
    "06_cloth_blocks": """Create a five-second animation of a free horizontal cloth sheet falling under gravity onto two adjacent static blocks of different heights. The cloth should contact both blocks, drape and fold over them, and settle stably without passing through them or becoming excessively stretched. Show both blocks and the full cloth.""",
    "07_smoke_partitions": """Create a five-second animation inside a closed rectangular chamber with no ground plane. Emit smoke briefly from a small source near the lower left, with initial motion upward and toward the right. The smoke should expand through the chamber, spread across multiple regions, and remain visible rather than disappearing immediately. Frame the complete chamber.""",
    "08_liquid_pour": """Create a five-second animation of water pouring from a tilted container on the left into a lower open receiving container on the right. Most of the emitted water should transfer into and remain inside the right container without leaking through its walls. Do not add a ground plane, and clearly show both containers and the stream.""",
    "09_liquid_splash": """Create a five-second animation of a heavy rigid ball falling into a pool of water held by an open container. The ball should enter and sink into the water, producing a visible upward splash. The liquid must remain contained. Do not add a ground plane, and frame both the pool and the falling ball.""",
    "10_floating_blocks": """Create a five-second animation of three lightweight rigid blocks floating together in a water-filled open pool. They should bob, rotate, and make contact with one another, then show damping toward a calmer state while remaining near the water surface. Keep all blocks and the pool visible, with no separate ground plane.""",
}


def build_agent_prompt(case_name: str, output_dir: str) -> str:
    """Build the complete agent instruction for one black-box experiment."""
    task = AGENT_PROMPTS[case_name]
    return f"""Use only the ca-scene MCP tools to create and run the requested animation. Do not inspect the host repository, search for reference scenes, or write an alternative simulation implementation.

Requested animation:
{task}

Execution requirements:
1. Call get_capabilities and get_scene_schema before authoring the scene.
2. Create exactly one scene with the name {case_name}.
3. Inspect the completed scene with get_scene, then call validate_scene.
4. If validation or preview reports a problem, repair it with MCP editing tools. Use at most three repair rounds.
5. Call preview_scene before the final run.
6. Call run_scene with output_dir set exactly to {output_dir}.
7. Poll get_job only while it reports queued or running. Treat completed, failed, physics_failed, cancelled, and interrupted as terminal statuses.
8. Report success only when the job completed and animation.mp4, scene.json, program.py, metrics.json, diagnostics.jsonl, and cache/ were produced.
"""


__all__ = ["AGENT_PROMPTS", "build_agent_prompt"]
