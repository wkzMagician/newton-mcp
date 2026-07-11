---
name: orchestrate-animation-scene
description: Orchestrate prompt-to-scene editing, validation, preview repair, final execution, and artifact inspection for Newton animation jobs.
---

# Orchestrate an animation scene

1. Translate the prompt only into supported schema-version-2 IR primitives; do not generate Warp kernels.
2. Check structured resource budgets before preview. Perform at most three repair rounds. Each round is `apply_scene_patch`, `validate_scene`, then `preview_scene`.
3. Repair only diagnostics tied to the requested outcome. Do not silently simplify away contacts, fluids, constraints, or motion.
4. Keep preview limits separate from the stored IR: one second, 640 by 360 pixels, reduced fluid grids, and at most 50,000 liquid particles.
5. Once validation and preview metrics are credible, call `run_scene(scene_name, output_dir)`.
6. Poll `get_job` and inspect its stage, frame progress, diagnostics, and artifact paths. Use `cancel_job` only when the requested job should stop.
   Poll only while status is `queued` or `running`; `completed`, `failed`, `physics_failed`, `cancelled`, and `interrupted` are terminal.
7. Treat a render failure as a render-stage problem when `cache/manifest.json` is complete; reuse the cache instead of repeating simulation.
8. Deliver `animation.mp4`, `scene.json`, `program.py`, `metrics.json`, `diagnostics.jsonl`, and `cache/` together.

In each repair round, change one category of parameters only. Do not treat steadily increasing resolution, substeps, iterations, particle capacity, or grid size as a repair strategy. Address resource errors by reducing the dominant metric; address energy failures by reducing forcing/stiffness; address deformation failures by reducing impact and balancing damping; address pressure failures by fixing boundary clearance and particle spacing.
