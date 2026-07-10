---
name: design-fluid-scenes
description: Design particle-fluid animation scenes for Newton or MPM backends, including source volumes, particle spacing, containers, gravity, and force fields. Use when a prompt involves liquids, granular media, splashes, pouring, or volume-filling particle simulations.
---

# Design fluid scenes

1. Represent the initial occupied volume as a `fluid` object and set `particle_spacing` in meters.
2. Use static rigid bodies for floors, walls, obstacles, and containers.
3. Leave clearance of at least one particle spacing between fluid particles and container boundaries.
4. Select `mpm` when the material or Newton implementation requires it; otherwise keep the backend choice explicit in scene metadata.
5. Use uniform or radial fields for wind, attraction, repulsion, and other bulk effects.
6. Estimate particle count from volume divided by particle spacing cubed before increasing resolution.
7. Prefer a short low-resolution validation run before a final render.

State the intended material behavior in metadata until viscosity and constitutive parameters are part of the scene schema.
