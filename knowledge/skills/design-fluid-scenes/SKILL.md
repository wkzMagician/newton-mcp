---
name: design-fluid-scenes
description: Design Newton smoke and APIC/FLIP liquid scenes, including domains, emitters, particle spacing, containers, gravity, and force fields. Use when a prompt involves smoke, liquids, splashes, pouring, buoyancy, or volume-filling simulations.
---

# Design fluid scenes

1. Choose `phase: smoke` for a MAC-grid gas or `phase: liquid` for APIC/FLIP particles.
2. Represent liquid initial volume with `size` and `particle_spacing` [m]; describe smoke sources with timed `emitters`.
3. Use `motion: static` rigid bodies or `container` objects for floors, walls, and obstacles.
4. Leave clearance of at least one particle spacing between fluid particles and container boundaries.
5. Use `solver: auto`; smoke routes to `smoke` and liquid routes to `apic`.
6. Use uniform or radial fields for wind, attraction, repulsion, and other bulk effects.
7. Estimate particle count from volume divided by particle spacing cubed before increasing resolution.
8. Prefer a short low-resolution validation run before a final render.

Specify liquid density, viscosity, surface tension, and `flip_ratio`, or smoke buoyancy and dissipation, directly in the scene IR.
