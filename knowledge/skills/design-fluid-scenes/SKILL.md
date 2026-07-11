---
name: design-fluid-scenes
description: Design Newton smoke and APIC/FLIP liquid scenes, including domains, emitters, particle spacing, containers, gravity, and force fields. Use when a prompt involves smoke, liquids, splashes, pouring, buoyancy, or volume-filling simulations.
---

# Design fluid scenes

1. Choose `phase: smoke` for a MAC-grid gas or `phase: liquid` for APIC/FLIP particles.
2. Represent liquid initial volume with `size` and `particle_spacing` [m]; describe smoke sources with timed `emitters`.
3. Use `motion: static` rigid bodies or `container` objects for floors, walls, and obstacles.
4. Leave clearance of at least one particle spacing between fluid particles and container boundaries; use more clearance for fast rigid impacts.
5. Use `solver: auto`; smoke routes to `smoke` and liquid routes to `apic`.
6. Use uniform or radial fields for wind, attraction, repulsion, and other bulk effects.
7. Estimate initial particles as `volume / spacing^3`; add emitter cross-section samples multiplied by active simulation substeps. For APIC adaptive resampling, also reserve roughly five particles per liquid grid cell. Capacity must cover the largest of these estimates.
8. Prefer a short low-resolution validation run before a final render.

Specify liquid density, viscosity, surface tension, and `flip_ratio`, or smoke buoyancy and dissipation, directly in the scene IR.

Start smoke at 20--30 FPS and liquid at 20--25 FPS. Higher FLIP ratios preserve splash energy but need more boundary clearance; viscosity damps motion; smaller spacing increases particle count cubically. For `pressure_nonconvergence`, fix clearance and spacing before raising iterations. For capacity failures, increase spacing or shorten/narrow emission rather than continually raising capacity.
