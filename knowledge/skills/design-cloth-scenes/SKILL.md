---
name: design-cloth-scenes
description: Design Newton cloth animation scenes with particle resolution, thickness, anchors, distance constraints, collision objects, and external fields. Use when a prompt involves fabric, sheets, flags, curtains, nets, or other thin deformable surfaces.
---

# Design cloth scenes

1. Represent each thin deformable sheet as a `cloth` object with physical size in meters.
2. Choose resolution from the visible deformation scale; begin around 11--16 vertices per axis. Treat 24 per axis as a preview ceiling, not a default target.
3. Keep particle spacing approximately uniform in both axes.
4. Use `fixed-point` constraints for pins, hooks, curtain rails, and flag attachments.
5. Use rigid objects as colliders rather than encoding collision behavior as constraints.
6. Prefer `solver: auto` or `solver: vbd` for cloth. Use XPBD only when a tested interaction specifically requires it.
7. Start surface density in the low single-digit kg/m2 range. Very low surface density makes ordinary forces and impacts disproportionately violent.
8. Set area stiffness alongside stretch stiffness; begin with comparable values. Use bend stiffness at least one to two orders below in-plane stiffness.
9. Use material damping in the low single digits as a stable starting range. Values far below one provide little control for impacts or hanging motion.
10. Tune in this order: geometry and clearance, surface density, paired area/stretch stiffness, damping, substeps, solver iterations, then resolution. Change only one category per repair.
7. Treat thickness as collision thickness [m], not visual fabric thickness alone.

Verify that fixed points correspond to cloth vertices when compiling to Newton. Avoid over-constraining the same region with contradictory anchors.

Never increase resolution, substeps, and solver iterations together. Avoid enabling self-collision unless the requested folds require it because it adds contact work and can amplify unstable material choices. For `cloth_deformation_failure`, first reduce impact or forcing, then improve surface density, paired area/stretch stiffness, and damping; for `energy_explosion`, remove extreme forces and stiffness before adding substeps. Explicit vertex-index pins depend on topology, while edge and UV-corner selectors remain meaningful after safe preview downsampling.
