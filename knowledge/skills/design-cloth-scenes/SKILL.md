---
name: design-cloth-scenes
description: Design Newton cloth animation scenes with particle resolution, thickness, anchors, distance constraints, collision objects, and external fields. Use when a prompt involves fabric, sheets, flags, curtains, nets, or other thin deformable surfaces.
---

# Design cloth scenes

1. Represent each thin deformable sheet as a `cloth` object with physical size in meters.
2. Choose resolution from the visible deformation scale; begin near `16 x 16` and increase only when folds require it.
3. Keep particle spacing approximately uniform in both axes.
4. Use `fixed-point` constraints for pins, hooks, curtain rails, and flag attachments.
5. Use rigid objects as colliders rather than encoding collision behavior as constraints.
6. Increase substeps before increasing mesh resolution when the scene is unstable.
7. Treat thickness as collision thickness [m], not visual fabric thickness alone.

Verify that fixed points correspond to cloth vertices when compiling to Newton. Avoid over-constraining the same region with contradictory anchors.
