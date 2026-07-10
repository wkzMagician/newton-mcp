---
name: apply-fields-and-constraints
description: Add and validate fixed-point constraints, distance constraints, uniform fields, and radial fields in animation scenes. Use when a prompt describes attachments, rods, ropes, attraction, repulsion, wind, gravity, or other external influence.
---

# Apply fields and constraints

1. Decide whether the prompt describes a kinematic relationship (constraint) or an environmental influence (field).
2. Use `fixed-point` to anchor one object point in world space.
3. Use `distance` to maintain separation between two object-local points; specify distance in meters and dimensionless normalized stiffness.
4. Use a `uniform` field for constant force [N] or acceleration [m/s^2].
5. Use a `radial` field for attraction or repulsion around a center; use the sign of `strength` consistently.
6. Set `object_ids` only when the influence is selective; omit it for a global field.
7. Inspect object references before deletion and remove dependent constraints or fields first.

Do not emulate a constraint with an extremely strong force unless the requested motion is intentionally compliant.
