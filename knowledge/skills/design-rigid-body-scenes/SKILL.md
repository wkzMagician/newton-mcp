---
name: design-rigid-body-scenes
description: Design Newton animation scenes containing rigid bodies, primitive collision geometry, contacts, gravity, and rigid-body motion. Use when translating a prompt about solid moving objects into scene objects and simulation settings.
---

# Design rigid-body scenes

1. Identify each independently moving solid as a `rigid` object.
2. Approximate collision geometry with `box`, `sphere`, or `capsule`; keep visual detail separate from collision detail.
3. Express all positions and sizes in meters, time in seconds, and acceleration in meters per second squared.
4. Mark floors and immovable obstacles with `dynamic: false`.
5. Use a uniform acceleration field for gravity only when the scene needs gravity different from `settings.gravity`.
6. Start with `xpbd`; select `vbd` only after confirming all requested features are supported by that backend.
7. Check that dynamic objects do not begin deeply intersecting and that the simulation duration is long enough to show the requested action.

Before simulation, inspect the complete scene with `get_scene`. Prefer stable, simple geometry over unnecessary detail.
