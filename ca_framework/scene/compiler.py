# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Deterministic scene compilation and reproducible program export."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import warp as wp

import newton

from .model import (
    ActionEmit,
    ActionTransform,
    ConstraintDistance,
    ConstraintFixedPoint,
    ObjectCloth,
    ObjectContainer,
    ObjectFluid,
    ObjectRigid,
    Scene,
    ShapePrimitive,
    VertexSelector,
)
from .validation import select_pipeline, validate_scene


@dataclass(slots=True)
class CompiledScene:
    """Validated execution description consumed by local runners."""

    scene: Scene
    pipeline: list[str]
    container_colliders: dict[str, list[dict[str, Any]]]
    cloth_particle_indices: dict[str, list[int]]
    pinned_particles: dict[str, list[int]]
    model: Any
    state_0: Any
    state_1: Any
    control: Any
    contacts: Any
    solver: Any
    body_indices: dict[str, int]
    shape_indices: dict[str, list[int]]
    initial_shape_scales: np.ndarray
    initial_particle_q: np.ndarray
    fluid_solvers: dict[str, Any]


class SceneCompilerNewton:
    """Compile scene IR into a deterministic Newton execution description."""

    def compile(self, scene: Scene) -> CompiledScene:
        """Validate and compile a scene, raising with structured diagnostics."""
        report = validate_scene(scene)
        if not report["valid"]:
            raise ValueError(json.dumps({"error": "scene_validation_failed", **report}, sort_keys=True))
        containers = {
            item.id: self._container_shapes(item)
            for item in scene.objects.values()
            if isinstance(item, ObjectContainer)
        }
        cloth_indices: dict[str, list[int]] = {}
        pinned: dict[str, list[int]] = {}
        particle_start = 0
        for item in scene.objects.values():
            if not isinstance(item, ObjectCloth):
                continue
            count = item.resolution[0] * item.resolution[1]
            cloth_indices[item.id] = list(range(particle_start, particle_start + count))
            local_pins = {
                index for selector in item.pinned for index in self._selector_indices(selector, item.resolution)
            }
            for constraint in scene.constraints.values():
                if isinstance(constraint, ConstraintFixedPoint) and constraint.object_id == item.id:
                    if constraint.selector is not None:
                        local_pins.update(self._selector_indices(constraint.selector, item.resolution))
                    else:
                        x = round(constraint.point[0] / item.size[0] * (item.resolution[0] - 1))
                        y = round(constraint.point[1] / item.size[1] * (item.resolution[1] - 1))
                        x = max(0, min(item.resolution[0] - 1, x))
                        y = max(0, min(item.resolution[1] - 1, y))
                        local_pins.add(y * item.resolution[0] + x)
            pinned[item.id] = [particle_start + index for index in sorted(local_pins)]
            particle_start += count
        (
            model,
            state_0,
            state_1,
            control,
            contacts,
            solver,
            body_indices,
            shape_indices,
            initial_shape_scales,
            fluid_solvers,
        ) = self._compile_newton(scene, containers, pinned, cloth_indices)
        return CompiledScene(
            scene=scene,
            pipeline=select_pipeline(scene),
            container_colliders=containers,
            cloth_particle_indices=cloth_indices,
            pinned_particles=pinned,
            model=model,
            state_0=state_0,
            state_1=state_1,
            control=control,
            contacts=contacts,
            solver=solver,
            body_indices=body_indices,
            shape_indices=shape_indices,
            initial_shape_scales=initial_shape_scales,
            initial_particle_q=(
                state_0.particle_q.numpy().copy()
                if state_0.particle_q is not None
                else np.empty((0, 3), dtype=np.float32)
            ),
            fluid_solvers=fluid_solvers,
        )

    def _compile_newton(
        self,
        scene: Scene,
        containers: dict[str, list[dict[str, Any]]],
        pinned: dict[str, list[int]],
        cloth_indices: dict[str, list[int]],
    ) -> tuple[Any, Any, Any, Any, Any, Any, dict[str, int], dict[str, list[int]], np.ndarray, dict[str, Any]]:
        """Build the concrete Newton objects required by the frame runtime."""
        builder = newton.ModelBuilder(up_axis="Z")
        body_indices: dict[str, int] = {}
        shape_indices: dict[str, list[int]] = {}
        if scene.render.ground:
            builder.add_ground_plane()
        for item in scene.objects.values():
            if isinstance(item, ObjectRigid):
                body = -1
                if item.motion != "static":
                    fixed_constraint = next(
                        (
                            constraint
                            for constraint in scene.constraints.values()
                            if isinstance(constraint, ConstraintFixedPoint)
                            and constraint.object_id == item.id
                            and constraint.selector is None
                        ),
                        None,
                    )
                    if fixed_constraint is None or item.motion == "kinematic":
                        body = builder.add_body(
                            xform=self._wp_transform(item.transform.position, item.transform.rotation),
                            label=item.id,
                            lock_inertia=item.motion == "kinematic",
                        )
                    else:
                        anchor_offset = self._rotate_vector(fixed_constraint.point, item.transform.rotation)
                        world_anchor = tuple(item.transform.position[axis] + anchor_offset[axis] for axis in range(3))
                        body = builder.add_link(
                            xform=self._wp_transform(item.transform.position, item.transform.rotation),
                            label=item.id,
                        )
                        joint = builder.add_joint_ball(
                            parent=-1,
                            child=body,
                            parent_xform=self._wp_transform(world_anchor, (0.0, 0.0, 0.0, 1.0)),
                            child_xform=self._wp_transform(fixed_constraint.point, (0.0, 0.0, 0.0, 1.0)),
                            label=f"{item.id}_fixed_point",
                        )
                        builder.add_articulation([joint], label=f"{item.id}_fixed_point_articulation")
                    body_indices[item.id] = body
                    builder.set_body_velocity(
                        body,
                        wp.spatial_vector(*item.linear_velocity, *item.angular_velocity),
                    )
                primitives = (
                    item.shapes if item.shape == "compound" else [ShapePrimitive(kind=item.shape, size=item.size)]
                )
                for primitive in primitives:
                    shape_indices.setdefault(item.id, []).append(self._add_primitive(builder, body, item, primitive))
            elif isinstance(item, ObjectContainer):
                body = -1
                shapes = containers[item.id]
                if item.motion != "static":
                    body = builder.add_body(
                        xform=self._wp_transform(item.transform.position, item.transform.rotation),
                        label=item.id,
                        lock_inertia=item.motion == "kinematic",
                    )
                    body_indices[item.id] = body
                    shapes = self._container_local_shapes(item)
                for shape in shapes:
                    shape_index = builder.add_shape_box(
                        body,
                        xform=self._wp_transform(shape["position"], shape.get("rotation", (0.0, 0.0, 0.0, 1.0))),
                        hx=shape["size"][0] * 0.5,
                        hy=shape["size"][1] * 0.5,
                        hz=shape["size"][2] * 0.5,
                        cfg=self._shape_config(builder, item),
                        label=f"{item.id}_wall",
                    )
                    shape_indices.setdefault(item.id, []).append(shape_index)
            elif isinstance(item, ObjectCloth):
                width, height = item.resolution
                scale_x, scale_y, _ = item.transform.scale
                particle_mass = item.surface_density * item.size[0] * item.size[1] / (width * height)
                builder.add_cloth_grid(
                    pos=wp.vec3(*item.transform.position),
                    rot=wp.quat(*item.transform.rotation),
                    vel=wp.vec3(),
                    dim_x=width - 1,
                    dim_y=height - 1,
                    cell_x=item.size[0] * scale_x / (width - 1),
                    cell_y=item.size[1] * scale_y / (height - 1),
                    mass=particle_mass,
                    tri_ke=item.stretch_stiffness,
                    tri_ka=item.area_stiffness or item.stretch_stiffness,
                    tri_kd=item.stretch_damping if item.stretch_damping is not None else item.damping,
                    tri_drag=item.air_drag,
                    edge_ke=item.bend_stiffness,
                    edge_kd=item.bend_damping if item.bend_damping is not None else item.damping,
                    add_springs=False,
                    particle_radius=item.collision_radius or item.thickness,
                )
                for particle_index in pinned[item.id]:
                    builder.particle_mass[particle_index] = 0.0
        for constraint in scene.constraints.values():
            if not isinstance(constraint, ConstraintDistance):
                continue
            body_a = body_indices.get(constraint.object_a)
            body_b = body_indices.get(constraint.object_b)
            if body_a is None or body_b is None:
                continue
            builder.add_joint_distance(
                parent=body_a,
                child=body_b,
                parent_xform=self._wp_transform(constraint.point_a, (0.0, 0.0, 0.0, 1.0)),
                child_xform=self._wp_transform(constraint.point_b, (0.0, 0.0, 0.0, 1.0)),
                min_distance=constraint.distance,
                max_distance=constraint.distance,
                collision_filter_parent=False,
            )
        pipeline = select_pipeline(scene)
        if pipeline[0] == "vbd":
            builder.color(include_bending=True)
        model = builder.finalize()
        initial_shape_scales = model.shape_scale.numpy().copy()
        model.set_gravity(scene.settings.gravity)
        state_0 = model.state()
        state_1 = model.state()
        control = model.control()
        contacts = model.contacts()
        if pipeline[0] == "vbd":
            solver = newton.solvers.SolverVBD(model, iterations=scene.settings.cloth.iterations)
        else:
            solver = newton.solvers.SolverXPBD(
                model,
                iterations=max(scene.settings.rigid.iterations, scene.settings.cloth.iterations),
                enable_restitution=True,
            )
        fluid_solvers = {}
        for item in scene.objects.values():
            if not isinstance(item, ObjectFluid):
                continue
            if item.phase == "smoke":
                emitter_type = newton.solvers.SolverFluidSmoke.Emitter
                emitters = [
                    emitter_type(
                        position=emitter.position,
                        size=emitter.size,
                        start_time=next(
                            (
                                action.start_time
                                for action in scene.actions.values()
                                if isinstance(action, ActionEmit)
                                and action.object_id == item.id
                                and action.emitter_index == index
                            ),
                            emitter.start_time,
                        ),
                        end_time=next(
                            (
                                action.end_time
                                for action in scene.actions.values()
                                if isinstance(action, ActionEmit)
                                and action.object_id == item.id
                                and action.emitter_index == index
                            ),
                            emitter.end_time,
                        ),
                        density=emitter.density,
                        velocity=emitter.velocity,
                    )
                    for index, emitter in enumerate(item.emitters)
                ]
                domain_size = tuple(item.size[axis] * item.transform.scale[axis] for axis in range(3))
                domain_min = tuple(item.transform.position[axis] - domain_size[axis] * 0.5 for axis in range(3))
                smoke_solver = newton.solvers.SolverFluidSmoke(
                    model,
                    res=item.grid_resolution,
                    domain_size=domain_size,
                    domain_min=domain_min,
                    buoyancy=item.buoyancy,
                    dissipation=item.dissipation,
                    pressure_iters=scene.settings.fluid.pressure_iterations,
                    emitters=emitters,
                    drag_density=scene.settings.coupling.smoke_drag_density,
                    drag_coefficient=scene.settings.coupling.smoke_drag_coefficient,
                    coupling_iterations=(
                        scene.settings.coupling.iterations if scene.settings.coupling.mode == "strong" else 1
                    ),
                    coupling_relaxation=scene.settings.coupling.relaxation,
                )
                boundary_type = newton.solvers.SolverFluidSmoke.Boundary
                for obstacle in scene.objects.values():
                    if isinstance(obstacle, ObjectRigid):
                        smoke_solver.add_boundary(
                            boundary_type(
                                position=obstacle.transform.position,
                                half_extent=tuple(
                                    obstacle.size[axis] * obstacle.transform.scale[axis] * 0.5 for axis in range(3)
                                ),
                                body=body_indices.get(obstacle.id),
                            )
                        )
                if scene.settings.coupling.cloth_fluid:
                    cloth_boundary_type = newton.solvers.SolverFluidSmoke.ClothBoundary
                    for cloth in (value for value in scene.objects.values() if isinstance(value, ObjectCloth)):
                        indices = cloth_indices[cloth.id]
                        width, height = cloth.resolution
                        triangles = []
                        for y in range(height - 1):
                            for x in range(width - 1):
                                lower = indices[y * width + x]
                                triangles.extend(
                                    (
                                        (lower, lower + 1, lower + width + 1),
                                        (lower, lower + width + 1, lower + width),
                                    )
                                )
                        smoke_solver.add_cloth_boundary(
                            cloth_boundary_type(cloth.id, np.asarray(triangles, dtype=np.int32))
                        )
                fluid_solvers[item.id] = smoke_solver
            else:
                domain_size = tuple(item.size[axis] * item.transform.scale[axis] for axis in range(3))
                domain_min = tuple(item.transform.position[axis] - domain_size[axis] * 0.5 for axis in range(3))
                scene_containers = [
                    container for container in scene.objects.values() if isinstance(container, ObjectContainer)
                ]
                if scene_containers:
                    container_positions = {
                        container.id: [container.transform.position]
                        + [
                            keyframe.transform.position
                            for action in scene.actions.values()
                            if isinstance(action, ActionTransform) and action.object_id == container.id
                            for keyframe in action.keyframes
                        ]
                        for container in scene_containers
                    }
                    container_min = np.asarray(
                        [
                            min(
                                position[axis] - container.inner_size[axis] * container.transform.scale[axis] * 0.5
                                for container in scene_containers
                                for position in container_positions[container.id]
                            )
                            for axis in (0, 1)
                        ]
                        + [
                            min(
                                position[2]
                                for container in scene_containers
                                for position in container_positions[container.id]
                            )
                        ]
                    )
                    container_max = np.asarray(
                        [
                            max(
                                position[axis] + container.inner_size[axis] * container.transform.scale[axis] * 0.5
                                for container in scene_containers
                                for position in container_positions[container.id]
                            )
                            for axis in (0, 1)
                        ]
                        + [
                            max(
                                position[2] + container.inner_size[2] * container.transform.scale[2]
                                for container in scene_containers
                                for position in container_positions[container.id]
                            )
                        ]
                    )
                    padding = np.asarray(item.grid_resolution, dtype=float) ** -1 * np.asarray(domain_size)
                    domain_min = tuple(container_min - padding)
                    domain_size = tuple(container_max - container_min + 2.0 * padding)
                liquid_solver = newton.solvers.SolverFluidAPIC(
                    model,
                    grid_resolution=item.grid_resolution,
                    domain_size=domain_size,
                    domain_min=domain_min,
                    flip_ratio=item.flip_ratio,
                    max_particles=scene.settings.max_particles,
                    density=item.density,
                    viscosity=item.viscosity,
                    surface_tension=item.surface_tension,
                    pressure_iters=scene.settings.fluid.pressure_iterations,
                    coupling_iterations=(
                        scene.settings.coupling.iterations if scene.settings.coupling.mode == "strong" else 1
                    ),
                    coupling_relaxation=scene.settings.coupling.relaxation,
                    cfl_number=scene.settings.fluid.cfl_number,
                    cloth_drag=scene.settings.coupling.cloth_fluid_drag,
                    cloth_permeability=scene.settings.coupling.cloth_permeability,
                    boundary_friction=scene.settings.coupling.boundary_friction,
                )
                volume_size = tuple(item.size[axis] * item.transform.scale[axis] for axis in range(3))
                minimum = tuple(item.transform.position[axis] - volume_size[axis] * 0.5 for axis in range(3))
                maximum = tuple(item.transform.position[axis] + volume_size[axis] * 0.5 for axis in range(3))
                liquid_solver.initialize_box(minimum, maximum, item.particle_spacing)
                emitter_type = newton.solvers.SolverFluidAPIC.Emitter
                for index, emitter in enumerate(item.emitters):
                    emission_action = next(
                        (
                            action
                            for action in scene.actions.values()
                            if isinstance(action, ActionEmit)
                            and action.object_id == item.id
                            and action.emitter_index == index
                        ),
                        None,
                    )
                    liquid_solver.add_emitter(
                        emitter_type(
                            position=emitter.position,
                            size=emitter.size,
                            start_time=emission_action.start_time if emission_action else emitter.start_time,
                            end_time=emission_action.end_time if emission_action else emitter.end_time,
                            velocity=emitter.velocity,
                            spacing=item.particle_spacing,
                        )
                    )
                boundary_type = newton.solvers.SolverFluidAPIC.Boundary
                for obstacle in scene.objects.values():
                    if not isinstance(obstacle, ObjectRigid):
                        continue
                    half_extent = tuple(obstacle.size[axis] * obstacle.transform.scale[axis] * 0.5 for axis in range(3))
                    liquid_solver.add_boundary(
                        boundary_type(
                            position=obstacle.transform.position,
                            half_extent=half_extent,
                            body=body_indices.get(obstacle.id),
                        )
                    )
                for container in scene_containers:
                    world_walls = containers[container.id]
                    local_walls = self._container_local_shapes(container)
                    for wall, local_wall in zip(world_walls, local_walls, strict=True):
                        liquid_solver.add_boundary(
                            boundary_type(
                                position=wall["position"],
                                half_extent=tuple(value * 0.5 for value in wall["size"]),
                                body=body_indices.get(container.id),
                                local_position=local_wall["position"],
                            )
                        )
                cloth_boundary_type = newton.solvers.SolverFluidAPIC.ClothBoundary
                for cloth in (
                    value
                    for value in scene.objects.values()
                    if isinstance(value, ObjectCloth) and scene.settings.coupling.cloth_fluid
                ):
                    indices = cloth_indices[cloth.id]
                    width, height = cloth.resolution
                    triangles = []
                    for y in range(height - 1):
                        for x in range(width - 1):
                            lower = indices[y * width + x]
                            triangles.extend(
                                (
                                    (lower, lower + 1, lower + width + 1),
                                    (lower, lower + width + 1, lower + width),
                                )
                            )
                    liquid_solver.add_cloth_boundary(
                        cloth_boundary_type(cloth.id, np.asarray(triangles, dtype=np.int32))
                    )
                fluid_solvers[item.id] = liquid_solver
        return (
            model,
            state_0,
            state_1,
            control,
            contacts,
            solver,
            body_indices,
            shape_indices,
            initial_shape_scales,
            fluid_solvers,
        )

    @staticmethod
    def _shape_config(builder: Any, item: ObjectRigid | ObjectContainer) -> Any:
        cfg = builder.ShapeConfig()
        cfg.density = item.physical_material.density
        cfg.mu = item.physical_material.friction_dynamic
        cfg.restitution = item.physical_material.restitution
        cfg.mu_rolling = item.physical_material.friction_rolling
        return cfg

    def _add_primitive(self, builder: Any, body: int, item: ObjectRigid, primitive: ShapePrimitive) -> int:
        scale = tuple(
            primitive.size[axis] * primitive.transform.scale[axis] * item.transform.scale[axis] for axis in range(3)
        )
        if body == -1:
            local = tuple(primitive.transform.position[axis] * item.transform.scale[axis] for axis in range(3))
            rotated = self._rotate_vector(local, item.transform.rotation)
            position = tuple(item.transform.position[axis] + rotated[axis] for axis in range(3))
            rotation = self._quat_multiply(item.transform.rotation, primitive.transform.rotation)
        else:
            position = primitive.transform.position
            rotation = primitive.transform.rotation
        xform = self._wp_transform(position, rotation)
        cfg = self._shape_config(builder, item)
        if primitive.kind == "box":
            return builder.add_shape_box(
                body, xform=xform, hx=scale[0] * 0.5, hy=scale[1] * 0.5, hz=scale[2] * 0.5, cfg=cfg
            )
        elif primitive.kind == "sphere":
            return builder.add_shape_sphere(body, xform=xform, radius=max(scale) * 0.5, cfg=cfg)
        return builder.add_shape_capsule(
            body,
            xform=xform,
            radius=max(scale[0], scale[1]) * 0.5,
            half_height=max(0.0, scale[2] * 0.5 - max(scale[0], scale[1]) * 0.5),
            cfg=cfg,
        )

    @staticmethod
    def _wp_transform(position: Any, rotation: Any) -> Any:
        return wp.transform(wp.vec3(*position), wp.quat(*rotation))

    @staticmethod
    def _quat_multiply(a: tuple[float, float, float, float], b: tuple[float, float, float, float]):
        ax, ay, az, aw = a
        bx, by, bz, bw = b
        return (
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        )

    def export_program(self, scene: Scene, output: str | Path) -> Path:
        """Write a standalone Python program embedding the exact scene IR."""
        report = validate_scene(scene)
        if not report["valid"]:
            raise ValueError(json.dumps({"error": "scene_validation_failed", **report}, sort_keys=True))
        output = Path(output).resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(scene.to_dict(), sort_keys=True, indent=2)
        program = f'''# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0
"""Generated reproducible Newton scene program."""

import json

from ca_framework import SceneExecutorLocal
from ca_framework.scene import Scene

SCENE_IR = json.loads(r\'''{payload}\''')

if __name__ == "__main__":
    result = SceneExecutorLocal().simulate(Scene.from_dict(SCENE_IR))
    print(json.dumps(result, sort_keys=True))
'''
        output.write_text(program, encoding="utf-8")
        return output

    @staticmethod
    def _container_shapes(container: ObjectContainer) -> list[dict[str, Any]]:
        local = SceneCompilerNewton._container_local_shapes(container)
        result = []
        for shape in local:
            rotated_position = SceneCompilerNewton._rotate_vector(shape["position"], container.transform.rotation)
            result.append(
                {
                    **shape,
                    "position": tuple(container.transform.position[axis] + rotated_position[axis] for axis in range(3)),
                    "rotation": container.transform.rotation,
                }
            )
        return result

    @staticmethod
    def _container_local_shapes(container: ObjectContainer) -> list[dict[str, Any]]:
        x, y, z = container.inner_size
        t = container.wall_thickness
        scale = container.transform.scale
        raw = [
            {"kind": "box", "size": (x + 2 * t, y + 2 * t, t), "position": (0.0, 0.0, -t / 2)},
            {"kind": "box", "size": (t, y + 2 * t, z), "position": (-(x + t) / 2, 0.0, z / 2)},
            {"kind": "box", "size": (t, y + 2 * t, z), "position": ((x + t) / 2, 0.0, z / 2)},
            {"kind": "box", "size": (x, t, z), "position": (0.0, -(y + t) / 2, z / 2)},
            {"kind": "box", "size": (x, t, z), "position": (0.0, (y + t) / 2, z / 2)},
        ]
        if container.closed:
            raw.append({"kind": "box", "size": (x + 2 * t, y + 2 * t, t), "position": (0.0, 0.0, z + t / 2)})
        return [
            {
                **shape,
                "size": tuple(shape["size"][axis] * scale[axis] for axis in range(3)),
                "position": tuple(shape["position"][axis] * scale[axis] for axis in range(3)),
            }
            for shape in raw
        ]

    @staticmethod
    def _selector_indices(selector: VertexSelector, resolution: tuple[int, int]) -> list[int]:
        """Resolve a selector to stable row-major Newton particle indices."""
        width, height = resolution
        if selector.kind == "indices":
            return list(selector.indices)
        if selector.kind == "edge":
            if selector.edge == "bottom":
                return list(range(width))
            if selector.edge == "top":
                return list(range((height - 1) * width, height * width))
            if selector.edge == "left":
                return [row * width for row in range(height)]
            return [row * width + width - 1 for row in range(height)]
        corners = {
            "bottom-left": 0,
            "bottom-right": width - 1,
            "top-left": (height - 1) * width,
            "top-right": height * width - 1,
        }
        return [corners[name] for name in selector.corners]

    @staticmethod
    def _rotate_vector(vector: tuple[float, float, float], quaternion: tuple[float, float, float, float]):
        """Rotate a vector by an ``(x, y, z, w)`` unit quaternion."""
        x, y, z, w = quaternion
        vx, vy, vz = vector
        tx, ty, tz = 2.0 * (y * vz - z * vy), 2.0 * (z * vx - x * vz), 2.0 * (x * vy - y * vx)
        return (
            vx + w * tx + y * tz - z * ty,
            vy + w * ty + z * tx - x * tz,
            vz + w * tz + x * ty - y * tx,
        )
