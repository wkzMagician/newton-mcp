# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Strongly typed, backend-neutral animation scene schema."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, TypeAlias

Vec2: TypeAlias = tuple[float, float]
Vec3: TypeAlias = tuple[float, float, float]
Quat: TypeAlias = tuple[float, float, float, float]
Color: TypeAlias = tuple[float, float, float, float]
SCHEMA_VERSION = 3


@dataclass(slots=True)
class Transform:
    """World transform.

    Attributes:
        position: Translation [m].
        rotation: Unit quaternion in ``(x, y, z, w)`` order.
        scale: Dimensionless local scale.
    """

    position: Vec3 = (0.0, 0.0, 0.0)
    rotation: Quat = (0.0, 0.0, 0.0, 1.0)
    scale: Vec3 = (1.0, 1.0, 1.0)


@dataclass(slots=True)
class MaterialPhysical:
    """Contact and mass properties.

    Attributes:
        density: Mass density [kg/m^3].
        friction_static: Static Coulomb friction coefficient.
        friction_dynamic: Dynamic Coulomb friction coefficient.
        friction_rolling: Rolling friction coefficient.
        restitution: Coefficient of restitution.
    """

    density: float = 1000.0
    friction_static: float = 0.6
    friction_dynamic: float = 0.5
    friction_rolling: float = 0.0
    restitution: float = 0.0


@dataclass(slots=True)
class MaterialVisual:
    """Surface appearance."""

    color: Color = (0.7, 0.7, 0.7, 1.0)
    transparency: float = 0.0
    roughness: float = 0.5


# Compatibility name retained for v1 callers.
Material = MaterialPhysical


@dataclass(slots=True)
class ShapePrimitive:
    """One primitive in a rigid or compound collider."""

    kind: Literal["box", "sphere", "capsule"] = "box"
    size: Vec3 = (1.0, 1.0, 1.0)
    transform: Transform = field(default_factory=Transform)


@dataclass(slots=True)
class ObjectBase:
    """Properties shared by scene objects."""

    id: str
    transform: Transform = field(default_factory=Transform)
    physical_material: MaterialPhysical = field(default_factory=MaterialPhysical)
    visual_material: MaterialVisual = field(default_factory=MaterialVisual)
    motion: Literal["dynamic", "static", "kinematic"] = "dynamic"


@dataclass(slots=True)
class ObjectRigid(ObjectBase):
    """Rigid body composed of one or more collision primitives."""

    kind: Literal["rigid"] = "rigid"
    shape: Literal["box", "sphere", "capsule", "compound"] = "box"
    size: Vec3 = (1.0, 1.0, 1.0)
    shapes: list[ShapePrimitive] = field(default_factory=list)
    linear_velocity: Vec3 = (0.0, 0.0, 0.0)
    angular_velocity: Vec3 = (0.0, 0.0, 0.0)


@dataclass(slots=True)
class VertexSelector:
    """Select cloth vertices to pin."""

    kind: Literal["uv-corners", "edge", "indices"] = "indices"
    corners: list[Literal["bottom-left", "bottom-right", "top-left", "top-right"]] = field(default_factory=list)
    edge: Literal["top", "bottom", "left", "right"] | None = None
    indices: list[int] = field(default_factory=list)


@dataclass(slots=True)
class ObjectCloth(ObjectBase):
    """Rectangular cloth discretized into particles."""

    kind: Literal["cloth"] = "cloth"
    size: Vec2 = (1.0, 1.0)
    resolution: tuple[int, int] = (16, 16)
    thickness: float = 0.01
    surface_density: float = 0.2
    stretch_stiffness: float = 1000.0
    area_stiffness: float | None = None
    bend_stiffness: float = 1.0
    damping: float = 0.01
    stretch_damping: float | None = None
    bend_damping: float | None = None
    air_drag: float = 0.0
    collision_radius: float | None = None
    max_stretch_ratio: float = 1.2
    self_collision: bool = False
    pinned: list[VertexSelector] = field(default_factory=list)


@dataclass(slots=True)
class FluidEmitter:
    """Smoke or liquid emission volume."""

    position: Vec3 = (0.0, 0.0, 0.0)
    size: Vec3 = (0.25, 0.25, 0.25)
    start_time: float = 0.0
    end_time: float = 1.0
    density: float = 1.0
    velocity: Vec3 = (0.0, 0.0, 0.0)


@dataclass(slots=True)
class ObjectFluid(ObjectBase):
    """Smoke grid or APIC/FLIP free-surface liquid."""

    kind: Literal["fluid"] = "fluid"
    phase: Literal["smoke", "liquid"] = "liquid"
    size: Vec3 = (1.0, 1.0, 1.0)
    grid_resolution: tuple[int, int, int] = (32, 32, 32)
    particle_spacing: float = 0.05
    density: float = 1000.0
    viscosity: float = 0.001
    surface_tension: float = 0.072
    buoyancy: float = 1.0
    dissipation: float = 0.01
    flip_ratio: float = 0.95
    emitters: list[FluidEmitter] = field(default_factory=list)


@dataclass(slots=True)
class ObjectContainer(ObjectBase):
    """Compound container generated from a floor, four walls, and optional lid."""

    kind: Literal["container"] = "container"
    inner_size: Vec3 = (1.0, 1.0, 1.0)
    wall_thickness: float = 0.05
    transparent_shell: bool = True
    closed: bool = False


SceneObject: TypeAlias = ObjectRigid | ObjectCloth | ObjectFluid | ObjectContainer


@dataclass(slots=True)
class ConstraintFixedPoint:
    """Pin an object point or selected cloth vertices."""

    id: str
    object_id: str
    point: Vec3 = (0.0, 0.0, 0.0)
    selector: VertexSelector | None = None
    kind: Literal["fixed-point"] = "fixed-point"


@dataclass(slots=True)
class ConstraintDistance:
    """Maintain a distance [m] between points on two objects."""

    id: str
    object_a: str
    object_b: str
    point_a: Vec3 = (0.0, 0.0, 0.0)
    point_b: Vec3 = (0.0, 0.0, 0.0)
    distance: float = 1.0
    stiffness: float = 1.0
    kind: Literal["distance"] = "distance"


Constraint: TypeAlias = ConstraintFixedPoint | ConstraintDistance


@dataclass(slots=True)
class FieldUniform:
    """Uniform acceleration [m/s^2] or force [N]."""

    id: str
    vector: Vec3
    mode: Literal["acceleration", "force"] = "acceleration"
    object_ids: list[str] | None = None
    kind: Literal["uniform"] = "uniform"


@dataclass(slots=True)
class FieldRadial:
    """Radial acceleration [m/s^2] or force [N]."""

    id: str
    center: Vec3
    strength: float
    falloff: Literal["constant", "linear", "inverse-square"] = "inverse-square"
    mode: Literal["acceleration", "force"] = "force"
    object_ids: list[str] | None = None
    kind: Literal["radial"] = "radial"


Field: TypeAlias = FieldUniform | FieldRadial


@dataclass(slots=True)
class Keyframe:
    """A transform sample on a timeline."""

    time: float
    transform: Transform


@dataclass(slots=True)
class ActionTransform:
    """Animate a kinematic object's transform."""

    id: str
    object_id: str
    keyframes: list[Keyframe]
    kind: Literal["transform"] = "transform"


@dataclass(slots=True)
class ActionImpulse:
    """Apply an instantaneous linear impulse [N s]."""

    id: str
    object_id: str
    impulse: Vec3
    time: float = 0.0
    point: Vec3 | None = None
    kind: Literal["impulse"] = "impulse"


@dataclass(slots=True)
class ActionForce:
    """Apply a force [N] over a time interval [s]."""

    id: str
    object_id: str
    force: Vec3
    start_time: float = 0.0
    end_time: float = 1.0
    kind: Literal["force"] = "force"


@dataclass(slots=True)
class ActionEmit:
    """Enable a fluid emitter over a time interval [s]."""

    id: str
    object_id: str
    emitter_index: int = 0
    start_time: float = 0.0
    end_time: float = 1.0
    kind: Literal["emit"] = "emit"


Action: TypeAlias = ActionTransform | ActionImpulse | ActionForce | ActionEmit


@dataclass(slots=True)
class Camera:
    """Render camera using a look-at transform.

    Attributes:
        position: Camera position [m].
        target: Look-at target [m].
        up: Preferred image-up direction.
        field_of_view: Vertical field of view [deg].
        auto_frame: Whether the renderer may choose framing automatically.
    """

    position: Vec3 | None = None
    target: Vec3 | None = None
    up: Vec3 = (0.0, 0.0, 1.0)
    field_of_view: float = 45.0
    auto_frame: bool = True


@dataclass(slots=True)
class Light:
    """Directional or point light."""

    kind: Literal["directional", "point"] = "directional"
    position: Vec3 = (3.0, -3.0, 6.0)
    direction: Vec3 = (-0.5, 0.5, -1.0)
    color: Color = (1.0, 1.0, 1.0, 1.0)
    intensity: float = 3.0


@dataclass(slots=True)
class RenderSettings:
    """Off-screen animation settings."""

    resolution: tuple[int, int] = (1280, 720)
    fps: int = 30
    quality: Literal["preview", "medium", "final"] = "medium"
    camera: Camera = field(default_factory=Camera)
    lights: list[Light] = field(default_factory=lambda: [Light()])
    ground: bool = True


@dataclass(slots=True)
class RigidSolverSettings:
    """Rigid-body solver settings."""

    method: Literal["auto", "xpbd", "vbd"] = "auto"
    iterations: int = 10
    contact_margin: float = 1.0e-3
    contact_compliance: float = 0.0


@dataclass(slots=True)
class ClothSolverSettings:
    """Cloth solver settings."""

    method: Literal["auto", "xpbd", "vbd"] = "auto"
    iterations: int = 10
    strain_limit_iterations: int = 3
    enable_self_collision: bool = False


@dataclass(slots=True)
class FluidSolverSettings:
    """Grid and particle fluid solver settings."""

    liquid_method: Literal["apic"] = "apic"
    smoke_method: Literal["mac"] = "mac"
    pressure_iterations: int = 40
    cfl_number: float = 0.5


@dataclass(slots=True)
class CouplingSettings:
    """Settings shared by multiphysics interface solvers."""

    mode: Literal["loose", "strong"] = "loose"
    iterations: int = 1
    relaxation: float = 0.7
    interface_tolerance: float = 1.0e-3
    rigid_cloth: bool = True
    rigid_fluid: bool = True
    cloth_fluid: bool = True
    boundary_friction: float = 0.0
    cloth_fluid_drag: float = 1.0
    cloth_permeability: float = 0.0
    smoke_drag_density: float = 1.225
    smoke_drag_coefficient: float = 0.0


@dataclass(slots=True)
class SimulationSettings:
    """Scene simulation settings; all physical values use SI units."""

    fps: int = 60
    substeps: int = 8
    duration: float = 5.0
    gravity: Vec3 = (0.0, 0.0, -9.81)
    max_particles: int = 250_000
    rigid: RigidSolverSettings = field(default_factory=RigidSolverSettings)
    cloth: ClothSolverSettings = field(default_factory=ClothSolverSettings)
    fluid: FluidSolverSettings = field(default_factory=FluidSolverSettings)
    coupling: CouplingSettings = field(default_factory=CouplingSettings)

    @property
    def solver(self) -> Literal["auto", "xpbd", "vbd"]:
        """Return the legacy solid-solver selection."""
        return self.rigid.method

    @solver.setter
    def solver(self, value: Literal["auto", "xpbd", "vbd"]) -> None:
        self.rigid.method = value
        self.cloth.method = value

    @property
    def solver_iterations(self) -> int:
        """Return the legacy shared solid-solver iteration count."""
        return self.rigid.iterations

    @solver_iterations.setter
    def solver_iterations(self, value: int) -> None:
        self.rigid.iterations = value
        self.cloth.iterations = value


@dataclass(slots=True)
class Scene:
    """Backend-neutral animation scene schema version 3."""

    name: str
    objects: dict[str, SceneObject] = field(default_factory=dict)
    constraints: dict[str, Constraint] = field(default_factory=dict)
    fields: dict[str, Field] = field(default_factory=dict)
    actions: dict[str, Action] = field(default_factory=dict)
    settings: SimulationSettings = field(default_factory=SimulationSettings)
    render: RenderSettings = field(default_factory=RenderSettings)
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable scene representation."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Scene:
        """Parse and, when necessary, explicitly migrate a scene mapping."""
        values = migrate_scene_dict(data)
        return cls(
            name=values["name"],
            objects={key: _object_from_dict(item) for key, item in values.get("objects", {}).items()},
            constraints={key: _constraint_from_dict(item) for key, item in values.get("constraints", {}).items()},
            fields={key: _field_from_dict(item) for key, item in values.get("fields", {}).items()},
            actions={key: _action_from_dict(item) for key, item in values.get("actions", {}).items()},
            settings=_simulation_settings_from_dict(values.get("settings", {})),
            render=_render_from_dict(values.get("render", {})),
            metadata=values.get("metadata", {}),
        )


def migrate_scene_dict(data: dict[str, Any]) -> dict[str, Any]:
    """Return a schema-v3 copy of a v1, v2, or v3 scene mapping."""
    version = data.get("schema_version", 1)
    if version == SCHEMA_VERSION:
        return dict(data)
    if version not in {1, 2}:
        raise ValueError(f"Unsupported scene schema version: {version}")
    migrated = dict(data)
    objects = {}
    for key, original in data.get("objects", {}).items():
        item = dict(original)
        material = item.pop("material", {}) if version == 1 else {}
        if material:
            friction = material.get("friction", 0.5)
            item["physical_material"] = {
                "density": material.get("density", 1000.0),
                "friction_static": friction,
                "friction_dynamic": friction,
                "restitution": material.get("restitution", 0.0),
            }
            item["visual_material"] = {"color": material.get("color", (0.7, 0.7, 0.7, 1.0))}
        if version == 1 and "dynamic" in item:
            item["motion"] = "dynamic" if item.pop("dynamic") else "static"
        if version == 1 and item.get("kind") == "fluid":
            item.setdefault("phase", "liquid")
        objects[key] = item
    settings = dict(data.get("settings", {}))
    legacy_solver = settings.pop("solver", "auto")
    legacy_iterations = settings.pop("solver_iterations", 10)
    if legacy_solver == "mpm":
        legacy_solver = "apic"
    solid_method = legacy_solver if legacy_solver in {"auto", "xpbd", "vbd"} else "auto"
    rigid = dict(settings.get("rigid", {}))
    rigid.setdefault("method", solid_method)
    rigid.setdefault("iterations", legacy_iterations)
    cloth = dict(settings.get("cloth", {}))
    cloth.setdefault("method", solid_method)
    cloth.setdefault("iterations", legacy_iterations)
    # The v3 rigid-cloth solver applies true bidirectional interface impulses;
    # two extra safety projections preserve v2 cloth-folding quality during migration.
    cloth.setdefault("strain_limit_iterations", 5)
    settings["rigid"] = rigid
    settings["cloth"] = cloth
    settings.setdefault("fluid", {})
    settings.setdefault("coupling", {})
    migrated.update(objects=objects, settings=settings, schema_version=SCHEMA_VERSION)
    migrated.setdefault("actions", {})
    migrated.setdefault("render", {})
    return migrated


def _simulation_settings_from_dict(data: dict[str, Any]) -> SimulationSettings:
    values = dict(data)
    values["rigid"] = RigidSolverSettings(**values.get("rigid", {}))
    values["cloth"] = ClothSolverSettings(**values.get("cloth", {}))
    values["fluid"] = FluidSolverSettings(**values.get("fluid", {}))
    values["coupling"] = CouplingSettings(**values.get("coupling", {}))
    return SimulationSettings(**values)


def _object_from_dict(data: dict[str, Any]) -> SceneObject:
    values = dict(data)
    kind = values.pop("kind")
    values["transform"] = Transform(**values.get("transform", {}))
    # Accept v1 item dictionaries passed directly to add_object.
    if "material" in values:
        material = values.pop("material")
        friction = material.get("friction", 0.5)
        values["physical_material"] = {
            "density": material.get("density", 1000.0),
            "friction_static": friction,
            "friction_dynamic": friction,
            "restitution": material.get("restitution", 0.0),
        }
        values["visual_material"] = {"color": material.get("color", (0.7, 0.7, 0.7, 1.0))}
    if "dynamic" in values:
        values["motion"] = "dynamic" if values.pop("dynamic") else "static"
    values["physical_material"] = MaterialPhysical(**values.get("physical_material", {}))
    values["visual_material"] = MaterialVisual(**values.get("visual_material", {}))
    if kind == "rigid":
        values["shapes"] = [_shape_from_dict(item) for item in values.get("shapes", [])]
    elif kind == "cloth":
        values["pinned"] = [VertexSelector(**item) for item in values.get("pinned", [])]
    elif kind == "fluid":
        values["emitters"] = [FluidEmitter(**item) for item in values.get("emitters", [])]
    object_types = {"rigid": ObjectRigid, "cloth": ObjectCloth, "fluid": ObjectFluid, "container": ObjectContainer}
    try:
        return object_types[kind](**values)
    except KeyError as error:
        raise ValueError(f"Unknown object kind: {kind}") from error


def _shape_from_dict(data: dict[str, Any]) -> ShapePrimitive:
    values = dict(data)
    values["transform"] = Transform(**values.get("transform", {}))
    return ShapePrimitive(**values)


def _constraint_from_dict(data: dict[str, Any]) -> Constraint:
    values = dict(data)
    kind = values.pop("kind")
    if values.get("selector") is not None:
        values["selector"] = VertexSelector(**values["selector"])
    types = {"fixed-point": ConstraintFixedPoint, "distance": ConstraintDistance}
    try:
        return types[kind](**values)
    except KeyError as error:
        raise ValueError(f"Unknown constraint kind: {kind}") from error


def _field_from_dict(data: dict[str, Any]) -> Field:
    values = dict(data)
    kind = values.pop("kind")
    types = {"uniform": FieldUniform, "radial": FieldRadial}
    try:
        return types[kind](**values)
    except KeyError as error:
        raise ValueError(f"Unknown field kind: {kind}") from error


def _action_from_dict(data: dict[str, Any]) -> Action:
    values = dict(data)
    kind = values.pop("kind")
    if kind == "transform":
        values["keyframes"] = [
            Keyframe(time=item["time"], transform=Transform(**item["transform"])) for item in values["keyframes"]
        ]
    types = {"transform": ActionTransform, "impulse": ActionImpulse, "force": ActionForce, "emit": ActionEmit}
    try:
        return types[kind](**values)
    except KeyError as error:
        raise ValueError(f"Unknown action kind: {kind}") from error


def _render_from_dict(data: dict[str, Any]) -> RenderSettings:
    values = dict(data)
    values["camera"] = Camera(**values.get("camera", {}))
    values["lights"] = [Light(**item) for item in values.get("lights", [{}])]
    return RenderSettings(**values)
