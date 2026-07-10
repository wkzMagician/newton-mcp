# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Strongly typed MCP wire models for Newton scene schema version 2."""

from __future__ import annotations

from typing import Annotated, Literal, TypeAlias

from pydantic import BaseModel, ConfigDict, Field, JsonValue

Vec2: TypeAlias = tuple[float, float]
Vec3: TypeAlias = tuple[float, float, float]
Quat: TypeAlias = tuple[float, float, float, float]
Color: TypeAlias = tuple[float, float, float, float]


class DTOBase(BaseModel):
    """Base class that rejects misspelled or unsupported wire fields."""

    model_config = ConfigDict(extra="forbid")


class TransformDTO(DTOBase):
    """World transform using SI units."""

    position: Vec3 = Field((0.0, 0.0, 0.0), description="World translation [m].")
    rotation: Quat = Field((0.0, 0.0, 0.0, 1.0), description="Quaternion in (x, y, z, w) order.")
    scale: Vec3 = Field((1.0, 1.0, 1.0), description="Positive dimensionless local scale.")


class MaterialPhysicalDTO(DTOBase):
    """Contact and mass properties."""

    density: float = Field(1000.0, gt=0.0, description="Mass density [kg/m^3].")
    friction_static: float = Field(0.6, ge=0.0, description="Static Coulomb friction coefficient.")
    friction_dynamic: float = Field(0.5, ge=0.0, description="Dynamic Coulomb friction coefficient.")
    friction_rolling: float = Field(0.0, ge=0.0, description="Rolling friction coefficient.")
    restitution: float = Field(0.0, ge=0.0, le=1.0, description="Coefficient of restitution.")


class MaterialVisualDTO(DTOBase):
    """Surface appearance."""

    color: Color = Field((0.7, 0.7, 0.7, 1.0), description="Linear RGBA color.")
    transparency: float = Field(0.0, ge=0.0, le=1.0, description="Surface transparency.")
    roughness: float = Field(0.5, ge=0.0, le=1.0, description="Surface roughness.")


class ShapePrimitiveDTO(DTOBase):
    """Primitive belonging to a compound rigid body."""

    kind: Literal["box", "sphere", "capsule"]
    size: Vec3 = Field((1.0, 1.0, 1.0), description="Primitive dimensions [m].")
    transform: TransformDTO = Field(default_factory=TransformDTO)


class VertexSelectorDTO(DTOBase):
    """Cloth vertex selection."""

    kind: Literal["uv-corners", "edge", "indices"]
    corners: list[Literal["bottom-left", "bottom-right", "top-left", "top-right"]] = Field(default_factory=list)
    edge: Literal["top", "bottom", "left", "right"] | None = None
    indices: list[int] = Field(default_factory=list)


class FluidEmitterDTO(DTOBase):
    """Smoke or liquid emission volume."""

    position: Vec3 = Field((0.0, 0.0, 0.0), description="Emitter center [m].")
    size: Vec3 = Field((0.25, 0.25, 0.25), description="Emitter dimensions [m].")
    start_time: float = Field(0.0, ge=0.0, description="Emission start time [s].")
    end_time: float = Field(1.0, ge=0.0, description="Emission end time [s].")
    density: float = Field(1.0, ge=0.0, description="Injected density.")
    velocity: Vec3 = Field((0.0, 0.0, 0.0), description="Initial velocity [m/s].")


class ObjectBaseDTO(DTOBase):
    """Fields shared by all object creation DTOs."""

    id: str = Field(min_length=1, description="Unique object identifier.")
    transform: TransformDTO = Field(default_factory=TransformDTO)
    physical_material: MaterialPhysicalDTO = Field(default_factory=MaterialPhysicalDTO)
    visual_material: MaterialVisualDTO = Field(default_factory=MaterialVisualDTO)
    motion: Literal["dynamic", "static", "kinematic"] = "dynamic"


class ObjectRigidDTO(ObjectBaseDTO):
    """Rigid body creation parameters."""

    kind: Literal["rigid"]
    shape: Literal["box", "sphere", "capsule", "compound"] = "box"
    size: Vec3 = Field((1.0, 1.0, 1.0), description="Primitive dimensions [m].")
    shapes: list[ShapePrimitiveDTO] = Field(default_factory=list)
    linear_velocity: Vec3 = Field((0.0, 0.0, 0.0), description="Initial linear velocity [m/s].")
    angular_velocity: Vec3 = Field((0.0, 0.0, 0.0), description="Initial angular velocity [rad/s].")


class ObjectClothDTO(ObjectBaseDTO):
    """Rectangular cloth creation parameters."""

    kind: Literal["cloth"]
    size: Vec2 = Field((1.0, 1.0), description="Cloth dimensions [m].")
    resolution: tuple[int, int] = Field((16, 16), description="Vertex counts along both cloth axes.")
    thickness: float = Field(0.01, gt=0.0, description="Collision thickness [m].")
    surface_density: float = Field(0.2, gt=0.0, description="Surface mass density [kg/m^2].")
    stretch_stiffness: float = Field(1000.0, ge=0.0, description="Stretch stiffness [N/m].")
    bend_stiffness: float = Field(1.0, ge=0.0, description="Bending stiffness [N m].")
    damping: float = Field(0.01, ge=0.0, description="Damping coefficient.")
    air_drag: float = Field(0.0, ge=0.0, description="Air drag coefficient.")
    self_collision: bool = False
    pinned: list[VertexSelectorDTO] = Field(default_factory=list)


class ObjectFluidDTO(ObjectBaseDTO):
    """Smoke grid or APIC/FLIP liquid creation parameters."""

    kind: Literal["fluid"]
    phase: Literal["smoke", "liquid"] = "liquid"
    size: Vec3 = Field((1.0, 1.0, 1.0), description="Initial fluid volume [m].")
    grid_resolution: tuple[int, int, int] = Field((32, 32, 32), description="MAC grid cell counts.")
    particle_spacing: float = Field(0.05, gt=0.0, description="Liquid particle spacing [m].")
    density: float = Field(1000.0, gt=0.0, description="Liquid mass density [kg/m^3].")
    viscosity: float = Field(0.001, ge=0.0, description="Dynamic viscosity [Pa s].")
    surface_tension: float = Field(0.072, ge=0.0, description="Surface tension [N/m].")
    buoyancy: float = Field(1.0, description="Smoke buoyancy acceleration scale [m/s^2].")
    dissipation: float = Field(0.01, ge=0.0, le=1.0, description="Smoke density dissipation per step.")
    flip_ratio: float = Field(0.95, ge=0.0, le=1.0, description="FLIP contribution to the PIC/FLIP blend.")
    emitters: list[FluidEmitterDTO] = Field(default_factory=list)


class ObjectContainerDTO(ObjectBaseDTO):
    """Open-top compound container creation parameters."""

    kind: Literal["container"]
    inner_size: Vec3 = Field((1.0, 1.0, 1.0), description="Interior dimensions [m].")
    wall_thickness: float = Field(0.05, gt=0.0, description="Wall thickness [m].")
    transparent_shell: bool = True


ObjectSpecDTO = Annotated[
    ObjectRigidDTO | ObjectClothDTO | ObjectFluidDTO | ObjectContainerDTO,
    Field(discriminator="kind"),
]


class ConstraintFixedPointDTO(DTOBase):
    """Fixed-point constraint creation parameters."""

    id: str = Field(min_length=1)
    object_id: str = Field(min_length=1)
    point: Vec3 = Field((0.0, 0.0, 0.0), description="Pinned world-space point [m].")
    selector: VertexSelectorDTO | None = None
    kind: Literal["fixed-point"]


class ConstraintDistanceDTO(DTOBase):
    """Distance constraint creation parameters."""

    id: str = Field(min_length=1)
    object_a: str = Field(min_length=1)
    object_b: str = Field(min_length=1)
    point_a: Vec3 = Field((0.0, 0.0, 0.0), description="First local point [m].")
    point_b: Vec3 = Field((0.0, 0.0, 0.0), description="Second local point [m].")
    distance: float = Field(1.0, ge=0.0, description="Target distance [m].")
    stiffness: float = Field(1.0, ge=0.0, description="Constraint stiffness.")
    kind: Literal["distance"]


ConstraintSpecDTO = Annotated[ConstraintFixedPointDTO | ConstraintDistanceDTO, Field(discriminator="kind")]


class FieldUniformDTO(DTOBase):
    """Uniform force or acceleration field creation parameters."""

    id: str = Field(min_length=1)
    vector: Vec3 = Field(description="Acceleration [m/s^2] or force [N].")
    mode: Literal["acceleration", "force"] = "acceleration"
    object_ids: list[str] | None = None
    kind: Literal["uniform"]


class FieldRadialDTO(DTOBase):
    """Radial force or acceleration field creation parameters."""

    id: str = Field(min_length=1)
    center: Vec3 = Field(description="Field center [m].")
    strength: float = Field(description="Acceleration [m/s^2] or force [N].")
    falloff: Literal["constant", "linear", "inverse-square"] = "inverse-square"
    mode: Literal["acceleration", "force"] = "force"
    object_ids: list[str] | None = None
    kind: Literal["radial"]


FieldSpecDTO = Annotated[FieldUniformDTO | FieldRadialDTO, Field(discriminator="kind")]


class KeyframeDTO(DTOBase):
    """Transform sample on a timeline."""

    time: float = Field(ge=0.0, description="Keyframe time [s].")
    transform: TransformDTO


class ActionTransformDTO(DTOBase):
    """Kinematic transform action creation parameters."""

    id: str = Field(min_length=1)
    object_id: str = Field(min_length=1)
    keyframes: list[KeyframeDTO] = Field(min_length=1)
    kind: Literal["transform"]


class ActionImpulseDTO(DTOBase):
    """Instantaneous impulse action creation parameters."""

    id: str = Field(min_length=1)
    object_id: str = Field(min_length=1)
    impulse: Vec3 = Field(description="Linear impulse [N s].")
    time: float = Field(0.0, ge=0.0, description="Application time [s].")
    point: Vec3 | None = Field(None, description="Optional world-space application point [m].")
    kind: Literal["impulse"]


class ActionForceDTO(DTOBase):
    """Continuous force action creation parameters."""

    id: str = Field(min_length=1)
    object_id: str = Field(min_length=1)
    force: Vec3 = Field(description="Applied force [N].")
    start_time: float = Field(0.0, ge=0.0, description="Start time [s].")
    end_time: float = Field(1.0, ge=0.0, description="End time [s].")
    kind: Literal["force"]


class ActionEmitDTO(DTOBase):
    """Fluid emission action creation parameters."""

    id: str = Field(min_length=1)
    object_id: str = Field(min_length=1)
    emitter_index: int = Field(0, ge=0)
    start_time: float = Field(0.0, ge=0.0, description="Start time [s].")
    end_time: float = Field(1.0, ge=0.0, description="End time [s].")
    kind: Literal["emit"]


ActionSpecDTO = Annotated[
    ActionTransformDTO | ActionImpulseDTO | ActionForceDTO | ActionEmitDTO,
    Field(discriminator="kind"),
]


class ObjectPatchBaseDTO(DTOBase):
    """Fields shared by object update DTOs."""

    id: str | None = Field(None, description="If supplied, must equal the existing id.")
    transform: TransformDTO | None = None
    physical_material: MaterialPhysicalDTO | None = None
    visual_material: MaterialVisualDTO | None = None
    motion: Literal["dynamic", "static", "kinematic"] | None = None


class ObjectRigidPatchDTO(ObjectPatchBaseDTO):
    """Rigid body update fields."""

    kind: Literal["rigid"]
    shape: Literal["box", "sphere", "capsule", "compound"] | None = None
    size: Vec3 | None = None
    shapes: list[ShapePrimitiveDTO] | None = None
    linear_velocity: Vec3 | None = None
    angular_velocity: Vec3 | None = None


class ObjectClothPatchDTO(ObjectPatchBaseDTO):
    """Cloth update fields."""

    kind: Literal["cloth"]
    size: Vec2 | None = None
    resolution: tuple[int, int] | None = None
    thickness: float | None = Field(None, gt=0.0)
    surface_density: float | None = Field(None, gt=0.0)
    stretch_stiffness: float | None = Field(None, ge=0.0)
    bend_stiffness: float | None = Field(None, ge=0.0)
    damping: float | None = Field(None, ge=0.0)
    air_drag: float | None = Field(None, ge=0.0)
    self_collision: bool | None = None
    pinned: list[VertexSelectorDTO] | None = None


class ObjectFluidPatchDTO(ObjectPatchBaseDTO):
    """Fluid update fields."""

    kind: Literal["fluid"]
    phase: Literal["smoke", "liquid"] | None = None
    size: Vec3 | None = None
    grid_resolution: tuple[int, int, int] | None = None
    particle_spacing: float | None = Field(None, gt=0.0)
    density: float | None = Field(None, gt=0.0)
    viscosity: float | None = Field(None, ge=0.0)
    surface_tension: float | None = Field(None, ge=0.0)
    buoyancy: float | None = None
    dissipation: float | None = Field(None, ge=0.0, le=1.0)
    flip_ratio: float | None = Field(None, ge=0.0, le=1.0)
    emitters: list[FluidEmitterDTO] | None = None


class ObjectContainerPatchDTO(ObjectPatchBaseDTO):
    """Container update fields."""

    kind: Literal["container"]
    inner_size: Vec3 | None = None
    wall_thickness: float | None = Field(None, gt=0.0)
    transparent_shell: bool | None = None


ObjectPatchDTO = Annotated[
    ObjectRigidPatchDTO | ObjectClothPatchDTO | ObjectFluidPatchDTO | ObjectContainerPatchDTO,
    Field(discriminator="kind"),
]


class ConstraintFixedPointPatchDTO(DTOBase):
    """Fixed-point constraint update fields."""

    kind: Literal["fixed-point"]
    id: str | None = None
    object_id: str | None = None
    point: Vec3 | None = None
    selector: VertexSelectorDTO | None = None


class ConstraintDistancePatchDTO(DTOBase):
    """Distance constraint update fields."""

    kind: Literal["distance"]
    id: str | None = None
    object_a: str | None = None
    object_b: str | None = None
    point_a: Vec3 | None = None
    point_b: Vec3 | None = None
    distance: float | None = Field(None, ge=0.0)
    stiffness: float | None = Field(None, ge=0.0)


ConstraintPatchDTO = Annotated[
    ConstraintFixedPointPatchDTO | ConstraintDistancePatchDTO,
    Field(discriminator="kind"),
]


class FieldUniformPatchDTO(DTOBase):
    """Uniform field update fields."""

    kind: Literal["uniform"]
    id: str | None = None
    vector: Vec3 | None = None
    mode: Literal["acceleration", "force"] | None = None
    object_ids: list[str] | None = None


class FieldRadialPatchDTO(DTOBase):
    """Radial field update fields."""

    kind: Literal["radial"]
    id: str | None = None
    center: Vec3 | None = None
    strength: float | None = None
    falloff: Literal["constant", "linear", "inverse-square"] | None = None
    mode: Literal["acceleration", "force"] | None = None
    object_ids: list[str] | None = None


FieldPatchDTO = Annotated[FieldUniformPatchDTO | FieldRadialPatchDTO, Field(discriminator="kind")]


class ActionTransformPatchDTO(DTOBase):
    """Transform action update fields."""

    kind: Literal["transform"]
    id: str | None = None
    object_id: str | None = None
    keyframes: list[KeyframeDTO] | None = None


class ActionImpulsePatchDTO(DTOBase):
    """Impulse action update fields."""

    kind: Literal["impulse"]
    id: str | None = None
    object_id: str | None = None
    impulse: Vec3 | None = None
    time: float | None = Field(None, ge=0.0)
    point: Vec3 | None = None


class ActionForcePatchDTO(DTOBase):
    """Force action update fields."""

    kind: Literal["force"]
    id: str | None = None
    object_id: str | None = None
    force: Vec3 | None = None
    start_time: float | None = Field(None, ge=0.0)
    end_time: float | None = Field(None, ge=0.0)


class ActionEmitPatchDTO(DTOBase):
    """Emission action update fields."""

    kind: Literal["emit"]
    id: str | None = None
    object_id: str | None = None
    emitter_index: int | None = Field(None, ge=0)
    start_time: float | None = Field(None, ge=0.0)
    end_time: float | None = Field(None, ge=0.0)


ActionPatchDTO = Annotated[
    ActionTransformPatchDTO | ActionImpulsePatchDTO | ActionForcePatchDTO | ActionEmitPatchDTO,
    Field(discriminator="kind"),
]


class CameraDTO(DTOBase):
    """Render camera parameters."""

    position: Vec3 | None = Field(None, description="Camera position [m].")
    target: Vec3 | None = Field(None, description="Look-at target [m].")
    field_of_view: float = Field(45.0, gt=0.0, lt=180.0, description="Vertical field of view [deg].")
    auto_frame: bool = True


class LightDTO(DTOBase):
    """Scene light parameters."""

    kind: Literal["directional", "point"] = "directional"
    position: Vec3 = Field((3.0, -3.0, 6.0), description="Light position [m].")
    direction: Vec3 = (-0.5, 0.5, -1.0)
    color: Color = (1.0, 1.0, 1.0, 1.0)
    intensity: float = Field(3.0, ge=0.0)


class SimulationSettingsDTO(DTOBase):
    """Simulation settings."""

    fps: int = Field(60, gt=0)
    substeps: int = Field(8, gt=0)
    duration: float = Field(5.0, gt=0.0, description="Animation duration [s].")
    gravity: Vec3 = Field((0.0, 0.0, -9.81), description="Gravity acceleration [m/s^2].")
    solver: Literal["auto", "xpbd", "vbd", "smoke", "apic"] = "auto"
    max_particles: int = Field(250_000, gt=0)


class RenderSettingsDTO(DTOBase):
    """Off-screen render settings."""

    resolution: tuple[int, int] = (1280, 720)
    fps: int = Field(30, gt=0)
    quality: Literal["preview", "medium", "final"] = "medium"
    camera: CameraDTO = Field(default_factory=CameraDTO)
    lights: list[LightDTO] = Field(default_factory=lambda: [LightDTO()])
    ground: bool = True


class SimulationSettingsPatchDTO(DTOBase):
    """Partial simulation settings."""

    fps: int | None = Field(None, gt=0)
    substeps: int | None = Field(None, gt=0)
    duration: float | None = Field(None, gt=0.0)
    gravity: Vec3 | None = None
    solver: Literal["auto", "xpbd", "vbd", "smoke", "apic"] | None = None
    max_particles: int | None = Field(None, gt=0)


class RenderSettingsPatchDTO(DTOBase):
    """Partial render settings."""

    resolution: tuple[int, int] | None = None
    fps: int | None = Field(None, gt=0)
    quality: Literal["preview", "medium", "final"] | None = None
    camera: CameraDTO | None = None
    lights: list[LightDTO] | None = None
    ground: bool | None = None


class SceneDTO(DTOBase):
    """Complete Newton scene IR schema version 2."""

    name: str = Field(min_length=1)
    objects: dict[str, ObjectSpecDTO] = Field(default_factory=dict)
    constraints: dict[str, ConstraintSpecDTO] = Field(default_factory=dict)
    fields: dict[str, FieldSpecDTO] = Field(default_factory=dict)
    actions: dict[str, ActionSpecDTO] = Field(default_factory=dict)
    settings: SimulationSettingsDTO = Field(default_factory=SimulationSettingsDTO)
    render: RenderSettingsDTO = Field(default_factory=RenderSettingsDTO)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)
    schema_version: Literal[2] = 2


class ScenePatchDTO(DTOBase):
    """Typed recursive merge patch for a scene."""

    objects: dict[str, ObjectPatchDTO | None] | None = None
    constraints: dict[str, ConstraintPatchDTO | None] | None = None
    fields: dict[str, FieldPatchDTO | None] | None = None
    actions: dict[str, ActionPatchDTO | None] | None = None
    settings: SimulationSettingsPatchDTO | None = None
    render: RenderSettingsPatchDTO | None = None
    metadata: dict[str, JsonValue] | None = None
