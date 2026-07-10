# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Serializable scene description used by agents and simulation backends."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Literal, TypeAlias

Vec3: TypeAlias = tuple[float, float, float]
Quat: TypeAlias = tuple[float, float, float, float]


@dataclass(slots=True)
class Transform:
    """World transform of an object.

    Attributes:
        position: Translation [m].
        rotation: Quaternion in ``(x, y, z, w)`` order.
        scale: Dimensionless local scale.
    """

    position: Vec3 = (0.0, 0.0, 0.0)
    rotation: Quat = (0.0, 0.0, 0.0, 1.0)
    scale: Vec3 = (1.0, 1.0, 1.0)


@dataclass(slots=True)
class Material:
    """Physical and visual material.

    Attributes:
        density: Mass density [kg/m^3].
        friction: Dimensionless Coulomb friction coefficient.
        restitution: Dimensionless coefficient of restitution.
        color: Linear RGBA color.
    """

    density: float = 1000.0
    friction: float = 0.5
    restitution: float = 0.0
    color: tuple[float, float, float, float] = (0.7, 0.7, 0.7, 1.0)


@dataclass(slots=True)
class ObjectBase:
    """Properties shared by all scene objects."""

    id: str
    transform: Transform = field(default_factory=Transform)
    material: Material = field(default_factory=Material)
    dynamic: bool = True


@dataclass(slots=True)
class ObjectRigid(ObjectBase):
    """Rigid object represented by a primitive collision shape."""

    kind: Literal["rigid"] = "rigid"
    shape: Literal["box", "sphere", "capsule"] = "box"
    size: Vec3 = (1.0, 1.0, 1.0)


@dataclass(slots=True)
class ObjectCloth(ObjectBase):
    """Rectangular cloth discretized into particles."""

    kind: Literal["cloth"] = "cloth"
    size: tuple[float, float] = (1.0, 1.0)
    resolution: tuple[int, int] = (16, 16)
    thickness: float = 0.01


@dataclass(slots=True)
class ObjectFluid(ObjectBase):
    """Particle fluid initialized inside an axis-aligned volume."""

    kind: Literal["fluid"] = "fluid"
    size: Vec3 = (1.0, 1.0, 1.0)
    particle_spacing: float = 0.05


SceneObject: TypeAlias = ObjectRigid | ObjectCloth | ObjectFluid


@dataclass(slots=True)
class ConstraintFixedPoint:
    """Pin one object point to a world-space position."""

    id: str
    object_id: str
    point: Vec3
    kind: Literal["fixed-point"] = "fixed-point"


@dataclass(slots=True)
class ConstraintDistance:
    """Maintain a distance between points on two objects."""

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
    """Uniform acceleration or force field."""

    id: str
    vector: Vec3
    mode: Literal["acceleration", "force"] = "acceleration"
    object_ids: list[str] | None = None
    kind: Literal["uniform"] = "uniform"


@dataclass(slots=True)
class FieldRadial:
    """Radial acceleration or force field centered at a point."""

    id: str
    center: Vec3
    strength: float
    falloff: Literal["constant", "linear", "inverse-square"] = "inverse-square"
    mode: Literal["acceleration", "force"] = "force"
    object_ids: list[str] | None = None
    kind: Literal["radial"] = "radial"


Field: TypeAlias = FieldUniform | FieldRadial


@dataclass(slots=True)
class SimulationSettings:
    """Scene simulation settings."""

    fps: int = 60
    substeps: int = 8
    duration: float = 5.0
    gravity: Vec3 = (0.0, 0.0, -9.81)
    solver: Literal["xpbd", "vbd", "mpm"] = "xpbd"


@dataclass(slots=True)
class Scene:
    """Backend-neutral animation scene."""

    name: str
    objects: dict[str, SceneObject] = field(default_factory=dict)
    constraints: dict[str, Constraint] = field(default_factory=dict)
    fields: dict[str, Field] = field(default_factory=dict)
    settings: SimulationSettings = field(default_factory=SimulationSettings)
    metadata: dict[str, Any] = field(default_factory=dict)
    schema_version: int = 1

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable scene representation."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Scene:
        """Create a scene from its JSON representation.

        Args:
            data: Scene mapping using schema version 1.

        Returns:
            Parsed scene.
        """
        objects = {item_id: _object_from_dict(item) for item_id, item in data.get("objects", {}).items()}
        constraints = {item_id: _constraint_from_dict(item) for item_id, item in data.get("constraints", {}).items()}
        fields = {item_id: _field_from_dict(item) for item_id, item in data.get("fields", {}).items()}
        return cls(
            name=data["name"],
            objects=objects,
            constraints=constraints,
            fields=fields,
            settings=SimulationSettings(**data.get("settings", {})),
            metadata=data.get("metadata", {}),
            schema_version=data.get("schema_version", 1),
        )


def _object_from_dict(data: dict[str, Any]) -> SceneObject:
    values = dict(data)
    kind = values.pop("kind")
    values["transform"] = Transform(**values.get("transform", {}))
    values["material"] = Material(**values.get("material", {}))
    object_types = {"rigid": ObjectRigid, "cloth": ObjectCloth, "fluid": ObjectFluid}
    try:
        return object_types[kind](**values)
    except KeyError as error:
        raise ValueError(f"Unknown object kind: {kind}") from error


def _constraint_from_dict(data: dict[str, Any]) -> Constraint:
    values = dict(data)
    kind = values.pop("kind")
    constraint_types = {"fixed-point": ConstraintFixedPoint, "distance": ConstraintDistance}
    try:
        return constraint_types[kind](**values)
    except KeyError as error:
        raise ValueError(f"Unknown constraint kind: {kind}") from error


def _field_from_dict(data: dict[str, Any]) -> Field:
    values = dict(data)
    kind = values.pop("kind")
    field_types = {"uniform": FieldUniform, "radial": FieldRadial}
    try:
        return field_types[kind](**values)
    except KeyError as error:
        raise ValueError(f"Unknown field kind: {kind}") from error
