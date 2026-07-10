# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import tempfile
import unittest
from dataclasses import fields
from pathlib import Path
from unittest.mock import patch

from ca_mcp_server.dto import (
    ObjectClothDTO,
    ObjectContainerDTO,
    ObjectFluidDTO,
    ObjectRigidDTO,
    ObjectSpecDTO,
    SceneDTO,
)
from ca_mcp_server.server import mcp
from pydantic import TypeAdapter, ValidationError

from ca_framework.mcp import SceneTools
from ca_framework.scene import SceneExecutorLocal, SceneStore
from ca_framework.scene.model import ObjectCloth, ObjectContainer, ObjectFluid, ObjectRigid


def _tool_schemas() -> dict[str, dict]:
    async def get_tools():
        return {tool.name: tool.inputSchema for tool in await mcp.list_tools()}

    return asyncio.run(get_tools())


def _has_unrestricted_object(value) -> bool:
    if isinstance(value, dict):
        return value.get("additionalProperties") is True or any(
            _has_unrestricted_object(item) for item in value.values()
        )
    if isinstance(value, list):
        return any(_has_unrestricted_object(item) for item in value)
    return False


class TestMcpSchema(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.schemas = _tool_schemas()

    def test_add_object_exposes_discriminated_complete_union(self):
        schema = self.schemas["add_object"]
        spec = schema["properties"]["spec"]

        self.assertEqual(spec["discriminator"]["propertyName"], "kind")
        self.assertEqual(set(spec["discriminator"]["mapping"]), {"rigid", "cloth", "fluid", "container"})
        self.assertIn("shape", schema["$defs"]["ObjectRigidDTO"]["properties"])
        self.assertIn("pinned", schema["$defs"]["ObjectClothDTO"]["properties"])
        self.assertIn("emitters", schema["$defs"]["ObjectFluidDTO"]["properties"])
        self.assertIn("wall_thickness", schema["$defs"]["ObjectContainerDTO"]["properties"])

    def test_all_add_tools_are_discriminated(self):
        for name in ("add_object", "add_constraint", "add_field", "add_action"):
            with self.subTest(name=name):
                spec = self.schemas[name]["properties"]["spec"]
                self.assertIn("discriminator", spec)
                self.assertIn("oneOf", spec)

    def test_structured_tool_arguments_have_no_unrestricted_objects(self):
        names = (
            "add_object",
            "add_constraint",
            "add_field",
            "add_action",
            "apply_scene_patch",
            "update_object",
            "update_constraint",
            "update_field",
            "update_action",
        )
        for name in names:
            with self.subTest(name=name):
                self.assertFalse(_has_unrestricted_object(self.schemas[name]))

    def test_update_item_is_replaced_by_typed_tools(self):
        self.assertNotIn("update_item", self.schemas)
        for name in ("update_object", "update_constraint", "update_field", "update_action"):
            self.assertIn(name, self.schemas)
            self.assertIn("discriminator", self.schemas[name]["properties"]["patch"])

    def test_scene_patch_and_complete_scene_are_typed(self):
        patch = self.schemas["apply_scene_patch"]["properties"]["patch"]
        self.assertEqual(patch["$ref"], "#/$defs/ScenePatchDTO")
        scene_schema = SceneDTO.model_json_schema()
        object_values = scene_schema["properties"]["objects"]["additionalProperties"]
        self.assertEqual(object_values["discriminator"]["propertyName"], "kind")
        self.assertIn("ObjectFluidDTO", scene_schema["$defs"])

    def test_wire_validation_rejects_unknown_fields_and_bad_vectors(self):
        adapter = TypeAdapter(ObjectSpecDTO)
        with self.assertRaises(ValidationError):
            adapter.validate_python({"id": "bad", "kind": "rigid", "unknown": 1})
        with self.assertRaises(ValidationError):
            adapter.validate_python({"id": "bad", "kind": "rigid", "size": [1.0, 2.0]})
        with self.assertRaises(ValidationError):
            adapter.validate_python({"id": "bad", "kind": "other"})

    def test_fastmcp_calls_create_and_partially_update_typed_fluid(self):
        async def exercise_tools():
            with tempfile.TemporaryDirectory() as temporary:
                tools = SceneTools(SceneStore(Path(temporary) / "scenes"), SceneExecutorLocal())
                with patch("ca_mcp_server.server._tools", return_value=tools):
                    await mcp.call_tool("create_scene", {"name": "typed"})
                    await mcp.call_tool(
                        "add_object",
                        {
                            "scene_name": "typed",
                            "spec": {
                                "id": "water",
                                "kind": "fluid",
                                "phase": "liquid",
                                "grid_resolution": [8, 9, 10],
                                "emitters": [{"size": [0.1, 0.2, 0.3]}],
                            },
                        },
                    )
                    _, updated = await mcp.call_tool(
                        "update_object",
                        {
                            "scene_name": "typed",
                            "item_id": "water",
                            "patch": {"kind": "fluid", "flip_ratio": 0.8},
                        },
                    )
                    return updated

        updated = asyncio.run(exercise_tools())
        self.assertEqual(updated["flip_ratio"], 0.8)
        self.assertEqual(updated["grid_resolution"], [8, 9, 10])
        self.assertEqual(updated["emitters"][0]["size"], [0.1, 0.2, 0.3])

    def test_object_dto_fields_track_runtime_dataclasses(self):
        pairs = (
            (ObjectRigidDTO, ObjectRigid),
            (ObjectClothDTO, ObjectCloth),
            (ObjectFluidDTO, ObjectFluid),
            (ObjectContainerDTO, ObjectContainer),
        )
        for dto, runtime in pairs:
            with self.subTest(dto=dto.__name__):
                self.assertEqual(set(dto.model_fields), {field.name for field in fields(runtime)})


if __name__ == "__main__":
    unittest.main()
