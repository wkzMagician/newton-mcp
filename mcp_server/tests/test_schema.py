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

    def test_cloth_schema_states_that_position_is_the_sheet_center(self):
        schema = self.schemas["add_object"]
        transform_position = schema["$defs"]["TransformDTO"]["properties"]["position"]["description"]
        cloth_size = schema["$defs"]["ObjectClothDTO"]["properties"]["size"]["description"]
        tool_description = next(tool.description for tool in asyncio.run(mcp.list_tools()) if tool.name == "add_object")
        self.assertIn("center", transform_position)
        self.assertIn("half", cloth_size)
        self.assertIn("sheet center", tool_description)

    def test_container_schema_states_floor_center_origin(self):
        schema = self.schemas["add_object"]
        properties = schema["$defs"]["ObjectContainerDTO"]["properties"]

        self.assertIn("interior floor", properties["transform"]["description"])
        self.assertIn("z interval runs from 0", properties["inner_size"]["description"])

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
            "create_optimization_plan",
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

    def test_set_camera_exposes_typed_fixed_view(self):
        schema = self.schemas["set_camera"]
        camera = schema["properties"]["camera"]
        self.assertEqual(camera["$ref"], "#/$defs/CameraLookAtDTO")
        properties = schema["$defs"]["CameraLookAtDTO"]["properties"]
        self.assertEqual(set(properties), {"position", "target", "up", "field_of_view"})

    def test_optimization_lifecycle_tools_are_registered(self):
        expected = {
            "create_optimization_plan",
            "validate_optimization_plan",
            "start_optimization",
            "get_optimization_job",
            "list_optimization_trials",
            "get_optimization_trial",
            "apply_optimization_trial",
            "compare_optimization_trials",
        }
        self.assertTrue(expected.issubset(self.schemas))
        plan = self.schemas["create_optimization_plan"]["properties"]["plan"]
        self.assertEqual(plan["$ref"], "#/$defs/OptimizationPlanDTO")
        plan_schema = self.schemas["create_optimization_plan"]["$defs"]["OptimizationPlanDTO"]
        self.assertIn("initial_parameters", plan_schema["properties"])
        task_spec = self.schemas["create_optimization_plan"]["properties"]["task_spec"]
        self.assertEqual(task_spec["anyOf"][0]["$ref"], "#/$defs/TaskSpecDTO")
        objective_settings = self.schemas["create_optimization_plan"]["properties"]["objective_settings"]
        self.assertEqual(objective_settings["anyOf"][0]["$ref"], "#/$defs/ObjectiveSettingsDTO")

    def test_capabilities_name_the_required_final_bundle_workflow_tools(self):
        with tempfile.TemporaryDirectory() as temporary:
            capabilities = SceneTools(SceneStore(Path(temporary) / "scenes")).get_capabilities()

        self.assertEqual(
            capabilities["workflow_tools"],
            {
                "validate": "validate_scene",
                "preview": "preview_scene",
                "final_bundle": "run_scene",
                "job_status": "get_job",
            },
        )

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
                                "transform": {"position": [1.0, 2.0, 3.0]},
                            },
                        },
                    )
                    _, updated = await mcp.call_tool(
                        "update_object",
                        {
                            "scene_name": "typed",
                            "item_id": "water",
                            "patch": {
                                "kind": "fluid",
                                "flip_ratio": 0.8,
                                "transform": {"rotation": [0.0, 0.0, 0.0, 1.0]},
                            },
                        },
                    )
                    return updated

        updated = asyncio.run(exercise_tools())
        self.assertEqual(updated["flip_ratio"], 0.8)
        self.assertEqual(updated["grid_resolution"], [8, 9, 10])
        self.assertEqual(updated["emitters"][0]["size"], [0.1, 0.2, 0.3])
        self.assertEqual(updated["transform"]["position"], [1.0, 2.0, 3.0])

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
