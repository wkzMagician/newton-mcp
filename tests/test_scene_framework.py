# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import json
import tempfile
import unittest
from pathlib import Path

from ca_framework.mcp import SceneTools
from ca_framework.scene import SceneExecutorLocal, SceneStore


class TestSceneFramework(unittest.TestCase):
    def setUp(self):
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.tools = SceneTools(SceneStore(self.root / "scenes"), SceneExecutorLocal())
        self.tools.create_scene("demo")

    def tearDown(self):
        self.temporary_directory.cleanup()

    def test_create_edit_and_round_trip_scene(self):
        self.tools.add_object(
            "demo",
            {
                "id": "ball",
                "kind": "rigid",
                "shape": "sphere",
                "size": [0.5, 0.5, 0.5],
                "transform": {"position": [0.0, 0.0, 2.0]},
            },
        )
        self.tools.add_field(
            "demo",
            {"id": "wind", "kind": "uniform", "vector": [1.0, 0.0, 0.0], "object_ids": ["ball"]},
        )

        scene = self.tools.get_scene("demo")

        self.assertEqual(scene["objects"]["ball"]["shape"], "sphere")
        self.assertEqual(scene["fields"]["wind"]["object_ids"], ["ball"])
        self.assertEqual(SceneStore(self.root / "scenes").load("demo").name, "demo")

    def test_rejects_dangling_references(self):
        self.tools.add_object("demo", {"id": "a", "kind": "rigid"})
        with self.assertRaisesRegex(ValueError, "Unknown object id: missing"):
            self.tools.add_constraint(
                "demo",
                {"id": "link", "kind": "distance", "object_a": "a", "object_b": "missing"},
            )

    def test_rejects_removing_referenced_object(self):
        self.tools.add_object("demo", {"id": "a", "kind": "rigid"})
        self.tools.add_object("demo", {"id": "b", "kind": "rigid"})
        self.tools.add_constraint("demo", {"id": "link", "kind": "distance", "object_a": "a", "object_b": "b"})

        with self.assertRaisesRegex(ValueError, "constraint:link"):
            self.tools.remove_item("demo", "objects", "a")

    def test_rejects_kind_change_and_dangling_update(self):
        self.tools.add_object("demo", {"id": "a", "kind": "rigid"})
        self.tools.add_object("demo", {"id": "b", "kind": "rigid"})
        self.tools.add_constraint("demo", {"id": "link", "kind": "distance", "object_a": "a", "object_b": "b"})

        with self.assertRaisesRegex(ValueError, "kinds cannot be changed"):
            self.tools.update_item("demo", "objects", "a", {"kind": "cloth"})
        with self.assertRaisesRegex(ValueError, "Unknown object id: missing"):
            self.tools.update_item("demo", "constraints", "link", {"object_b": "missing"})

    def test_exports_json(self):
        output = self.root / "exports" / "demo.json"

        result = self.tools.export_scene("demo", str(output))

        self.assertEqual(result["status"], "completed")
        self.assertEqual(json.loads(output.read_text())["name"], "demo")

    def test_scene_name_cannot_escape_workspace(self):
        with self.assertRaises(ValueError):
            self.tools.create_scene("../outside")


if __name__ == "__main__":
    unittest.main()
