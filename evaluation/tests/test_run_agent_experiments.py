# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

import tempfile
import unittest
from pathlib import Path

from evaluation.agent_prompts import AGENT_PROMPTS, build_agent_prompt
from evaluation.canonical import canonical_scenes
from evaluation.run_agent_experiments import REQUIRED_FILES, _artifact_status, _bundle_status, run_case


class TestAgentPrompts(unittest.TestCase):
    def test_prompts_cover_the_canonical_cases_without_serialized_scenes(self):
        self.assertEqual(list(AGENT_PROMPTS), list(canonical_scenes()))
        for case_name, prompt in AGENT_PROMPTS.items():
            with self.subTest(case=case_name):
                self.assertGreater(len(prompt.split()), 25)
                self.assertNotIn('"objects"', prompt)
                self.assertNotIn('"settings"', prompt)
                self.assertNotIn("schema_version", prompt)

    def test_complete_prompt_requires_validation_preview_and_final_job(self):
        prompt = build_agent_prompt("02_domino_wave", "/isolated/result")
        for requirement in (
            "get_capabilities",
            "get_scene_schema",
            "validate_scene",
            "preview_scene",
            "run_scene",
            "get_job",
        ):
            self.assertIn(requirement, prompt)
        self.assertIn("/isolated/result", prompt)


class TestRunAgentExperiments(unittest.TestCase):
    def test_empty_diagnostics_file_is_a_complete_success_artifact(self):
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            for name in REQUIRED_FILES:
                content = "" if name == "diagnostics.jsonl" else "content"
                (output_dir / name).write_text(content, encoding="utf-8")
            (output_dir / "cache").mkdir()
            (output_dir / "cache" / "manifest.json").write_text("{}", encoding="utf-8")

            self.assertTrue(all(_artifact_status(output_dir).values()))

    def test_bundle_status_rejects_physics_failure_even_with_video(self):
        with tempfile.TemporaryDirectory() as temporary:
            output_dir = Path(temporary)
            (output_dir / "manifest.json").write_text('{"status": "physics_failed"}', encoding="utf-8")
            (output_dir / "metrics.json").write_text('{"physics": {"valid": false}}', encoding="utf-8")

            self.assertEqual(_bundle_status(output_dir), ("physics_failed", False))

    def test_dry_run_prepares_only_agent_facing_inputs(self):
        repo = Path(__file__).resolve().parents[2]
        with tempfile.TemporaryDirectory() as temporary:
            run_root = Path(temporary) / "run"
            run_root.mkdir()
            result = run_case(
                "01_rigid_contacts",
                repo=repo,
                run_root=run_root,
                auth_source=None,
                model=None,
                timeout=1.0,
                dry_run=True,
            )

            workspace = Path(result["workspace"])
            self.assertEqual(result["status"], "dry-run")
            self.assertTrue((run_root / "01_rigid_contacts" / "prompt.txt").is_file())
            self.assertTrue((workspace / ".agents" / "skills" / "orchestrate-animation-scene" / "SKILL.md").is_file())
            self.assertFalse((workspace / "evaluation").exists())
            self.assertFalse((workspace / "scene.json").exists())


if __name__ == "__main__":
    unittest.main()
