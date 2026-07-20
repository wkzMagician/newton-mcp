# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Tests for Optuna ask/tell and framework lease integration."""

from __future__ import annotations

import json
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np
import optuna

from ca_framework.optimization import (
    FidelitySettings,
    OptimizationPlan,
    OptimizationRunner,
    ParameterSpec,
    run_optimizer_suite,
)
from ca_framework.optimization.optuna_backend import create_sampler, decode_parameters, distributions_for
from ca_framework.optimization.storage import OptimizationJob, OptimizationJobStore
from ca_framework.mcp.optimization_tools import OptimizationTools
from ca_framework.scene import ObjectCloth, ObjectFluid, ObjectRigid, Scene, SceneStore


class FakeExecutor:
    def __init__(self):
        self.simulation_calls = 0

    def simulate(self, scene, *, frames=None, capture_cache=False):
        self.simulation_calls += 1
        value = float(scene.settings.coupling.relaxation)
        result = {
            "status": "completed",
            "metrics": {
                "max_penetration": value * 0.001,
                "physics": {"valid": True},
                "fluid": {},
                "cloth": {},
                "coupling": {"impulse_balance_error": 0.0},
            },
            "diagnostics": [],
        }
        if capture_cache:
            result["telemetry_records"] = [
                {
                    "frame": 0,
                    "substep": 0,
                    "time": 0.1,
                    "body_states": {},
                    "cloth_states": {},
                    "fluid_states": {},
                    "contacts": [],
                    "coupling_exchanges": [],
                    "solver_stats": {},
                }
            ]
            result["state_frames"] = [{"body_q": np.zeros((0, 7), dtype=np.float32)}]
        return result


class CancelAwareExecutor:
    def simulate(self, scene, *, frames=None, capture_cache=False, cancel_event=None):
        while cancel_event is None or not cancel_event.wait(0.001):
            pass
        return {"status": "cancelled", "metrics": {}, "diagnostics": []}


class BlockingExecutor:
    def simulate(self, scene, *, frames=None, capture_cache=False):
        time.sleep(10.0)
        return {"status": "completed", "metrics": {"physics": {"valid": True}}, "diagnostics": []}


class LowFidelityFalseNegativeExecutor(FakeExecutor):
    def simulate(self, scene, *, frames=None, capture_cache=False):
        if frames is not None:
            return {
                "status": "physics_failed",
                "metrics": {
                    "max_penetration": 0.0,
                    "physics": {"valid": False},
                    "fluid": {"water": {"finite": True, "peak_divergence": 100.0}},
                    "cloth": {},
                    "coupling": {"impulse_balance_error": 0.0},
                },
                "diagnostics": [{"code": "low_fidelity_failure"}],
            }
        return super().simulate(scene, frames=frames, capture_cache=capture_cache)


class RenderFakeExecutor(FakeExecutor):
    def __init__(self):
        super().__init__()
        self.rendered = []

    def run(self, scene, *, output_dir):
        self.rendered.append(Path(output_dir))
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        Path(output_dir, "scene.mp4").write_bytes(b"candidate")
        return {"status": "completed"}


class TestOptunaBackend(unittest.TestCase):
    def test_supported_sampler_factory(self) -> None:
        self.assertIsInstance(create_sampler("random", 1), optuna.samplers.RandomSampler)
        self.assertIsInstance(create_sampler("tpe", 1), optuna.samplers.TPESampler)
        self.assertEqual(create_sampler("tpe", 1, tpe_startup_trials=8)._n_startup_trials, 8)
        self.assertIsInstance(create_sampler("cmaes", 1), optuna.samplers.CmaEsSampler)
        self.assertEqual(create_sampler("cmaes", 1, cmaes_startup_trials=8)._n_startup_trials, 8)
        with self.assertRaises(ValueError):
            create_sampler("unknown", 1)

    def test_distributions_follow_optimizer_stage(self) -> None:
        global_spec = ParameterSpec(
            "relaxation",
            "settings.coupling.relaxation",
            "coupling",
            "float",
            0.1,
            1.0,
        )
        refine_spec = ParameterSpec(
            "drag",
            "settings.coupling.cloth_fluid_drag",
            "coupling",
            "float",
            0.1,
            2.0,
            stage="refine",
        )
        plan = OptimizationPlan("stages", parameters=(global_spec, refine_spec))
        self.assertEqual(set(distributions_for(plan)), {global_spec.path})

    def test_logit_parameters_are_sampled_in_latent_space_and_decoded(self) -> None:
        spec = ParameterSpec(
            "ratio",
            "settings.coupling.relaxation",
            "coupling",
            "float",
            0.1,
            0.9,
            transform="logit",
        )
        distribution = distributions_for(OptimizationPlan("logit", parameters=(spec,)))[spec.path]
        self.assertLess(distribution.low, 0.0)
        self.assertGreater(distribution.high, 0.0)
        self.assertAlmostEqual(decode_parameters({spec.path: spec}, {spec.path: 0.0})[spec.path], 0.5)


class TestOptimizationRunner(unittest.TestCase):
    def test_ask_tell_trials_write_reproducible_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            spec = ParameterSpec(
                "relaxation",
                "settings.coupling.relaxation",
                "coupling",
                "float",
                0.2,
                0.9,
            )
            plan = OptimizationPlan("runner-test", optimizer="random", parameters=(spec,), trials=2, seed=7)
            summary = OptimizationRunner(directory, executor=FakeExecutor()).run(Scene("base"), plan)
            self.assertEqual(summary["trial_count"], 2)
            self.assertIsNotNone(summary["best_feasible_trial"])
            self.assertIsNotNone(summary["fastest_acceptable_trial"])
            self.assertIn("time_to_objective_threshold_sec", summary)
            self.assertEqual(summary["timeout_rate"], 0.0)
            self.assertTrue((Path(directory) / "best_so_far.svg").exists())
            self.assertTrue((Path(directory) / "optimization_plan.json").exists())
            self.assertTrue((Path(directory) / "base_scene.json").exists())
            for index in range(2):
                trial = Path(directory) / f"trial_{index:05d}"
                for name in ("scene.json", "parameters.json", "result.json", "metrics.json", "diagnostics.jsonl"):
                    self.assertTrue((trial / name).exists(), name)
                self.assertTrue((trial / "simulation_metrics.json").exists())
                self.assertFalse((trial / "telemetry").exists())

    def test_initial_parameter_candidate_is_executed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            spec = ParameterSpec(
                "relaxation",
                "settings.coupling.relaxation",
                "coupling",
                "float",
                0.2,
                0.9,
            )
            plan = OptimizationPlan(
                "seeded-runner-test",
                optimizer="tpe",
                parameters=(spec,),
                initial_parameters=({spec.path: 0.4},),
                trials=1,
                robustness_top_k=0,
            )
            scene = Scene(
                "base",
                objects={"body": ObjectRigid("body"), "cloth": ObjectCloth("cloth")},
            )
            summary = OptimizationRunner(directory, executor=FakeExecutor()).run(scene, plan)
            self.assertEqual(summary["trial_count"], 1)
            self.assertEqual(summary["best_feasible_trial"]["parameters"], {spec.path: 0.4})

    def test_framework_store_detects_stale_external_lease(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = OptimizationJobStore(Path(directory) / "jobs.sqlite3")
            store.create(OptimizationJob("job", "study", 3, "worker", "running", 1.0, 1.0, directory))
            self.assertEqual([item.job_id for item in store.stale(10.0, now=20.0)], ["job"])
            self.assertTrue(store.heartbeat("job", "worker", timestamp=time.time()))

    def test_trial_timeout_is_recorded(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            plan = OptimizationPlan(
                "timeout-test",
                trials=1,
                timeout_sec=0.01,
                robustness_top_k=0,
            )
            summary = OptimizationRunner(directory, executor=CancelAwareExecutor()).run(Scene("base"), plan)
            result = Path(directory, "trial_00000", "result.json").read_text(encoding="utf-8")
            self.assertEqual(summary["trial_count"], 1)
            self.assertIn('"status": "timeout"', result)

    def test_trial_timeout_terminates_non_cooperative_simulation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            plan = OptimizationPlan(
                "hard-timeout-test",
                trials=1,
                timeout_sec=0.01,
                robustness_top_k=0,
            )
            started = time.monotonic()
            summary = OptimizationRunner(directory, executor=BlockingExecutor()).run(Scene("base"), plan)
            elapsed = time.monotonic() - started
            result = Path(directory, "trial_00000", "result.json").read_text(encoding="utf-8")
            progress = Path(directory, "trial_00000", "progress.jsonl").read_text(encoding="utf-8")
            self.assertLess(elapsed, 2.0)
            self.assertEqual(summary["trial_count"], 1)
            self.assertIn('"status": "timeout"', result)
            self.assertIn("timeout", progress)

    def test_full_fidelity_can_recover_from_low_fidelity_false_negative(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            plan = OptimizationPlan(
                "f1-false-negative-test",
                trials=1,
                robustness_top_k=0,
            )
            summary = OptimizationRunner(directory, executor=LowFidelityFalseNegativeExecutor()).run(Scene("base"), plan)
            diagnostics = Path(directory, "trial_00000", "diagnostics.jsonl").read_text(encoding="utf-8")
            self.assertIsNotNone(summary["best_feasible_trial"])
            self.assertIn("low_fidelity_infeasible", diagnostics)

    def test_study_wall_time_budget_limits_simulation(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            plan = OptimizationPlan(
                "wall-budget-test",
                trials=10,
                max_wall_time_sec=0.01,
                robustness_top_k=0,
            )
            summary = OptimizationRunner(directory, executor=CancelAwareExecutor()).run(
                Scene("base"), plan
            )
            self.assertLess(summary["trial_count"], plan.trials)
            self.assertTrue(summary["wall_time_budget_exhausted"])

    def test_f3_renders_selected_finalist_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            executor = RenderFakeExecutor()
            plan = OptimizationPlan("render-test", trials=1, robustness_top_k=0)
            summary = OptimizationRunner(directory, executor=executor).run(Scene("base"), plan)
            self.assertEqual(len(summary["finalist_artifacts"]), 1)
            self.assertEqual(
                summary["finalist_artifacts"][0]["trial_number"],
                summary["median_feasible_trial"]["number"],
            )
            self.assertTrue((executor.rendered[0] / "scene.mp4").exists())

    def test_representative_render_prefers_median_feasible_trial(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            runner = OptimizationRunner(directory, executor=RenderFakeExecutor())
            plan = OptimizationPlan("median-render-test", trials=1, robustness_top_k=0)
            artifacts = runner._render_finalists(
                Scene("base"),
                plan,
                {},
                {
                    "best_feasible_trial": {"number": 0, "parameters": {}},
                    "median_feasible_trial": {"number": 2, "parameters": {}},
                },
            )
            self.assertEqual(artifacts[0]["trial_number"], 2)

    def test_optimizer_suite_runs_independent_samplers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            specs = (
                ParameterSpec(
                    "substeps", "settings.substeps", "solver", "int", 1, 12, stage="global"
                ),
                ParameterSpec(
                    "relaxation",
                    "settings.coupling.relaxation",
                    "coupling",
                    "float",
                    0.2,
                    0.9,
                    stage="refine",
                ),
            )
            result = run_optimizer_suite(
                Scene("base"),
                OptimizationPlan("sampler-suite", parameters=specs, robustness_top_k=0),
                directory,
                trials=1,
                executor=FakeExecutor(),
            )
            self.assertEqual(set(result), {"tpe", "cmaes", "random"})
            for stage in result.values():
                self.assertIsNotNone(stage)
                self.assertIn("best_feasible_trial", stage)
            tpe_plan = OptimizationPlan.from_dict(
                json.loads(Path(directory, "tpe", "optimization_plan.json").read_text())
            )
            self.assertEqual(tpe_plan.tpe_startup_trials, 1)
            cmaes_plan = OptimizationPlan.from_dict(
                json.loads(Path(directory, "cmaes", "optimization_plan.json").read_text())
            )
            self.assertEqual(
                cmaes_plan.initial_parameters,
                ({"settings.substeps": Scene("base").settings.substeps},),
            )

    def test_full_fidelity_equivalent_skips_duplicate_f1(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            executor = FakeExecutor()
            plan = OptimizationPlan(
                "single-pass",
                trials=1,
                fidelity=FidelitySettings(short_duration_fraction=1.0, low_resolution_scale=1.0),
                robustness_top_k=0,
            )
            OptimizationRunner(directory, executor=executor).run(Scene("base"), plan)
            self.assertEqual(executor.simulation_calls, 1)
            self.assertIn("f1_skipped", Path(directory, "trial_00000", "progress.jsonl").read_text())

    def test_cmaes_backend_completes_continuous_trials(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            specs = (
                ParameterSpec(
                    "relaxation",
                    "settings.coupling.relaxation",
                    "coupling",
                    "float",
                    0.2,
                    0.9,
                    stage="refine",
                ),
                ParameterSpec(
                    "drag",
                    "settings.coupling.cloth_fluid_drag",
                    "coupling",
                    "float",
                    0.2,
                    1.5,
                    stage="refine",
                ),
            )
            plan = OptimizationPlan(
                "cma-test",
                optimizer="cmaes",
                parameters=specs,
                trials=3,
                robustness_top_k=0,
            )
            summary = OptimizationRunner(directory, executor=FakeExecutor()).run(Scene("base"), plan)
            self.assertEqual(summary["trial_count"], 3)
            self.assertIsNotNone(summary["best_feasible_trial"])

    def test_rdb_study_resume_honors_total_trial_budget(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            database = Path(directory) / "study.sqlite3"
            plan = OptimizationPlan(
                "resume-test",
                optimizer="random",
                trials=2,
                storage_url=f"sqlite:///{database}",
                robustness_top_k=0,
            )
            runner = OptimizationRunner(Path(directory) / "run", executor=FakeExecutor())
            first = runner.run(Scene("base"), plan)
            second = runner.run(Scene("base"), plan)
            self.assertEqual(first["trial_count"], 2)
            self.assertEqual(second["trial_count"], 2)

    def test_mcp_lifecycle_creates_runs_and_applies_trial(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SceneStore(Path(directory) / "scenes")
            store.save(
                Scene(
                    "base",
                    objects={"ball": ObjectRigid("ball"), "cloth": ObjectCloth("cloth")},
                )
            )
            tools = OptimizationTools(store, FakeExecutor())
            plan = {
                "name": "mcp-plan",
                "optimizer": "random",
                "trials": 1,
                "parameters": [
                    {
                        "name": "relaxation",
                        "path": "settings.coupling.relaxation",
                        "scope": "coupling",
                        "kind": "float",
                        "lower": 0.2,
                        "upper": 0.9,
                    }
                ],
            }
            tools.create_plan(
                "base",
                plan,
                task_spec={
                    "name": "penetration",
                    "metrics": [
                        {
                            "name": "penetration_loss",
                            "path": "metrics.max_penetration",
                            "target": 0.0,
                            "tolerance": 0.001,
                        }
                    ],
                },
            )
            # SceneStore roots are used directly; optimization assets remain separate from Scene metadata.
            plan_dir = store.root / ".optimizations" / "plans" / "mcp-plan"
            self.assertTrue((plan_dir / "scene.json").exists())
            self.assertTrue((plan_dir / "optimization_plan.json").exists())
            self.assertTrue((plan_dir / "task_spec.json").exists())
            self.assertTrue(tools.validate_plan("mcp-plan")["valid"])
            job = tools.start("mcp-plan")
            for _ in range(100):
                status = tools.get_job(job["job_id"])
                if status["status"] not in {"queued", "running"}:
                    break
                time.sleep(0.01)
            self.assertEqual(status["status"], "completed")
            self.assertTrue((Path(status["output_dir"]) / "task_spec.json").exists())
            self.assertEqual(len(tools.list_trials(job["job_id"])), 1)
            applied = tools.apply_trial(job["job_id"], 0, "base")
            self.assertNotEqual(applied["settings"]["coupling"]["relaxation"], 0.7)

    def test_mcp_rejects_enabled_parameter_inactive_for_scene(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SceneStore(Path(directory) / "scenes")
            store.save(Scene("liquid", objects={"water": ObjectFluid("water")}))
            tools = OptimizationTools(store, FakeExecutor())
            plan = {
                "name": "inactive-coupling",
                "optimizer": "random",
                "trials": 1,
                "parameters": [
                    {
                        "name": "coupling_iterations",
                        "path": "settings.coupling.iterations",
                        "scope": "coupling",
                        "kind": "int",
                        "lower": 1,
                        "upper": 4,
                    }
                ],
            }
            with self.assertRaisesRegex(ValueError, "not active for this scene"):
                tools.create_plan("liquid", plan)

    def test_mcp_reports_queued_for_job_waiting_on_single_worker(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            store = SceneStore(Path(directory) / "scenes")
            store.save(
                Scene(
                    "base",
                    objects={"ball": ObjectRigid("ball"), "cloth": ObjectCloth("cloth")},
                )
            )
            tools = OptimizationTools(store, BlockingExecutor())
            plan = {
                "name": "queued-plan",
                "optimizer": "random",
                "trials": 1,
                "timeout_sec": 0.2,
                "robustness_top_k": 0,
                "parameters": [
                    {
                        "name": "relaxation",
                        "path": "settings.coupling.relaxation",
                        "scope": "coupling",
                        "kind": "float",
                        "lower": 0.2,
                        "upper": 0.9,
                    }
                ],
            }
            tools.create_plan("base", plan)
            first = tools.start("queued-plan")
            second = tools.start("queued-plan")
            status = tools.get_job(second["job_id"])
            self.assertEqual(status["status"], "queued")
            self.assertEqual(status["completed_trials"], 0)
            for job in (first, second):
                saw_progress = False
                for _ in range(100):
                    status = tools.get_job(job["job_id"])
                    if status["status"] == "running":
                        self.assertGreaterEqual(status["active_trials"], 1)
                        self.assertTrue(status["active_trial_numbers"])
                        self.assertEqual(status["completed_trials"], 0)
                        if status["progress_tail"]:
                            self.assertEqual(status["progress_tail"][-1]["event"], "f1_started")
                            saw_progress = True
                            break
                    if status["status"] not in {"queued", "running"}:
                        break
                    time.sleep(0.01)
                self.assertTrue(saw_progress)


if __name__ == "__main__":
    unittest.main()
