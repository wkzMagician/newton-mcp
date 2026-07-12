# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Reproducible Optuna ask/tell simulation runner."""

from __future__ import annotations

import json
import time
import uuid
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from threading import Event, Thread, Timer
from typing import Any, Callable

import numpy as np
import optuna

from ca_framework.metrics import MetricPipeline, ObjectiveResult, TaskSpec
from ca_framework.scene import Scene, SceneExecutorLocal, validate_scene

from .model import OptimizationPlan
from .fidelity import make_f1_scene
from .optuna_backend import best_trial_payload, create_study, decode_parameters, distributions_for
from .patch import apply_parameter_patch
from .parameter_space import activate_parameters
from .report import build_report
from .storage import OptimizationJob, OptimizationJobStore

TaskMetricFunction = Callable[[Scene, dict[str, Any]], dict[str, float]]
TaskMetrics = TaskMetricFunction | TaskSpec


class OptimizationRunner:
    """Execute framework-owned fidelity gates around an Optuna study."""

    def __init__(
        self,
        output_dir: str | Path,
        *,
        executor: SceneExecutorLocal | None = None,
        metrics: MetricPipeline | None = None,
        worker_id: str | None = None,
    ) -> None:
        self.output_dir = Path(output_dir).resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.executor = executor or SceneExecutorLocal()
        self.metrics = metrics or MetricPipeline()
        self.worker_id = worker_id or uuid.uuid4().hex
        self.jobs = OptimizationJobStore(self.output_dir / "optimization_jobs.sqlite3")

    def run(
        self,
        base_scene: Scene,
        plan: OptimizationPlan,
        *,
        task_metrics: TaskMetrics | None = None,
    ) -> dict[str, Any]:
        """Run all requested trials and return a reproducible study summary."""
        (self.output_dir / "optimization_plan.json").write_text(
            json.dumps(plan.to_dict(), indent=2) + "\n", encoding="utf-8"
        )
        (self.output_dir / "base_scene.json").write_text(
            json.dumps(base_scene.to_dict(), indent=2) + "\n", encoding="utf-8"
        )
        if isinstance(task_metrics, TaskSpec):
            (self.output_dir / "task_spec.json").write_text(
                json.dumps(task_metrics.to_dict(), indent=2) + "\n", encoding="utf-8"
            )
        effective_plan = replace(plan, parameters=activate_parameters(base_scene, plan.parameters, optimizer=plan.optimizer))
        study = create_study(effective_plan)
        study_started = time.monotonic()
        deadline = (
            study_started + plan.max_wall_time_sec
            if plan.max_wall_time_sec is not None
            else None
        )
        distributions = distributions_for(effective_plan)
        specs = {item.path: item for item in plan.parameters}
        remaining_trials = max(0, plan.trials - len(study.trials))
        for _ in range(remaining_trials):
            if deadline is not None and time.monotonic() >= deadline:
                break
            trial = study.ask(fixed_distributions=distributions)
            trial_dir = self.output_dir / f"trial_{trial.number:05d}"
            trial_dir.mkdir(parents=True, exist_ok=False)
            job_id = uuid.uuid4().hex
            started = time.time()
            self.jobs.create(
                OptimizationJob(
                    job_id=job_id,
                    study_name=study.study_name,
                    trial_number=trial.number,
                    worker_id=self.worker_id,
                    state="running",
                    created_at=started,
                    heartbeat_at=started,
                    result_dir=str(trial_dir),
                )
            )
            heartbeat_stop = Event()
            Thread(
                target=self._heartbeat_loop,
                args=(job_id, heartbeat_stop),
                daemon=True,
                name=f"optimization-heartbeat-{trial.number}",
            ).start()
            try:
                parameters = decode_parameters(specs, trial.params)
                candidate = apply_parameter_patch(base_scene, parameters, specs)
                static_report = validate_scene(candidate)
                if not static_report["valid"]:
                    objective = self._static_failure(static_report)
                    self._write_trial(
                        trial_dir, candidate, parameters, objective, None, static_report["diagnostics"]
                    )
                    study.tell(trial, objective.objective)
                    self._finish_job(job_id, "failed", heartbeat_stop)
                    continue

                f1_scene = make_f1_scene(candidate, plan.fidelity)
                short_frames = max(1, round(f1_scene.settings.duration * f1_scene.settings.fps))
                f1_started = time.perf_counter()
                f1 = self._simulate_with_timeout(
                    f1_scene,
                    timeout_sec=self._effective_timeout(plan.timeout_sec, deadline),
                    frames=short_frames,
                    capture_cache=False,
                )
                f1_runtime = time.perf_counter() - f1_started
                f1_objective = self.metrics.evaluate(f1, runtime_sec=f1_runtime)
                if not f1_objective.feasible:
                    self._write_trial(
                        trial_dir, candidate, parameters, f1_objective, f1, f1.get("diagnostics", [])
                    )
                    study.tell(trial, f1_objective.objective)
                    self._finish_job(job_id, "pruned", heartbeat_stop)
                    continue

                f2_started = time.perf_counter()
                f2 = self._simulate_with_timeout(
                    candidate,
                    timeout_sec=self._effective_timeout(plan.timeout_sec, deadline),
                    capture_cache=True,
                )
                runtime = time.perf_counter() - f2_started
                task = self._evaluate_task(task_metrics, candidate, f2)
                objective = self.metrics.evaluate(f2, runtime_sec=runtime, task_metrics=task)
                self._write_trial(trial_dir, candidate, parameters, objective, f2, f2.get("diagnostics", []))
                trial.set_user_attr("feasible", objective.feasible)
                trial.set_user_attr("result_dir", str(trial_dir))
                study.tell(trial, objective.objective)
                self._finish_job(
                    job_id, "completed" if objective.feasible else "failed", heartbeat_stop
                )
            except TimeoutError as error:
                failure = {
                    "status": "timeout",
                    "failure_reason": str(error),
                    "trial": trial.number,
                }
                (trial_dir / "result.json").write_text(json.dumps(failure, indent=2) + "\n", encoding="utf-8")
                study.tell(trial, state=optuna.trial.TrialState.FAIL)
                self._finish_job(job_id, "timeout", heartbeat_stop)
            except Exception as error:
                failure = {
                    "status": "failed",
                    "failure_reason": f"{type(error).__name__}: {error}",
                    "trial": trial.number,
                }
                (trial_dir / "result.json").write_text(json.dumps(failure, indent=2) + "\n", encoding="utf-8")
                study.tell(trial, state=optuna.trial.TrialState.FAIL)
                self._finish_job(job_id, "failed", heartbeat_stop)
        robust_scores = self._robust_recheck(
            base_scene, plan, study, specs, task_metrics, deadline=deadline
        )
        report = build_report(
            self.output_dir, robust_scores, acceptable_objective=plan.acceptable_objective
        )
        finalist_artifacts = self._render_finalists(base_scene, plan, specs, report)
        summary = {
            "study_name": study.study_name,
            "optimizer": plan.optimizer,
            "trial_count": len(study.trials),
            "wall_time_sec": time.monotonic() - study_started,
            "wall_time_budget_exhausted": deadline is not None and time.monotonic() >= deadline,
            "optuna_best_trial": best_trial_payload(study),
            **report,
            "finalist_artifacts": finalist_artifacts,
        }
        (self.output_dir / "optimization_summary.json").write_text(
            json.dumps(summary, indent=2) + "\n", encoding="utf-8"
        )
        return summary

    def _heartbeat_loop(self, job_id: str, stop: Event) -> None:
        """Renew an external ask/tell lease until its trial becomes terminal."""
        while not stop.wait(5.0):
            if not self.jobs.heartbeat(job_id, self.worker_id):
                return

    def _finish_job(self, job_id: str, state: str, stop: Event) -> None:
        stop.set()
        self.jobs.finish(job_id, state)

    def _render_finalists(
        self,
        base_scene: Scene,
        plan: OptimizationPlan,
        specs: dict[str, Any],
        report: dict[str, Any],
    ) -> list[dict[str, Any]]:
        if plan.fidelity.render_top_k <= 0 or not hasattr(self.executor, "run"):
            return []
        selected = []
        seen = set()
        for key in ("best_feasible_trial", "fastest_acceptable_trial", "most_robust_trial"):
            trial = report.get(key)
            if trial is not None and trial["number"] not in seen:
                selected.append(trial)
                seen.add(trial["number"])
            if len(selected) >= plan.fidelity.render_top_k:
                break
        artifacts = []
        for trial in selected:
            scene = apply_parameter_patch(base_scene, trial["parameters"], specs)
            directory = self.output_dir / "finalists" / f"trial_{trial['number']:05d}"
            result = self.executor.run(scene, output_dir=directory)
            artifacts.append({"trial_number": trial["number"], "status": result["status"], "output_dir": str(directory)})
        return artifacts

    def _robust_recheck(
        self,
        base_scene: Scene,
        plan: OptimizationPlan,
        study: optuna.study.Study,
        specs: dict[str, Any],
        task_metrics: TaskMetrics | None,
        *,
        deadline: float | None,
    ) -> dict[int, float]:
        candidates = sorted(
            (
                trial
                for trial in study.get_trials(states=(optuna.trial.TrialState.COMPLETE,))
                if trial.user_attrs.get("feasible", False) and trial.value is not None
            ),
            key=lambda trial: float(trial.value),
        )[: plan.robustness_top_k]
        if not candidates or plan.robustness_samples == 0:
            return {}
        scores = {}
        for trial in candidates:
            candidate = apply_parameter_patch(base_scene, decode_parameters(specs, trial.params), specs)
            objectives = []
            for sample in range(plan.robustness_samples):
                if deadline is not None and time.monotonic() >= deadline:
                    break
                perturbed = self._perturb_scene(candidate, plan.seed + trial.number * 1009 + sample)
                started = time.perf_counter()
                simulation = self._simulate_with_timeout(
                    perturbed,
                    timeout_sec=self._effective_timeout(plan.timeout_sec, deadline),
                    capture_cache=False,
                )
                runtime = time.perf_counter() - started
                task = self._evaluate_task(task_metrics, perturbed, simulation)
                objectives.append(self.metrics.evaluate(simulation, runtime_sec=runtime, task_metrics=task).objective)
            if not objectives:
                break
            values = np.asarray(objectives, dtype=float)
            scores[trial.number] = float(values.mean() + plan.robustness_beta * values.std())
            path = self.output_dir / f"trial_{trial.number:05d}" / "robustness.json"
            path.write_text(
                json.dumps({"objectives": objectives, "robust_objective": scores[trial.number]}, indent=2) + "\n",
                encoding="utf-8",
            )
        return scores

    def _simulate_with_timeout(
        self,
        scene: Scene,
        *,
        timeout_sec: float | None,
        frames: int | None = None,
        capture_cache: bool,
    ) -> dict[str, Any]:
        if timeout_sec is None:
            return self.executor.simulate(scene, frames=frames, capture_cache=capture_cache)
        cancel = Event()
        timer = Timer(timeout_sec, cancel.set)
        timer.start()
        try:
            try:
                result = self.executor.simulate(
                    scene,
                    frames=frames,
                    capture_cache=capture_cache,
                    cancel_event=cancel,
                )
            except TypeError as error:
                if "cancel_event" not in str(error):
                    raise
                result = self.executor.simulate(scene, frames=frames, capture_cache=capture_cache)
            if cancel.is_set():
                raise TimeoutError(f"Simulation exceeded {timeout_sec:g} seconds")
            return result
        except Exception:
            if cancel.is_set():
                raise TimeoutError(f"Simulation exceeded {timeout_sec:g} seconds") from None
            raise
        finally:
            timer.cancel()

    @staticmethod
    def _effective_timeout(timeout_sec: float | None, deadline: float | None) -> float | None:
        """Combine per-simulation and study-wide wall-clock limits."""
        if deadline is None:
            return timeout_sec
        remaining = deadline - time.monotonic()
        if remaining <= 0.0:
            raise TimeoutError("Optimization study wall-time budget exhausted")
        return min(timeout_sec, remaining) if timeout_sec is not None else remaining

    @staticmethod
    def _evaluate_task(
        evaluator: TaskMetrics | None, scene: Scene, simulation: dict[str, Any]
    ) -> dict[str, float] | None:
        if evaluator is None:
            return None
        if isinstance(evaluator, TaskSpec):
            return evaluator.evaluate(simulation)
        return evaluator(scene, simulation)

    @staticmethod
    def _perturb_scene(scene: Scene, seed: int) -> Scene:
        candidate = deepcopy(scene)
        generator = np.random.default_rng(seed)
        candidate.settings.substeps = max(
            1, candidate.settings.substeps + int(generator.choice((-1, 0, 1)))
        )
        for item in candidate.objects.values():
            if item.motion == "dynamic":
                position = np.asarray(item.transform.position, dtype=float)
                item.transform.position = tuple(position + generator.normal(0.0, 0.005, 3))
                item.physical_material.density *= float(generator.uniform(0.98, 1.02))
                if hasattr(item, "linear_velocity"):
                    velocity = np.asarray(item.linear_velocity, dtype=float)
                    item.linear_velocity = tuple(velocity + generator.normal(0.0, 0.01, 3))
        return candidate

    def fail_stale_trials(self, plan: OptimizationPlan, timeout_sec: float) -> list[int]:
        """Fail expired external leases because Optuna ask/tell has no heartbeat."""
        study = create_study(plan)
        failed = []
        for job in self.jobs.stale(timeout_sec):
            study.tell(job.trial_number, state=optuna.trial.TrialState.FAIL)
            self.jobs.finish(job.job_id, "timeout")
            failed.append(job.trial_number)
        return failed

    @staticmethod
    def _static_failure(report: dict[str, Any]) -> ObjectiveResult:
        count = sum(item["severity"] == "error" for item in report["diagnostics"])
        return ObjectiveResult(
            feasible=False,
            objective=1000.0 * (1.0 + count),
            metrics={},
            constraints={"static_validation_errors": float(count)},
            runtime_sec=0.0,
            status="infeasible",
            failure_reason="static_validation",
        )

    @staticmethod
    def _write_trial(
        directory: Path,
        scene: Scene,
        parameters: dict[str, Any],
        objective: ObjectiveResult,
        simulation: dict[str, Any] | None,
        diagnostics: list[dict[str, Any]],
    ) -> None:
        (directory / "scene.json").write_text(json.dumps(scene.to_dict(), indent=2) + "\n", encoding="utf-8")
        (directory / "parameters.json").write_text(json.dumps(parameters, indent=2) + "\n", encoding="utf-8")
        (directory / "metrics.json").write_text(
            json.dumps(objective.to_dict(), indent=2) + "\n", encoding="utf-8"
        )
        (directory / "diagnostics.jsonl").write_text(
            "".join(json.dumps(item, sort_keys=True) + "\n" for item in diagnostics), encoding="utf-8"
        )
        result = {
            "status": objective.status,
            "feasible": objective.feasible,
            "objective": objective.objective,
            "failure_reason": objective.failure_reason,
        }
        (directory / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
        if simulation is None or "telemetry_records" not in simulation:
            return
        telemetry_dir = directory / "telemetry"
        telemetry_dir.mkdir()
        records = simulation["telemetry_records"]
        (telemetry_dir / "solver.jsonl").write_text(
            "".join(
                json.dumps(
                    {
                        "frame": item["frame"],
                        "substep": item["substep"],
                        "time": item["time"],
                        **item["solver_stats"],
                    },
                    sort_keys=True,
                )
                + "\n"
                for item in records
            ),
            encoding="utf-8",
        )
        (telemetry_dir / "contacts.jsonl").write_text(
            "".join(
                json.dumps({"frame": item["frame"], "substep": item["substep"], **contact}) + "\n"
                for item in records
                for contact in item["contacts"]
            ),
            encoding="utf-8",
        )
        (telemetry_dir / "coupling.jsonl").write_text(
            "".join(
                json.dumps({"frame": item["frame"], "substep": item["substep"], **exchange}) + "\n"
                for item in records
                for exchange in item["coupling_exchanges"]
            ),
            encoding="utf-8",
        )
        arrays = {
            f"frame_{frame_index:05d}_{key}": value
            for frame_index, frame in enumerate(simulation.get("state_frames", []))
            for key, value in frame.items()
        }
        np.savez_compressed(telemetry_dir / "frames.npz", **arrays)
