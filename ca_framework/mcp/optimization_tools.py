# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Protocol-independent optimization lifecycle tools."""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import uuid4

from ca_framework.optimization import OptimizationPlan, OptimizationRunner, activate_parameters, apply_parameter_patch
from ca_framework.optimization.optuna_backend import distributions_for
from ca_framework.metrics import MetricPipeline, ObjectiveSettings, TaskSpec
from ca_framework.scene import Scene, SceneExecutorLocal, SceneStore, validate_scene


class OptimizationTools:
    """Persist plans and run optimization independently from Agent turns."""

    def __init__(self, store: SceneStore, executor: SceneExecutorLocal | None = None) -> None:
        self.store = store
        self.executor = executor or SceneExecutorLocal()
        self.root = store.root / ".optimizations"
        self.plans_dir = self.root / "plans"
        self.runs_dir = self.root / "runs"
        self.plans_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ca-optimization")
        self._jobs: dict[str, tuple[Future[dict[str, Any]], Path]] = {}
        self._lock = Lock()

    def create_plan(
        self,
        scene_name: str,
        value: dict[str, Any],
        *,
        task_spec: dict[str, Any] | None = None,
        objective_settings: dict[str, Any] | None = None,
        overwrite: bool = False,
    ) -> dict[str, Any]:
        """Validate and persist a plan beside an immutable base-scene snapshot."""
        scene = self.store.load(scene_name)
        plan = OptimizationPlan.from_dict(value)
        self._validate_plan_for_scene(scene.to_dict(), plan)
        path = self.plans_dir / plan.name
        if path.exists() and not overwrite:
            raise FileExistsError(f"Optimization plan already exists: {plan.name}")
        path.mkdir(parents=True, exist_ok=True)
        scene_payload = scene.to_dict()
        parsed_task = TaskSpec.from_dict(task_spec) if task_spec is not None else None
        parsed_objective = ObjectiveSettings(**objective_settings) if objective_settings is not None else None
        payload = {
            "plan": plan.to_dict(),
            "scene_name": scene_name,
            "base_scene": scene_payload,
            "base_scene_hash": _hash(scene_payload),
            "task_spec": parsed_task.to_dict() if parsed_task is not None else None,
            "objective_settings": asdict(parsed_objective) if parsed_objective is not None else None,
        }
        (path / "optimization_plan.json").write_text(
            json.dumps(plan.to_dict(), indent=2) + "\n", encoding="utf-8"
        )
        (path / "scene.json").write_text(json.dumps(scene_payload, indent=2) + "\n", encoding="utf-8")
        if parsed_task is not None:
            (path / "task_spec.json").write_text(
                json.dumps(parsed_task.to_dict(), indent=2) + "\n", encoding="utf-8"
            )
        if parsed_objective is not None:
            (path / "objective_settings.json").write_text(
                json.dumps(asdict(parsed_objective), indent=2) + "\n", encoding="utf-8"
            )
        (path / "manifest.json").write_text(
            json.dumps(
                {
                    "scene_name": scene_name,
                    "base_scene_hash": payload["base_scene_hash"],
                    "has_task_spec": parsed_task is not None,
                    "has_objective_settings": parsed_objective is not None,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return payload

    def validate_plan(self, plan_name: str) -> dict[str, Any]:
        """Validate a stored plan and report its active search space."""
        payload = self._load_plan(plan_name)
        plan = OptimizationPlan.from_dict(payload["plan"])
        self._validate_plan_for_scene(payload["base_scene"], plan)
        base = Scene.from_dict(payload["base_scene"])
        effective = replace(plan, parameters=activate_parameters(base, plan.parameters, optimizer=plan.optimizer))
        return {
            "valid": True,
            "plan": plan_name,
            "optimizer": plan.optimizer,
            "active_parameters": list(distributions_for(effective)),
            "base_scene_hash": payload["base_scene_hash"],
        }

    def start(self, plan_name: str, output_dir: str | None = None) -> dict[str, Any]:
        """Start a serialized external simulation study."""
        payload = self._load_plan(plan_name)
        plan = OptimizationPlan.from_dict(payload["plan"])
        scene = Scene.from_dict(payload["base_scene"])
        job_id = uuid4().hex
        directory = Path(output_dir).resolve() if output_dir else self.runs_dir / job_id
        objective_settings = payload.get("objective_settings")
        runner = OptimizationRunner(
            directory,
            executor=self.executor,
            metrics=(MetricPipeline(ObjectiveSettings(**objective_settings)) if objective_settings is not None else None),
        )
        task_spec = TaskSpec.from_dict(payload["task_spec"]) if payload.get("task_spec") is not None else None
        future = self._pool.submit(runner.run, scene, plan, task_metrics=task_spec)
        with self._lock:
            self._jobs[job_id] = (future, directory)
        return {"job_id": job_id, "status": "running", "plan": plan_name, "output_dir": str(directory)}

    def close(self) -> None:
        """Release the serial optimization worker after a benchmark case completes."""
        self._pool.shutdown(wait=True)

    def get_job(self, job_id: str) -> dict[str, Any]:
        """Return optimization progress or its terminal summary."""
        with self._lock:
            future, directory = self._jobs[job_id]
        trial_dirs = list(directory.glob("trial_*")) if directory.exists() else []
        if not future.done():
            completed = [self._read_trial(path) for path in sorted(trial_dirs) if self._trial_complete(path)]
            active = [path for path in sorted(trial_dirs) if not self._trial_complete(path)]
            feasible = [item for item in completed if item["result"].get("feasible", False)]
            best = min(feasible, key=lambda item: item["result"]["objective"], default=None)
            return {
                "job_id": job_id,
                "status": "running" if trial_dirs else "queued",
                "completed_trials": len(completed),
                "active_trials": len(active),
                "active_trial_numbers": [int(path.name.removeprefix("trial_")) for path in active],
                "progress_tail": self._progress_tail(active[-1]) if active else [],
                "best_trial": best,
                "output_dir": str(directory),
            }
        try:
            return {"job_id": job_id, "status": "completed", **future.result(), "output_dir": str(directory)}
        except Exception as error:
            return {"job_id": job_id, "status": "failed", "error": f"{type(error).__name__}: {error}"}

    def list_trials(self, job_id: str) -> list[dict[str, Any]]:
        """List completed trial summaries in numeric order."""
        directory = self._job_directory(job_id)
        return [self._read_trial(path) for path in sorted(directory.glob("trial_*")) if self._trial_complete(path)]

    def get_trial(self, job_id: str, trial_number: int) -> dict[str, Any]:
        """Return one trial's parameters, result, and metrics."""
        path = self._job_directory(job_id) / f"trial_{trial_number:05d}"
        if not path.is_dir():
            raise KeyError(f"Unknown optimization trial: {trial_number}")
        return self._read_trial(path)

    def apply_trial(self, job_id: str, trial_number: int, scene_name: str) -> dict[str, Any]:
        """Apply a trial as an immutable patch and save the resulting scene."""
        trial = self.get_trial(job_id, trial_number)
        scene = self.store.load(scene_name)
        candidate = apply_parameter_patch(scene, trial["parameters"])
        report = validate_scene(candidate)
        if not report["valid"]:
            raise ValueError(f"Trial produces an invalid scene: {report['diagnostics']}")
        self.store.save(candidate)
        return candidate.to_dict()

    def compare_trials(self, job_id: str, trial_numbers: list[int]) -> list[dict[str, Any]]:
        """Return comparable objective, feasibility, runtime, and parameter values."""
        return [self.get_trial(job_id, number) for number in trial_numbers]

    @staticmethod
    def _validate_plan_for_scene(scene_payload: dict[str, Any], plan: OptimizationPlan) -> None:
        base = Scene.from_dict(scene_payload)
        specs = {item.path: item for item in plan.parameters}
        samples = {}
        for item in plan.parameters:
            if not item.enabled:
                continue
            if item.kind == "bool":
                samples[item.path] = False
            elif item.kind == "categorical":
                samples[item.path] = item.choices[0]
            elif item.kind == "int":
                samples[item.path] = int((item.lower + item.upper) // 2)
            else:
                samples[item.path] = float(item.lower + item.upper) * 0.5
        candidate = apply_parameter_patch(base, samples, specs)
        report = validate_scene(candidate)
        if not report["valid"]:
            raise ValueError(f"Optimization plan midpoint is invalid: {report['diagnostics']}")
        effective = replace(
            plan,
            parameters=activate_parameters(base, plan.parameters, optimizer=plan.optimizer),
        )
        enabled_paths = {item.path for item in plan.parameters if item.enabled}
        active_paths = {item.path for item in effective.parameters}
        inactive_paths = sorted(enabled_paths.difference(active_paths))
        if inactive_paths:
            raise ValueError(
                "Enabled optimization parameters are not active for this scene: "
                f"{inactive_paths}. Remove them or set enabled=false."
            )
        active = len(distributions_for(effective))
        limit = 10 if plan.optimizer == "cmaes" else 15
        if active > limit:
            raise ValueError(f"{plan.optimizer} supports at most {limit} active parameters, got {active}")

    def _load_plan(self, name: str) -> dict[str, Any]:
        path = self.plans_dir / name
        plan = json.loads((path / "optimization_plan.json").read_text(encoding="utf-8"))
        scene = json.loads((path / "scene.json").read_text(encoding="utf-8"))
        manifest = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
        task_path = path / "task_spec.json"
        objective_path = path / "objective_settings.json"
        return {
            "plan": plan,
            "scene_name": manifest["scene_name"],
            "base_scene": scene,
            "base_scene_hash": manifest["base_scene_hash"],
            "task_spec": json.loads(task_path.read_text(encoding="utf-8")) if task_path.exists() else None,
            "objective_settings": (
                json.loads(objective_path.read_text(encoding="utf-8")) if objective_path.exists() else None
            ),
        }

    def _job_directory(self, job_id: str) -> Path:
        with self._lock:
            return self._jobs[job_id][1]

    @staticmethod
    def _trial_complete(path: Path) -> bool:
        return all((path / name).exists() for name in ("parameters.json", "result.json", "metrics.json"))

    @staticmethod
    def _progress_tail(path: Path, limit: int = 5) -> list[dict[str, Any]]:
        progress = path / "progress.jsonl"
        if not progress.exists():
            return []
        lines = progress.read_text(encoding="utf-8").splitlines()[-limit:]
        return [json.loads(line) for line in lines if line.strip()]

    @staticmethod
    def _read_trial(path: Path) -> dict[str, Any]:
        return {
            "trial_number": int(path.name.removeprefix("trial_")),
            "parameters": json.loads((path / "parameters.json").read_text(encoding="utf-8")),
            "result": json.loads((path / "result.json").read_text(encoding="utf-8")),
            "metrics": json.loads((path / "metrics.json").read_text(encoding="utf-8")),
            "result_dir": str(path),
        }


def _hash(value: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
