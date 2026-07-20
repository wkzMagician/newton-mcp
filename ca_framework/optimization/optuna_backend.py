# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Version-contained Optuna sampler, study, and distribution integration."""

from __future__ import annotations

import math
from typing import Any

import optuna

from .model import OptimizationPlan, ParameterSpec


def create_sampler(
    kind: str,
    seed: int,
    *,
    tpe_startup_trials: int = 16,
    cmaes_startup_trials: int = 1,
) -> optuna.samplers.BaseSampler:
    """Create one of the three supported deterministic samplers."""
    if kind == "random":
        return optuna.samplers.RandomSampler(seed=seed)
    if kind == "tpe":
        return optuna.samplers.TPESampler(
            seed=seed,
            multivariate=True,
            group=True,
            constant_liar=True,
            n_startup_trials=tpe_startup_trials,
        )
    if kind == "cmaes":
        # Optuna 4.9 deprecates restart_strategy for a future major release;
        # keeping it isolated here makes that migration local.
        return optuna.samplers.CmaEsSampler(
            seed=seed,
            restart_strategy="ipop",
            n_startup_trials=cmaes_startup_trials,
        )
    raise ValueError(f"Unsupported optimizer: {kind}")


def create_study(plan: OptimizationPlan) -> optuna.study.Study:
    """Create or resume a minimizing study for :paramref:`plan`."""
    return optuna.create_study(
        study_name=plan.study_name or plan.name,
        storage=plan.storage_url,
        sampler=create_sampler(
            plan.optimizer,
            plan.seed,
            tpe_startup_trials=plan.tpe_startup_trials,
            cmaes_startup_trials=plan.cmaes_startup_trials,
        ),
        direction="minimize",
        load_if_exists=plan.storage_url is not None,
    )


def distributions_for(plan: OptimizationPlan) -> dict[str, optuna.distributions.BaseDistribution]:
    """Build fixed ask/tell distributions keyed by scene path."""
    stage = "refine" if plan.optimizer == "cmaes" else "global"
    return {
        spec.path: distribution_for(spec)
        for spec in plan.parameters
        if spec.enabled and spec.stage == stage
    }


def distribution_for(spec: ParameterSpec) -> optuna.distributions.BaseDistribution:
    """Translate one framework parameter into an Optuna distribution."""
    if spec.kind == "float":
        if spec.transform == "logit":
            return optuna.distributions.FloatDistribution(
                math.log(float(spec.lower) / (1.0 - float(spec.lower))),
                math.log(float(spec.upper) / (1.0 - float(spec.upper))),
            )
        return optuna.distributions.FloatDistribution(
            float(spec.lower), float(spec.upper), log=spec.transform == "log"
        )
    if spec.kind == "int":
        return optuna.distributions.IntDistribution(
            int(spec.lower), int(spec.upper), log=spec.transform == "log"
        )
    if spec.kind == "bool":
        return optuna.distributions.CategoricalDistribution((False, True))
    if spec.kind == "categorical":
        return optuna.distributions.CategoricalDistribution(spec.choices)
    raise ValueError(f"Unsupported parameter kind: {spec.kind}")


def decode_parameters(specs: dict[str, ParameterSpec], values: dict[str, Any]) -> dict[str, Any]:
    """Decode sampler-space values into directly applicable scene values."""
    decoded = {}
    for path, value in values.items():
        spec = specs[path]
        if spec.transform == "logit":
            decoded[path] = 1.0 / (1.0 + math.exp(-float(value)))
        else:
            decoded[path] = value
    return decoded


def encode_parameters(specs: dict[str, ParameterSpec], values: dict[str, Any]) -> dict[str, Any]:
    """Encode physical values for :meth:`optuna.study.Study.enqueue_trial`."""
    encoded = {}
    for path, value in values.items():
        spec = specs[path]
        spec.validate_value(value)
        if spec.transform == "logit":
            numeric = float(value)
            encoded[path] = math.log(numeric / (1.0 - numeric))
        else:
            encoded[path] = value
    return encoded


def best_trial_payload(study: optuna.study.Study) -> dict[str, Any] | None:
    """Return the best completed trial without raising for an empty study."""
    completed = study.get_trials(states=(optuna.trial.TrialState.COMPLETE,))
    if not completed:
        return None
    trial = study.best_trial
    return {
        "number": trial.number,
        "objective": trial.value,
        "parameters": dict(trial.params),
        "user_attrs": dict(trial.user_attrs),
    }
