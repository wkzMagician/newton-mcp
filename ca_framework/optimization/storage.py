# SPDX-FileCopyrightText: Copyright (c) 2026 The Newton Developers
# SPDX-License-Identifier: Apache-2.0

"""Framework-level leases for external ask/tell simulation workers."""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class OptimizationJob:
    """Persistent ownership record for one external trial evaluation."""

    job_id: str
    study_name: str
    trial_number: int
    worker_id: str
    state: str
    created_at: float
    heartbeat_at: float
    result_dir: str


class OptimizationJobStore:
    """Small SQLite lease store independent from Optuna heartbeat support."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS optimization_jobs (
                    job_id TEXT PRIMARY KEY,
                    study_name TEXT NOT NULL,
                    trial_number INTEGER NOT NULL,
                    worker_id TEXT NOT NULL,
                    state TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    heartbeat_at REAL NOT NULL,
                    result_dir TEXT NOT NULL
                )
                """
            )

    def create(self, job: OptimizationJob) -> None:
        """Insert a newly leased job."""
        with self._connect() as connection:
            connection.execute(
                "INSERT INTO optimization_jobs VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    job.job_id,
                    job.study_name,
                    job.trial_number,
                    job.worker_id,
                    job.state,
                    job.created_at,
                    job.heartbeat_at,
                    job.result_dir,
                ),
            )

    def heartbeat(self, job_id: str, worker_id: str, *, timestamp: float | None = None) -> bool:
        """Renew a running job lease only for its current worker."""
        with self._connect() as connection:
            cursor = connection.execute(
                "UPDATE optimization_jobs SET heartbeat_at = ? WHERE job_id = ? AND worker_id = ? AND state = 'running'",
                (timestamp or time.time(), job_id, worker_id),
            )
            return cursor.rowcount == 1

    def finish(self, job_id: str, state: str) -> None:
        """Set a terminal job state."""
        if state not in {"completed", "failed", "timeout", "pruned"}:
            raise ValueError(f"Invalid terminal state: {state}")
        with self._connect() as connection:
            connection.execute("UPDATE optimization_jobs SET state = ? WHERE job_id = ?", (state, job_id))

    def stale(self, timeout_sec: float, *, now: float | None = None) -> list[OptimizationJob]:
        """Return running jobs whose framework lease has expired."""
        cutoff = (now or time.time()) - timeout_sec
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM optimization_jobs WHERE state = 'running' AND heartbeat_at < ?", (cutoff,)
            ).fetchall()
        return [OptimizationJob(*row) for row in rows]

    def get(self, job_id: str) -> OptimizationJob | None:
        """Return one job record."""
        with self._connect() as connection:
            row = connection.execute("SELECT * FROM optimization_jobs WHERE job_id = ?", (job_id,)).fetchone()
        return OptimizationJob(*row) if row else None

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path)
