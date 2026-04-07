"""
memory/episodic.py — SQLite-backed episodic memory (run history).

Persists completed research runs so the system can:
  - Provide an audit trail per user and topic
  - Support future "run history" UI features
  - Feed downstream memory layers (semantic, procedural, profile)
"""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from src.config import MEMORY_DB_PATH

_LOCK = threading.Lock()

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS runs (
    run_id                TEXT PRIMARY KEY,
    user_id               TEXT NOT NULL,
    question              TEXT NOT NULL,
    topic                 TEXT,
    task_type             TEXT,
    report_json           TEXT,
    confidence            REAL DEFAULT 0.0,
    cost_usd              REAL DEFAULT 0.0,
    iteration_count       INT  DEFAULT 0,
    outcome               TEXT,
    revision_instructions TEXT,
    created_at            TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_runs_user  ON runs(user_id);
CREATE INDEX IF NOT EXISTS idx_runs_topic ON runs(topic);
"""


class EpisodicStore:
    """Thread-safe SQLite-backed run history store."""

    def __init__(self, db_path: str = MEMORY_DB_PATH) -> None:
        self._db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.executescript(_CREATE_TABLE)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def save_run(
        self,
        run_id: str,
        user_id: str,
        state_snapshot: dict[str, Any],
        outcome: str,
    ) -> None:
        """Persist a completed run. Called from server after publish/escalate."""
        intent  = state_snapshot.get("intent")
        report  = state_snapshot.get("report")
        decision = state_snapshot.get("review_decision")

        report_json: Optional[str] = None
        if report is not None:
            try:
                report_json = report.model_dump_json()
            except Exception:
                pass

        topic     = intent.topic     if intent is not None else None
        task_type = intent.task_type if intent is not None else None
        conf      = state_snapshot.get("confidence", 0.0)
        cost      = state_snapshot.get("cost_usd", 0.0)
        iters     = state_snapshot.get("iteration_count", 0)
        rev_instr: Optional[str] = None
        if decision is not None and hasattr(decision, "revision_instructions"):
            rev_instr = decision.revision_instructions

        with _LOCK, self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO runs
                  (run_id, user_id, question, topic, task_type, report_json,
                   confidence, cost_usd, iteration_count, outcome,
                   revision_instructions, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    user_id,
                    state_snapshot.get("question", ""),
                    topic,
                    task_type,
                    report_json,
                    float(conf),
                    float(cost),
                    int(iters),
                    outcome,
                    rev_instr,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    def get_recent_runs(self, user_id: str, limit: int = 10) -> list[dict]:
        """Return the N most recent runs for a user (without full report JSON)."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT run_id, question, topic, task_type, confidence,
                       cost_usd, outcome, created_at
                FROM   runs
                WHERE  user_id = ?
                ORDER  BY created_at DESC
                LIMIT  ?
                """,
                (user_id, limit),
            ).fetchall()
        return [dict(r) for r in rows]

    def get_run(self, run_id: str) -> Optional[dict]:
        """Fetch a single run by ID (includes report JSON)."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM runs WHERE run_id = ?", (run_id,)
            ).fetchone()
        return dict(row) if row else None
