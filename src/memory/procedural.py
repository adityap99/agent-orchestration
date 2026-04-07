"""
memory/procedural.py — SQLite-backed procedural memory.

Learns and stores per-topic-cluster search strategies:
  - Average iterations needed to reach sufficient confidence
  - Average final confidence score
  - Common knowledge gaps the critic identifies

Updated after every completed run; read by the intent agent to inform
the planner of known effective approaches for a topic type.
"""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from src.config import MEMORY_DB_PATH

_LOCK = threading.Lock()

_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS patterns (
    topic_cluster      TEXT PRIMARY KEY,
    avg_iterations     REAL DEFAULT 1.0,
    avg_confidence     REAL DEFAULT 0.5,
    common_gaps_json   TEXT DEFAULT '[]',
    search_hints_json  TEXT DEFAULT '[]',
    sample_count       INT  DEFAULT 0,
    updated_at         TEXT DEFAULT CURRENT_TIMESTAMP
);
"""


class ProceduralStore:
    """Thread-safe SQLite store for per-topic search-strategy patterns."""

    def __init__(self, db_path: str = MEMORY_DB_PATH) -> None:
        self._db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(_CREATE_TABLE)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def get_pattern(self, topic_cluster: str) -> Optional[dict]:
        """Load the strategy for a topic cluster. Returns None if no data yet."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM patterns WHERE topic_cluster = ?",
                (topic_cluster,),
            ).fetchone()
        if not row:
            return None
        result = dict(row)
        result["common_gaps"]  = json.loads(result.pop("common_gaps_json",  "[]"))
        result["search_hints"] = json.loads(result.pop("search_hints_json", "[]"))
        return result

    def upsert_pattern(
        self,
        topic_cluster: str,
        iteration_count: int,
        confidence: float,
        gaps: list[str],
    ) -> None:
        """Update (or insert) the strategy record for a topic cluster."""
        with _LOCK, self._connect() as conn:
            existing = conn.execute(
                "SELECT * FROM patterns WHERE topic_cluster = ?",
                (topic_cluster,),
            ).fetchone()

            now = datetime.now(timezone.utc).isoformat()

            if not existing:
                conn.execute(
                    """
                    INSERT INTO patterns
                      (topic_cluster, avg_iterations, avg_confidence,
                       common_gaps_json, search_hints_json, sample_count, updated_at)
                    VALUES (?, ?, ?, ?, ?, 1, ?)
                    """,
                    (
                        topic_cluster,
                        float(iteration_count),
                        float(confidence),
                        json.dumps(gaps[:5]),
                        json.dumps([]),
                        now,
                    ),
                )
            else:
                n      = existing["sample_count"]
                new_n  = n + 1
                new_ai = (existing["avg_iterations"] * n + iteration_count) / new_n
                new_ac = (existing["avg_confidence"] * n + confidence)      / new_n
                # Merge gaps: keep most recent 5 unique entries
                old_gaps = json.loads(existing["common_gaps_json"] or "[]")
                merged   = list(dict.fromkeys(gaps + old_gaps))[:5]
                conn.execute(
                    """
                    UPDATE patterns SET
                      avg_iterations   = ?,
                      avg_confidence   = ?,
                      common_gaps_json = ?,
                      sample_count     = ?,
                      updated_at       = ?
                    WHERE topic_cluster = ?
                    """,
                    (new_ai, new_ac, json.dumps(merged), new_n, now, topic_cluster),
                )
