"""
memory/user_profile.py — SQLite-backed user preference memory.

Stores per-user preferences that adapt across runs:
  - Learned autonomy preference (inferred from run history)
  - Topic history (20 most recent topics)
  - Revision patterns (kinds of feedback the user typically gives)

Loaded at run start; updated after run end.
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
CREATE TABLE IF NOT EXISTS profiles (
    user_id                 TEXT PRIMARY KEY,
    preferred_sources_json  TEXT DEFAULT '[]',
    excluded_sources_json   TEXT DEFAULT '[]',
    report_style            TEXT DEFAULT 'standard',
    autonomy_preference     INT  DEFAULT 0,
    revision_patterns_json  TEXT DEFAULT '[]',
    topic_history_json      TEXT DEFAULT '[]',
    run_count               INT  DEFAULT 0,
    last_active             TEXT DEFAULT CURRENT_TIMESTAMP
);
"""


class UserProfileStore:
    """Thread-safe SQLite store for per-user preference profiles."""

    def __init__(self, db_path: str = MEMORY_DB_PATH) -> None:
        self._db_path = db_path
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as conn:
            conn.execute(_CREATE_TABLE)

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self._db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def get_profile(self, user_id: str) -> Optional[dict]:
        """Load a user's profile. Returns None if this user has no history yet."""
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM profiles WHERE user_id = ?", (user_id,)
            ).fetchone()
        if not row:
            return None
        result = dict(row)
        result["preferred_sources"] = json.loads(result.pop("preferred_sources_json", "[]"))
        result["excluded_sources"]  = json.loads(result.pop("excluded_sources_json",  "[]"))
        result["revision_patterns"] = json.loads(result.pop("revision_patterns_json", "[]"))
        result["topic_history"]     = json.loads(result.pop("topic_history_json",     "[]"))
        return result

    def record_run(
        self,
        user_id: str,
        topic: Optional[str],
        autonomy_level: int,
        revision_instructions: Optional[str],
    ) -> None:
        """Update (or create) a user's profile after a completed run."""
        with _LOCK, self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM profiles WHERE user_id = ?", (user_id,)
            ).fetchone()
            now = datetime.now(timezone.utc).isoformat()

            if not row:
                topic_history = [topic] if topic else []
                rev_patterns  = [revision_instructions] if revision_instructions else []
                conn.execute(
                    """
                    INSERT INTO profiles
                      (user_id, autonomy_preference, revision_patterns_json,
                       topic_history_json, run_count, last_active)
                    VALUES (?, ?, ?, ?, 1, ?)
                    """,
                    (
                        user_id,
                        autonomy_level,
                        json.dumps(rev_patterns[:5]),
                        json.dumps(topic_history[:20]),
                        now,
                    ),
                )
            else:
                n = row["run_count"]

                # Update topic history (keep 20 most recent, deduplicated)
                old_topics = json.loads(row["topic_history_json"] or "[]")
                new_topics = (
                    list(dict.fromkeys([topic] + old_topics))[:20]
                    if topic
                    else old_topics
                )

                # Update revision patterns (keep 5 most recent non-null)
                old_rev = json.loads(row["revision_patterns_json"] or "[]")
                new_rev = (
                    list(dict.fromkeys([revision_instructions] + old_rev))[:5]
                    if revision_instructions
                    else old_rev
                )

                # Slowly adapt autonomy preference via running average
                new_auto = round((row["autonomy_preference"] * n + autonomy_level) / (n + 1))

                conn.execute(
                    """
                    UPDATE profiles SET
                      autonomy_preference    = ?,
                      revision_patterns_json = ?,
                      topic_history_json     = ?,
                      run_count              = ?,
                      last_active            = ?
                    WHERE user_id = ?
                    """,
                    (
                        new_auto,
                        json.dumps(new_rev),
                        json.dumps(new_topics),
                        n + 1,
                        now,
                        user_id,
                    ),
                )
