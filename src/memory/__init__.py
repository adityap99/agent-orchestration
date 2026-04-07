"""
memory — 5-layer memory architecture for the research agent.

Layers:
  1. Episodic     — run history (SQLite)
  2. Semantic     — source / report vector store (ChromaDB, optional)
  3. Procedural   — per-topic search strategy (SQLite)
  4. UserProfile  — per-user preferences (SQLite)
  5. ContextWindow — context packing utility (pure Python)

The three SQLite stores share a single database file (MEMORY_DB_PATH).
Module-level singletons are initialised once and shared across all requests.
Any initialisation failure is caught and the stores degrade gracefully.
"""
from __future__ import annotations

from src.config import MEMORY_DB_PATH, SEMANTIC_STORE_DIR
from src.memory.context_window import ContextWindowManager
from src.memory.episodic import EpisodicStore
from src.memory.procedural import ProceduralStore
from src.memory.semantic import SemanticStore
from src.memory.user_profile import UserProfileStore

# ── Module-level singletons ───────────────────────────────────────────────────

episodic_store:   EpisodicStore   = EpisodicStore(MEMORY_DB_PATH)
semantic_store:   SemanticStore   = SemanticStore(SEMANTIC_STORE_DIR)
procedural_store: ProceduralStore = ProceduralStore(MEMORY_DB_PATH)
profile_store:    UserProfileStore = UserProfileStore(MEMORY_DB_PATH)

__all__ = [
    "episodic_store",
    "semantic_store",
    "procedural_store",
    "profile_store",
    "ContextWindowManager",
]
