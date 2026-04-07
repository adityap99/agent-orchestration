"""
config.py — Runtime configuration for the research agent.
Loads from environment/.env and exposes typed settings.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from dotenv import load_dotenv

load_dotenv()


# ── Model IDs ────────────────────────────────────────────────────────────────
MODEL_INTENT    = os.getenv("MODEL_INTENT",    "anthropic/claude-3-haiku")
MODEL_PLANNER   = os.getenv("MODEL_PLANNER",   "anthropic/claude-3.7-sonnet")
MODEL_EXECUTOR  = os.getenv("MODEL_EXECUTOR",  "anthropic/claude-3-haiku")
MODEL_CRITIC    = os.getenv("MODEL_CRITIC",    "anthropic/claude-3.7-sonnet")
MODEL_SYNTHESIS = os.getenv("MODEL_SYNTHESIS", "anthropic/claude-sonnet-4.6")
MODEL_VERIFIER  = os.getenv("MODEL_VERIFIER",  "anthropic/claude-3-haiku")

# ── OpenRouter ────────────────────────────────────────────────────────────────
OPENROUTER_API_KEY  = os.getenv("OPENROUTER_API_KEY", "")
OPENROUTER_BASE_URL = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

# ── Memory stores ─────────────────────────────────────────────────────────────
MEMORY_DB_PATH     = os.getenv("MEMORY_DB_PATH",     "memory/memory.db")
SEMANTIC_STORE_DIR = os.getenv("SEMANTIC_STORE_DIR", "memory/semantic_store")

# ── Cost per token (USD) — approximate OpenRouter/Anthropic rates ─────────────
COST_PER_TOKEN: dict[str, dict[str, float]] = {
    "anthropic/claude-3-haiku":    {"input": 0.00000025, "output": 0.00000125},
    "anthropic/claude-3.5-haiku":  {"input": 0.0000008,  "output": 0.000004},
    "anthropic/claude-3.7-sonnet": {"input": 0.000003,   "output": 0.000015},
    "anthropic/claude-sonnet-4":   {"input": 0.000003,   "output": 0.000015},
    "anthropic/claude-sonnet-4.5": {"input": 0.000003,   "output": 0.000015},
    "anthropic/claude-sonnet-4.6": {"input": 0.000003,   "output": 0.000015},
    "anthropic/claude-opus-4":     {"input": 0.000015,   "output": 0.000075},
    "anthropic/claude-opus-4.5":   {"input": 0.000015,   "output": 0.000075},
    "anthropic/claude-opus-4.6":   {"input": 0.000015,   "output": 0.000075},
    # Legacy aliases kept for backward compat 
    "anthropic/claude-3.5-sonnet": {"input": 0.000003,   "output": 0.000015},
    "anthropic/claude-3-opus":     {"input": 0.000015,   "output": 0.000075},
}


@dataclass
class RunConfig:
    """Immutable per-run configuration. Set once, never modified."""
    budget_usd: float = float(os.getenv("RUN_BUDGET_USD", "1.00"))
    max_iterations: int = int(os.getenv("MAX_ITERATIONS", "5"))
    soft_iteration_limit: int = 3
    confidence_threshold: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.65"))
    max_results_per_query: int = 5
    # Autonomy level: 0=full review, 1=spot check, 2=confidence-gated, 3=exception-only
    autonomy_level: int = 0
    # Models per agent (can be overridden per-run)
    model_intent:    str = field(default_factory=lambda: MODEL_INTENT)
    model_planner:   str = field(default_factory=lambda: MODEL_PLANNER)
    model_executor:  str = field(default_factory=lambda: MODEL_EXECUTOR)
    model_critic:    str = field(default_factory=lambda: MODEL_CRITIC)
    model_synthesis: str = field(default_factory=lambda: MODEL_SYNTHESIS)
    model_verifier:  str = field(default_factory=lambda: MODEL_VERIFIER)


# Global defaults
MAX_ITERATIONS      = int(os.getenv("MAX_ITERATIONS", "5"))
SOFT_LIMIT          = 3
CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.65"))
MAX_CONTENT_LENGTH  = 4000   # Chars injected into LLM per search result
