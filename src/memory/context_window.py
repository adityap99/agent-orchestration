"""
memory/context_window.py — Context window management utilities.

Prevents context-overflow in long multi-iteration runs by:
  1. Deduplicating search results by URL (keeping highest-credibility copy)
  2. Sorting by credibility descending, taking top N
  3. Formatting evidence compactly for each agent's specific needs
  4. Compressing critic history to gap lists only (saves ~200–400 tokens/iter)

This is pure Python — no storage, no I/O, no dependencies beyond the standard
library. Safe to call with any mix of SearchResult Pydantic objects or plain
dicts (both formats appear due to semantic cache hits).
"""
from __future__ import annotations

from typing import Any


def _field(r: Any, key: str, default: Any = "") -> Any:
    """Extract a field from either a Pydantic model or a dict."""
    if hasattr(r, "get"):   # dict-like
        return r.get(key, default)
    return getattr(r, key, default)


class ContextWindowManager:
    """Pure-Python context packaging — no storage, no I/O."""

    @staticmethod
    def pack_for_critic(state: dict, max_results: int = 25) -> str:
        """
        Build deduplicated, credibility-sorted evidence text for the critic.

        Deduplicates by URL (keeps the highest-credibility copy), sorts
        descending, and truncates content at 500 chars per result.
        """
        results = state.get("search_results", [])

        seen: dict[str, tuple[Any, float]] = {}
        for r in results:
            url  = _field(r, "url", "")
            cred = float(_field(r, "credibility_score", 0.5))
            if url not in seen or cred > seen[url][1]:
                seen[url] = (r, cred)

        top = sorted(seen.values(), key=lambda x: x[1], reverse=True)[:max_results]

        chunks: list[str] = []
        for i, (r, _) in enumerate(top, 1):
            chunks.append(
                f"[{i}] URL: {_field(r, 'url')}\n"
                f"    Credibility: {float(_field(r, 'credibility_score', 0.5)):.2f}\n"
                f"    Query: {_field(r, 'query')}\n"
                f"    Content: {str(_field(r, 'content'))[:500]}\n"
            )
        return "\n".join(chunks) if chunks else "(No results retrieved)"

    @staticmethod
    def pack_for_synthesis(state: dict, max_results: int = 20) -> str:
        """
        Build evidence block for the synthesis agent.

        Uses a 800-char content budget per source for richer drafting context.
        """
        results = state.get("search_results", [])

        seen: dict[str, tuple[Any, float]] = {}
        for r in results:
            url  = _field(r, "url", "")
            cred = float(_field(r, "credibility_score", 0.5))
            if url not in seen or cred > seen[url][1]:
                seen[url] = (r, cred)

        top = sorted(seen.values(), key=lambda x: x[1], reverse=True)[:max_results]

        parts: list[str] = []
        for i, (r, _) in enumerate(top, 1):
            parts.append(
                f"[Source {i}]\nURL: {_field(r, 'url')}\n"
                f"Credibility: {float(_field(r, 'credibility_score', 0.5)):.2f}\n"
                f"Content: {str(_field(r, 'content'))[:800]}\n"
            )
        return "\n---\n".join(parts) if parts else "(No results retrieved)"

    @staticmethod
    def pack_for_planner(state: dict) -> str:
        """
        Build compact query history + gap summary for the planner.

        Shows executed queries and aggregated gaps from all critic evals.
        """
        results = state.get("search_results", [])
        evals   = state.get("critic_evals",   [])

        executed_queries = list(
            dict.fromkeys(
                _field(r, "query", "") for r in results
            )
        )

        # Aggregate gaps from all critic evaluations (deduplicated)
        gap_list: list[str] = []
        for ev in evals:
            if isinstance(ev, dict):
                gap_list.extend(ev.get("gaps", []))
        unique_gaps = list(dict.fromkeys(gap_list))[:6]

        lines: list[str] = []
        if executed_queries:
            lines.append("Executed queries:")
            lines.extend(f"  - {q}" for q in executed_queries if q)
        if unique_gaps:
            lines.append("\nIdentified knowledge gaps:")
            lines.extend(f"  - {g}" for g in unique_gaps)

        return "\n".join(lines) if lines else "(first iteration)"
