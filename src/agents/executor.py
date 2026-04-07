"""
agents/executor.py — Search execution agent.

Executes the search plan produced by the planner. For each query:
  1. Calls search_with_fallback (DuckDuckGo → Bing, with retry)
  2. Sanitizes results for prompt injection
  3. Scores source credibility
  4. Returns typed SearchResult objects

All tool failures are captured to state['tool_errors'] rather than thrown,
so the critic receives full context about evidence quality.

Model: claude-3-haiku (mostly tool calls, minimal reasoning)
Tools: search_with_fallback
"""
from __future__ import annotations

import asyncio
import time
from typing import Any

from langchain_core.messages import AIMessage

from src.search import SearchUnavailableError, search_with_fallback
from src.state import AgentState, CostRecord, SearchResult


def search_executor_node(state: AgentState) -> dict[str, Any]:
    """
    Node: search_exec

    Reads: search_plan, config
    Writes: search_results (append), tool_errors (append), messages, cost_records

    Note: This node does NOT call the LLM — it is pure tool execution.
    The 'executor' model is used only if we add LLM-based result scoring later.
    """
    plan         = state.get("search_plan")
    config       = state["config"]
    new_results: list[SearchResult] = []
    new_errors: list[dict]          = []
    used_fallback                   = False

    if not plan or not plan.queries:
        return {
            "search_results": [],
            "tool_errors":    [{"node": "search_exec", "error": "No search plan provided"}],
            "messages":       [AIMessage(content="Search executor: no plan to execute", name="executor")],
            "cost_records":   [],
        }

    # Sort queries by priority descending
    sorted_queries = sorted(plan.queries, key=lambda q: q.priority, reverse=True)

    # ── Layer 2: pre-fill from semantic cache ─────────────────────────────────
    cached_results = _fetch_cached_sources(sorted_queries, plan.max_results_per_query)

    async def run_all() -> None:
        nonlocal used_fallback
        for sq in sorted_queries:
            start_ms = time.monotonic() * 1000
            try:
                results, is_fb = await search_with_fallback(
                    query=sq.query,
                    max_results=plan.max_results_per_query,
                )
                if is_fb:
                    used_fallback = True
                new_results.extend(results)
            except SearchUnavailableError as exc:
                new_errors.append({
                    "node":    "search_exec",
                    "query":   sq.query,
                    "error":   str(exc),
                    "tier":    "all_failed",
                })
            except Exception as exc:
                new_errors.append({
                    "node":  "search_exec",
                    "query": sq.query,
                    "error": str(exc),
                })

    asyncio.run(run_all())

    # Merge cached + fresh results (cached appear first, deduplicated by URL at critic)
    all_results = cached_results + new_results

    # Build summary message for critic context
    summary_parts = [
        f"Search complete: {len(new_results)} live results + {len(cached_results)} cached, "
        f"from {len(sorted_queries)} queries."
    ]
    if used_fallback:
        summary_parts.append("Note: used fallback search provider for some queries.")
    if new_errors:
        summary_parts.append(f"Errors: {len(new_errors)} queries failed.")

    avg_cred = (
        sum(r.credibility_score for r in new_results) / len(new_results)
        if new_results else 0.0
    )
    summary_parts.append(f"Average source credibility: {avg_cred:.2f}")

    return {
        "search_results": all_results,
        "tool_errors":    new_errors,
        "messages":       [AIMessage(content=" ".join(summary_parts), name="executor")],
        "cost_records":   [],  # No LLM call in executor
    }


def _fetch_cached_sources(sorted_queries: list, max_per_query: int) -> list[SearchResult]:
    """Query semantic memory for cached source chunks. Returns [] on any failure."""
    try:
        from src.memory import semantic_store
        cached: list[SearchResult] = []
        seen_urls: set[str] = set()
        for sq in sorted_queries:
            for hit in semantic_store.search_sources(sq.query, n=max_per_query):
                url = hit.get("url", "")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    try:
                        cached.append(
                            SearchResult(
                                url=url,
                                content=hit["content"],
                                credibility_score=float(hit.get("credibility_score", 0.5)),
                                query=hit.get("query", sq.query),
                            )
                        )
                    except Exception:
                        pass
        return cached
    except Exception:
        return []
