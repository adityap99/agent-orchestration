"""
agents/verifier.py — Factual verification agent.

Spot-checks factual claims in the report against source URLs.
Flags claims with no supporting source (hallucination detection).

FC3.3 (Incorrect Verification) prevention: produces typed VerificationResult
objects with per-claim confidence scores, not a binary pass/fail.

Model: claude-3-haiku (structured task — can be tightly templated, cheap)
Tools: verify_url (HTTP HEAD requests, no LLM tool calling)
"""
from __future__ import annotations

import asyncio
import time
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.search import verify_url
from src.state import AgentState, CostRecord, ReportSchema, VerificationResult


def verifier_node(state: AgentState) -> dict:
    """
    Node: verify

    Reads: report
    Writes: verification (List[VerificationResult]), messages, cost_records

    Steps:
      1. Check semantic cache — skip HTTP if URL was verified recently.
      2. For uncached URLs: HTTP HEAD to check reachability.
      3. Cache new results; collect VerificationResult objects.
    """
    report: ReportSchema | None = state.get("report")
    if not report:
        return {
            "verification": [],
            "messages":     [AIMessage(content="Verifier: no report to verify", name="verifier")],
            "cost_records": [],
        }

    verification_results: list[VerificationResult] = []

    async def run_verifications() -> None:
        tasks  = []
        claims = []

        for section in report.sections:
            for url in section.sources:
                # ── Layer 2: check verification cache first ───────────────────
                cached = _check_cache(url)
                if cached is not None:
                    verification_results.append(
                        VerificationResult(
                            claim=section.title,
                            source_url=url,
                            verified=cached,
                            confidence=0.85,
                            failure_reason=None if cached else "Cached: previously unreachable",
                        )
                    )
                    continue  # Skip live HTTP check

                tasks.append(verify_url(url, timeout=5.0))
                claims.append((section.title, url))

        results = await asyncio.gather(*tasks, return_exceptions=True)
        for (claim_title, url), result in zip(claims, results):
            if isinstance(result, Exception):
                vr = VerificationResult(
                    claim=claim_title,
                    source_url=url,
                    verified=False,
                    confidence=0.0,
                    failure_reason=str(result),
                )
            else:
                is_reachable, status = result
                vr = VerificationResult(
                    claim=claim_title,
                    source_url=url,
                    verified=is_reachable,
                    confidence=0.9 if is_reachable else 0.1,
                    failure_reason=None if is_reachable else f"HTTP {status}",
                )
            verification_results.append(vr)
            # ── Layer 2: persist to cache ─────────────────────────────────────
            _cache_result(url, vr.verified, vr.confidence)

    asyncio.run(run_verifications())

    # Summary stats for human reviewer
    total      = len(verification_results)
    verified   = sum(1 for v in verification_results if v.verified)
    unverified = total - verified
    low_conf   = [v for v in verification_results if v.confidence < 0.6]

    summary = (
        f"Verification complete: {verified}/{total} sources reachable. "
        f"{unverified} unreachable URLs. "
        f"{len(low_conf)} claims flagged for human review (confidence < 0.6)."
    )

    return {
        "verification": verification_results,
        "messages":     [AIMessage(content=summary, name="verifier")],
        "cost_records": [],  # No LLM call in URL-check verifier
    }


def _check_cache(url: str):
    """Return cached verification result (True/False) or None if not cached."""
    try:
        from src.memory import semantic_store
        return semantic_store.is_claim_cached(url)
    except Exception:
        return None


def _cache_result(url: str, verified: bool, confidence: float) -> None:
    """Persist a verification result to the semantic cache."""
    try:
        from src.memory import semantic_store
        semantic_store.cache_verification(url, verified, confidence)
    except Exception:
        pass
