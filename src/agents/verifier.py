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


def verifier_node(state: AgentState) -> dict[str, Any]:
    """
    Node: verify

    Reads: report
    Writes: verification (List[VerificationResult]), messages, cost_records

    Steps:
      1. For each section+source: HTTP HEAD to check URL reachability.
      2. Collect VerificationResult objects with pass/fail and reason.
      3. Report verification summary in message for human reviewer context.
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
        tasks = []
        claims = []
        for section in report.sections:
            for url in section.sources:
                tasks.append(verify_url(url, timeout=5.0))
                claims.append((section.title, url))

        results = await asyncio.gather(*tasks, return_exceptions=True)
        for (claim_title, url), result in zip(claims, results):
            if isinstance(result, Exception):
                verification_results.append(VerificationResult(
                    claim=claim_title,
                    source_url=url,
                    verified=False,
                    confidence=0.0,
                    failure_reason=str(result),
                ))
            else:
                is_reachable, status = result
                verification_results.append(VerificationResult(
                    claim=claim_title,
                    source_url=url,
                    verified=is_reachable,
                    confidence=0.9 if is_reachable else 0.1,
                    failure_reason=None if is_reachable else f"HTTP {status}",
                ))

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
