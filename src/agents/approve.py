"""
agents/approve.py — Human-in-the-loop (HITL) approval node.

This node is the interrupt point. The graph pauses here awaiting a human
ReviewDecision to be injected into state via app.update_state().

Production phases (from design doc §7.1):
  Phase 1: Graph runs to interrupt, state checkpointed
  Phase 2: Human notified with report + context
  Phase 3: Human submits typed ReviewDecision
  Phase 4: Graph resumed — this node converts the decision to an actionable
           HumanMessage (informed retry, NOT blind retry)

Autonomy levels (§7.4):
  Level 0: Full review     — always interrupts
  Level 1: Spot check      — 20% sample reviewed
  Level 2: Confidence-gate — auto-approve above threshold
  Level 3: Exception-only  — auto-approve, only flag outliers
"""
from __future__ import annotations

import random
from datetime import datetime
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage

from src.state import AgentState, ReviewDecision


# ── Auto-approve (for autonomy levels > 0) ───────────────────────────────────

def _auto_approve(state: AgentState, reason: str) -> ReviewDecision:
    return ReviewDecision(
        status="approved",
        reviewer_id="system_auto",
        reviewed_at=datetime.utcnow(),
        revision_instructions=None,
    )


def approve_node(state: AgentState) -> dict[str, Any]:
    """
    Node: approve  (interrupt point for HITL)

    In Level 0 (full review): graph pauses here. Human must inject a
    ReviewDecision via app.update_state() before resuming.

    For higher autonomy levels: may auto-approve without human intervention.

    Reads:  review_decision (injected by human or auto-set), config, report
    Writes: review_decision (normalised), messages
    """
    config         = state["config"]
    autonomy_level = getattr(config, "autonomy_level", 0)
    report         = state.get("report")
    decision       = state.get("review_decision")

    # ── Autonomy level routing ────────────────────────────────────────────────
    if decision is None:
        if autonomy_level == 0:
            # Full review — no decision yet; human must supply one.
            # In real HITL this node should NOT be reached without a decision
            # because the interrupt fires before the node executes.
            # We produce a placeholder that routes to escalate.
            decision = ReviewDecision(
                status="needs_revision",
                reviewer_id="pending_human",
                revision_instructions="Awaiting human review. Resume after injecting ReviewDecision.",
            )
        elif autonomy_level == 1:
            # Spot check: 20% reviewed
            if random.random() < 0.20:
                decision = ReviewDecision(
                    status="needs_revision",
                    reviewer_id="pending_human",
                    revision_instructions="Spot-check review required.",
                )
            else:
                decision = _auto_approve(state, "spot_check_pass")
        elif autonomy_level == 2:
            # Confidence-gated
            conf = state.get("confidence", 0.0)
            if conf >= config.confidence_threshold:
                decision = _auto_approve(state, f"confidence_gate_pass ({conf:.2f})")
            else:
                decision = ReviewDecision(
                    status="needs_revision",
                    reviewer_id="pending_human",
                    revision_instructions=f"Confidence {conf:.2f} below threshold {config.confidence_threshold}.",
                )
        else:
            # Level 3: exception-only — always auto-approve
            decision = _auto_approve(state, "exception_only_auto")

    # ── Convert ReviewDecision to actionable HumanMessage (informed retry) ────
    if decision.status in ("rejected", "needs_revision"):
        feedback_parts = [
            f"Your previous report was {decision.status}.",
        ]
        if decision.revision_instructions:
            feedback_parts.append(f"Revision instructions: {decision.revision_instructions}")
        if decision.required_sources:
            feedback_parts.append(f"Required sources to incorporate: {decision.required_sources}")
        if decision.remove_claims:
            feedback_parts.append(f"Claims to remove or revise: {decision.remove_claims}")
        feedback_parts.append(
            "Please research further if needed and produce an improved report."
        )
        feedback_msg = HumanMessage(content="\n".join(feedback_parts))
        return {
            "review_decision": decision,
            "messages":        [feedback_msg],
        }

    # Approved
    return {
        "review_decision": decision,
        "messages": [
            AIMessage(
                content=f"Report approved by {decision.reviewer_id} at {decision.reviewed_at.isoformat()}",
                name="approve",
            )
        ],
    }


def publish_node(state: AgentState) -> dict[str, Any]:
    """
    Node: publish

    In production this would write to a database, call a webhook, etc.
    Here we emit a structured log message and update state to signal completion.
    """
    report   = state.get("report")
    decision = state.get("review_decision")
    cost     = state.get("cost_usd", 0.0)

    summary = (
        f"PUBLISHED: '{report.title if report else 'untitled'}' | "
        f"Confidence: {report.confidence if report else 0:.2f} | "
        f"Words: {report.word_count if report else 0} | "
        f"Cost: ${cost:.4f} | "
        f"Approved by: {decision.reviewer_id if decision else 'unknown'}"
    )

    return {
        "messages": [AIMessage(content=summary, name="publish")],
    }


def escalation_node(state: AgentState) -> dict[str, Any]:
    """
    Node: escalate

    Graceful failure handler. Serialises partial state, formats a human-readable
    summary, and terminates the run with a partial result and audit record.
    """
    reason = state.get("abort_reason", "Unknown escalation reason")
    cost   = state.get("cost_usd", 0.0)
    iters  = state.get("iteration_count", 0)
    results_count = len(state.get("search_results", []))

    summary = (
        f"ESCALATED: {reason} | "
        f"Partial results: {results_count} | "
        f"Iterations: {iters} | "
        f"Cost so far: ${cost:.4f}"
    )

    return {
        "messages": [AIMessage(content=summary, name="escalate")],
    }
