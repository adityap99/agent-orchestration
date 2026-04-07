"""
observability.py — Structured logging and audit trail for the research agent.

Implements the 5-level trace hierarchy from the design doc:
  Session → Trace → Span → Generation → Tool Call
"""
from __future__ import annotations

import logging
import sys
from datetime import datetime
from hashlib import sha256
from typing import Any, Optional

import structlog

from src.state import ReviewDecision

# Configure structlog to output human-readable structured logs
structlog.configure(
    processors=[
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.dev.ConsoleRenderer(colors=True),
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
)

logger = structlog.get_logger("research_agent")


class AuditLogger:
    """Structured audit logging for compliance and debugging."""

    def __init__(self, run_id: str, user_id: str):
        self.run_id  = run_id
        self.user_id = user_id
        self._log    = logger.bind(run_id=run_id, user_id=user_id)

    # ── Session level ─────────────────────────────────────────────────────────

    def log_run_start(self, question: str, config: Any) -> None:
        self._log.info(
            "run_start",
            event_type="session",
            question=question[:200],
            budget_usd=getattr(config, "budget_usd", None),
            max_iterations=getattr(config, "max_iterations", None),
        )

    def log_run_end(
        self,
        outcome: str,
        total_cost_usd: float,
        total_tokens: int,
        duration_ms: float,
        review_cycles: int,
    ) -> None:
        self._log.info(
            "run_end",
            event_type="session",
            outcome=outcome,
            total_cost_usd=round(total_cost_usd, 6),
            total_tokens=total_tokens,
            duration_ms=round(duration_ms, 1),
            review_cycles=review_cycles,
        )

    # ── Span level (node execution) ───────────────────────────────────────────

    def log_node_start(self, node: str) -> None:
        self._log.info("node_start", event_type="span", node=node)

    def log_node_end(self, node: str, duration_ms: float, success: bool, error: Optional[str] = None) -> None:
        self._log.info(
            "node_end",
            event_type="span",
            node=node,
            duration_ms=round(duration_ms, 1),
            success=success,
            error=error,
        )

    # ── Generation level (LLM call) ───────────────────────────────────────────

    def log_generation(
        self,
        node: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        cost_usd: float,
        latency_ms: float,
    ) -> None:
        self._log.info(
            "generation",
            event_type="generation",
            node=node,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=round(cost_usd, 6),
            latency_ms=round(latency_ms, 1),
        )

    # ── Tool call level ───────────────────────────────────────────────────────

    def log_tool_call(
        self,
        tool_name: str,
        input_args: dict,
        output: str,
        duration_ms: float,
        success: bool,
        http_status: Optional[int] = None,
    ) -> None:
        self._log.info(
            "tool_call",
            event_type="tool_call",
            tool_name=tool_name,
            input_preview=str(input_args)[:200],
            output_hash=sha256(output.encode()).hexdigest()[:12],
            output_preview=output[:200],
            duration_ms=round(duration_ms, 1),
            success=success,
            http_status=http_status,
        )

    # ── Security events ───────────────────────────────────────────────────────

    def log_injection_attempt(self, pattern: str, source_url: str) -> None:
        self._log.warning(
            "prompt_injection_detected",
            event_type="security",
            pattern=pattern,
            source_url=source_url,
        )

    def log_hitl_decision(self, reviewer_id: str, decision: ReviewDecision) -> None:
        self._log.info(
            "hitl_decision",
            event_type="hitl",
            reviewer_id=reviewer_id,
            status=decision.status,
            reviewed_at=decision.reviewed_at.isoformat(),
            has_revision_instructions=decision.revision_instructions is not None,
        )

    def log_escalation(self, reason: str) -> None:
        self._log.warning(
            "escalation",
            event_type="escalation",
            reason=reason,
        )

    def log_budget_exceeded(self, spent: float, budget: float) -> None:
        self._log.error(
            "budget_exceeded",
            event_type="cost",
            spent_usd=round(spent, 6),
            budget_usd=round(budget, 6),
        )

    def log_injection_sanitized(self, pattern: str, source_url: str) -> None:
        self._log.warning(
            "injection_sanitized",
            event_type="security",
            pattern=pattern,
            source_url=source_url,
        )

    def log_circuit_breaker_open(self, service: str) -> None:
        self._log.error(
            "circuit_breaker_open",
            event_type="reliability",
            service=service,
        )

    def log_schema_violation(self, node: str, error: str) -> None:
        self._log.error(
            "schema_violation",
            event_type="p1_bug",
            node=node,
            error=error,
        )


# Module-level convenience logger (used by guardrails etc.)
module_logger = logger
