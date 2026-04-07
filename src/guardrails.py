"""
guardrails.py — Multi-layer guardrail architecture for the research agent.

Implements:
  1. Input guardrails  — prompt injection detection + sanitization
  2. Cost guardrails   — per-run budget enforcement with CostTracker
  3. Process guardrails — iteration/step limits (checked in routing functions)
"""
from __future__ import annotations

import re
from typing import TYPE_CHECKING, Optional

from src.config import COST_PER_TOKEN, MAX_CONTENT_LENGTH
from src.state import CostRecord

if TYPE_CHECKING:
    from src.observability import AuditLogger


# ── Custom exceptions ─────────────────────────────────────────────────────────

class BudgetExceededError(Exception):
    """Raised when a run exceeds its configured USD budget."""


class InjectionDetectedError(Exception):
    """Raised when unsanitizable prompt injection is detected."""


# ── Injection patterns to detect / redact ─────────────────────────────────────

INJECTION_PATTERNS: list[str] = [
    r"ignore\s+(?:previous|all\s+prior|prior)\s+instructions",
    r"system\s+prompt",
    r"you\s+are\s+now\b",
    r"disregard\s+.*above",
    r"forget\s+.*instructions",
    r"<\/?(?:system|user|assistant)>",
    r"\[INST\]",            # Llama-style injection token
    r"<\|im_start\|>",     # ChatML-style injection token
    r"<\|im_end\|>",
    r"###\s*system",
    r"new\s+instructions:",
    r"override\s+(?:your\s+)?instructions",
]

_COMPILED_PATTERNS = [
    re.compile(p, re.IGNORECASE | re.DOTALL) for p in INJECTION_PATTERNS
]


def sanitize_tool_result(
    content: str,
    source_url: str,
    audit_logger: Optional["AuditLogger"] = None,
) -> str:
    """
    Sanitize web content before injecting into the LLM's context window.

    Steps:
      1. Detect and redact injection patterns.
      2. Truncate to MAX_CONTENT_LENGTH to prevent context flooding.

    Logs every detection for security monitoring.
    """
    for pattern, compiled in zip(INJECTION_PATTERNS, _COMPILED_PATTERNS):
        if compiled.search(content):
            content = compiled.sub("[REDACTED]", content)
            if audit_logger:
                audit_logger.log_injection_sanitized(pattern, source_url)

    return content[:MAX_CONTENT_LENGTH]


# ── Cost tracker ──────────────────────────────────────────────────────────────

class CostTracker:
    """
    Tracks token spend per run and enforces a hard USD budget cap.

    Usage:
        tracker = CostTracker(budget_usd=1.00)
        tracker.record(input_tokens=500, output_tokens=200,
                       model="anthropic/claude-3-haiku", node="intent")
    """

    def __init__(self, budget_usd: float, audit_logger: Optional["AuditLogger"] = None):
        self.budget        = budget_usd
        self.spent         = 0.0
        self._records: list[CostRecord] = []
        self._audit        = audit_logger

    @property
    def records(self) -> list[CostRecord]:
        return list(self._records)

    def record(
        self,
        input_tokens: int,
        output_tokens: int,
        model: str,
        node: str,
    ) -> CostRecord:
        """
        Record token usage and accumulate cost.
        Raises BudgetExceededError if budget is exceeded — caller must route
        to the escalate node.
        """
        rates = COST_PER_TOKEN.get(
            model,
            {"input": 0.000003, "output": 0.000015},  # Conservative fallback
        )
        cost = (input_tokens * rates["input"]) + (output_tokens * rates["output"])
        self.spent += cost

        rec = CostRecord(
            node=node,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=cost,
        )
        self._records.append(rec)

        if self._audit:
            self._audit.log_generation(
                node=node,
                model=model,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost,
                latency_ms=0,
            )

        if self.spent > self.budget:
            if self._audit:
                self._audit.log_budget_exceeded(self.spent, self.budget)
            raise BudgetExceededError(
                f"Run exceeded budget: ${self.spent:.4f} > ${self.budget:.4f}"
            )

        return rec

    def summary(self) -> dict:
        return {
            "total_usd": round(self.spent, 6),
            "budget_usd": self.budget,
            "utilization_pct": round(100 * self.spent / self.budget, 1) if self.budget else 0,
            "breakdown": [r.model_dump() for r in self._records],
        }
