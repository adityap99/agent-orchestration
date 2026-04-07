"""
tests/test_routing.py — Unit tests for deterministic routing functions.

Design doc §3.6: Routing functions must be deterministic, testable in isolation,
and never call LLMs. Every execution path must be enumerable.

"If you cannot enumerate every path, the graph is not production-ready."
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from datetime import datetime

from src.config import RunConfig
from src.graph import (
    route_after_critic,
    route_after_intent,
    route_after_search,
    route_approval,
)
from src.state import (
    AgentState,
    CostRecord,
    IntentClassification,
    ReviewDecision,
    SearchResult,
)


def make_state(**overrides) -> AgentState:
    """Build a minimal valid AgentState for routing function tests."""
    base = {
        "run_id":          "test:user:001",
        "user_id":         "user",
        "question":        "What are the latest developments in fusion energy?",
        "config":          RunConfig(budget_usd=1.00, max_iterations=5, confidence_threshold=0.65),
        "messages":        [],
        "search_results":  [],
        "tool_errors":     [],
        "cost_records":    [],
        "intent":          None,
        "search_plan":     None,
        "report":          None,
        "verification":    None,
        "review_decision": None,
        "iteration_count": 0,
        "confidence":      0.0,
        "cost_usd":        0.0,
        "abort_reason":    None,
    }
    base.update(overrides)
    return base  # type: ignore[return-value]


def make_search_result(url: str = "http://example.com") -> SearchResult:
    return SearchResult(url=url, content="content", credibility_score=0.7)


# ── route_after_intent ────────────────────────────────────────────────────────

class TestRouteAfterIntent:
    def test_no_intent_escalates(self):
        state = make_state(intent=None)
        assert route_after_intent(state) == "escalate"

    def test_needs_clarification_and_low_confidence_escalates(self):
        intent = IntentClassification(
            topic="fusion",
            task_type="research",
            disambiguation="unclear",
            confidence=0.2,
            needs_clarification=True,
            clarification_reason="Too ambiguous",
        )
        state = make_state(intent=intent)
        assert route_after_intent(state) == "escalate"

    def test_clear_intent_routes_to_search(self):
        intent = IntentClassification(
            topic="fusion energy",
            task_type="research",
            disambiguation="nuclear fusion research",
            confidence=0.9,
        )
        state = make_state(intent=intent)
        assert route_after_intent(state) == "search"

    def test_needs_clarification_but_high_confidence_proceeds(self):
        """High-confidence interpretation proceeds even with clarification flag."""
        intent = IntentClassification(
            topic="fusion",
            task_type="research",
            disambiguation="most likely nuclear fusion",
            confidence=0.8,
            needs_clarification=True,
        )
        state = make_state(intent=intent)
        assert route_after_intent(state) == "search"


# ── route_after_search ────────────────────────────────────────────────────────

class TestRouteAfterSearch:
    def test_no_results_and_errors_aborts(self):
        state = make_state(
            search_results=[],
            tool_errors=[{"error": "Unavailable"}],
        )
        assert route_after_search(state) == "abort"

    def test_no_results_no_errors_aborts(self):
        state = make_state(search_results=[], tool_errors=[])
        assert route_after_search(state) == "abort"

    def test_results_with_errors_routes_ok(self):
        """Partial success (some results + some errors) routes to critic."""
        state = make_state(
            search_results=[make_search_result()],
            tool_errors=[{"error": "one failed"}],
        )
        assert route_after_search(state) == "ok"

    def test_results_routes_ok(self):
        state = make_state(search_results=[make_search_result()])
        assert route_after_search(state) == "ok"

    def test_budget_exceeded_routes_abort(self):
        """When budget is exceeded, route to abort even with results."""
        state = make_state(
            search_results=[make_search_result()],
            # Exceed budget via cost_records
            cost_records=[CostRecord(
                node="search_exec", model="x",
                input_tokens=1000000, output_tokens=1000000,
                cost_usd=2.00,  # Exceeds $1.00 budget
            )],
        )
        assert route_after_search(state) == "abort"


# ── route_after_critic ────────────────────────────────────────────────────────

class TestRouteAfterCritic:
    def test_quality_gate_passes(self):
        """Confidence above threshold → draft_report."""
        state = make_state(confidence=0.80, iteration_count=1)
        assert route_after_critic(state) == "draft_report"

    def test_confidence_exactly_at_threshold(self):
        state = make_state(confidence=0.65, iteration_count=1)
        assert route_after_critic(state) == "draft_report"

    def test_below_threshold_continues_research(self):
        state = make_state(confidence=0.50, iteration_count=1)
        assert route_after_critic(state) == "more_research"

    def test_hard_cap_forces_draft(self):
        """At MAX_ITERATIONS, always force draft regardless of confidence."""
        state = make_state(confidence=0.30, iteration_count=5)
        assert route_after_critic(state) == "force_draft"

    def test_soft_limit_drafts_at_threshold(self):
        """After SOFT_LIMIT iterations with low confidence, still draft."""
        state = make_state(confidence=0.45, iteration_count=3)
        assert route_after_critic(state) == "draft_report"

    def test_budget_exceeded_escalates(self):
        state = make_state(
            confidence=0.30,
            iteration_count=1,
            cost_records=[CostRecord(
                node="critic", model="x",
                input_tokens=100000, output_tokens=100000,
                cost_usd=2.00,
            )],
        )
        assert route_after_critic(state) == "escalate"

    def test_repeated_tool_errors_escalate(self):
        """FC1.2: repeated tool failures should trigger escalation."""
        state = make_state(
            confidence=0.30,
            iteration_count=2,
            tool_errors=[
                {"error": "x", "tier": "all_failed"},
                {"error": "y", "tier": "all_failed"},
                {"error": "z", "tier": "all_failed"},
            ],
        )
        assert route_after_critic(state) == "escalate"

    def test_no_infinite_loop(self):
        """
        Critical: every call to route_after_critic with iteration_count >= MAX_ITERATIONS
        must return 'force_draft'. This prevents infinite loops (FC1.1).
        """
        for iters in range(5, 10):
            state = make_state(confidence=0.0, iteration_count=iters)
            result = route_after_critic(state)
            assert result == "force_draft", (
                f"At iteration {iters} with 0.0 confidence, "
                f"routing returned '{result}' instead of 'force_draft'. "
                "Infinite loop vulnerability!"
            )


# ── route_approval ────────────────────────────────────────────────────────────

class TestRouteApproval:
    def test_approved_publishes(self):
        state = make_state(review_decision=ReviewDecision(
            status="approved", reviewer_id="alice",
        ))
        assert route_approval(state) == "publish"

    def test_rejected_returns_to_search(self):
        state = make_state(review_decision=ReviewDecision(
            status="rejected",
            reviewer_id="bob",
            revision_instructions="Need more sources on ITER.",
        ))
        assert route_approval(state) == "search_plan"

    def test_needs_revision_redrafts(self):
        state = make_state(review_decision=ReviewDecision(
            status="needs_revision",
            reviewer_id="carol",
            revision_instructions="Fix the intro paragraph.",
        ))
        assert route_approval(state) == "draft"

    def test_no_decision_escalates(self):
        state = make_state(review_decision=None)
        assert route_approval(state) == "escalate"

    def test_escalate_reason_escalates(self):
        state = make_state(review_decision=ReviewDecision(
            status="approved",
            reviewer_id="dave",
            escalate_reason="Flagged for compliance review",
        ))
        assert route_approval(state) == "escalate"
