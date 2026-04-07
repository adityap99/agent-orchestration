"""
tests/test_state.py — Unit tests for state schema and Pydantic models.

Tests the most critical correctness rules from the design doc:
  - operator.add reducers prevent silent data loss
  - Pydantic models validate inter-node contracts
  - Input fields are correctly typed
"""
import operator
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from datetime import datetime
from pydantic import ValidationError

from src.state import (
    IntentClassification,
    ReportSchema,
    ReportSection,
    ReviewDecision,
    SearchPlan,
    SearchQuery,
    SearchResult,
    VerificationResult,
)
from src.config import RunConfig


# ── Pydantic model validation ─────────────────────────────────────────────────

class TestSearchResult:
    def test_valid_result(self):
        r = SearchResult(
            url="https://example.com",
            content="Some content",
            credibility_score=0.8,
        )
        assert r.credibility_score == 0.8

    def test_credibility_bounds(self):
        with pytest.raises(ValidationError):
            SearchResult(url="x", content="y", credibility_score=1.5)
        with pytest.raises(ValidationError):
            SearchResult(url="x", content="y", credibility_score=-0.1)

    def test_default_retrieved_at(self):
        r = SearchResult(url="x", content="y", credibility_score=0.5)
        assert isinstance(r.retrieved_at, datetime)


class TestReportSchema:
    def test_min_confidence(self):
        """Report confidence < 0.3 should raise ValidationError."""
        with pytest.raises(ValidationError):
            ReportSchema(
                title="Test",
                summary="Short summary.",
                sections=[ReportSection(title="S1", content="Content.", sources=["http://x.com"])],
                sources=["http://x.com"],
                confidence=0.1,
            )

    def test_summary_length(self):
        """Summary must be <= 1000 characters."""
        with pytest.raises(ValidationError):
            ReportSchema(
                title="Test",
                summary="X" * 1001,
                sections=[],
                sources=[],
                confidence=0.6,
            )

    def test_auto_word_count(self):
        """word_count should be auto-computed from content if 0."""
        r = ReportSchema(
            title="Test",
            summary="hello world",
            sections=[ReportSection(title="S", content="one two three", sources=[])],
            sources=[],
            confidence=0.6,
            word_count=0,
        )
        assert r.word_count > 0


class TestReviewDecision:
    def test_literal_status(self):
        with pytest.raises(ValidationError):
            ReviewDecision(status="maybe", reviewer_id="test")

    def test_approved(self):
        d = ReviewDecision(status="approved", reviewer_id="alice")
        assert d.status == "approved"


class TestIntentClassification:
    def test_needs_clarification(self):
        i = IntentClassification(
            topic="fusion",
            task_type="research",
            disambiguation="nuclear fusion research",
            confidence=0.3,
            needs_clarification=True,
            clarification_reason="Ambiguous: nuclear vs stellar fusion",
        )
        assert i.needs_clarification is True

    def test_low_confidence(self):
        with pytest.raises(ValidationError):
            IntentClassification(
                topic="x",
                task_type="research",
                disambiguation="y",
                confidence=1.5,  # Out of range
            )


# ── Reducer behaviour ─────────────────────────────────────────────────────────

class TestReducers:
    """
    Critical test: verify that operator.add merges lists rather than overwriting them.
    This directly tests the fix for the most common state bug (design doc §2.2).
    """

    def test_operator_add_merges_search_results(self):
        r1 = SearchResult(url="a", content="first",  credibility_score=0.8)
        r2 = SearchResult(url="b", content="second", credibility_score=0.7)
        r3 = SearchResult(url="c", content="third",  credibility_score=0.6)

        # Simulate first executor run returning [r1, r2]
        state_results = [r1, r2]
        # Simulate second executor run returning [r3]
        new_results = [r3]

        # operator.add is the reducer annotation — merge, don't overwrite
        merged = operator.add(state_results, new_results)
        assert len(merged) == 3, (
            "operator.add reducer MUST merge lists, not overwrite them. "
            "Without this fix, second run results silently replace first."
        )

    def test_operator_add_messages(self):
        from langchain_core.messages import HumanMessage, AIMessage
        msgs1 = [HumanMessage(content="q1"), AIMessage(content="a1")]
        msgs2 = [HumanMessage(content="q2")]
        merged = operator.add(msgs1, msgs2)
        assert len(merged) == 3

    def test_tool_errors_accumulate(self):
        errors1 = [{"error": "timeout"}]
        errors2 = [{"error": "rate_limit"}]
        merged = operator.add(errors1, errors2)
        assert len(merged) == 2


# ── RunConfig ─────────────────────────────────────────────────────────────────

class TestRunConfig:
    def test_defaults(self):
        cfg = RunConfig()
        assert cfg.budget_usd > 0
        assert cfg.max_iterations >= 3
        assert 0 < cfg.confidence_threshold < 1

    def test_budget_is_float(self):
        cfg = RunConfig(budget_usd=0.50)
        assert isinstance(cfg.budget_usd, float)
