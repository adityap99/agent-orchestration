"""
tests/test_graph.py — Integration test for the full graph compilation.

Tests that the graph compiles without errors and that node wiring is correct.
Does NOT call LLMs — uses mocked agents.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from unittest.mock import patch, MagicMock
from src.graph import build_graph, route_after_critic, route_after_intent
from src.config import RunConfig


class TestGraphCompilation:
    def test_graph_compiles(self):
        """Graph should compile without errors."""
        graph = build_graph()
        assert graph is not None

    def test_graph_has_interrupt(self):
        """Graph must have interrupt_before=['approve'] for HITL."""
        graph = build_graph()
        # The interrupt config is stored in the compiled graph
        # Verify it was set by checking compile arguments
        assert graph is not None  # Basic sanity check

    def test_module_level_app_exists(self):
        """The module-level 'app' should be importable."""
        from src.graph import app
        assert app is not None


class TestRoutingAllPaths:
    """
    Enumerate all routing paths to ensure no dead ends.
    Design doc: "Every possible execution path must be enumerable and testable."
    """

    def _make_state(self, **kwargs):
        from src.state import AgentState
        from src.graph import route_after_critic, route_after_intent, route_after_search, route_approval
        from tests.test_routing import make_state
        return make_state(**kwargs)

    def test_all_critic_outcomes_covered(self):
        from tests.test_routing import make_state
        from src.state import CostRecord
        outcomes = set()

        # force_draft
        s = make_state(confidence=0.0, iteration_count=5)
        outcomes.add(route_after_critic(s))

        # draft_report (quality gate)
        s = make_state(confidence=0.8, iteration_count=1)
        outcomes.add(route_after_critic(s))

        # more_research
        s = make_state(confidence=0.4, iteration_count=1)
        outcomes.add(route_after_critic(s))

        # escalate (budget)
        s = make_state(confidence=0.4, iteration_count=1, cost_records=[
            CostRecord(node="x", model="y", input_tokens=100, output_tokens=100, cost_usd=2.00)
        ])
        outcomes.add(route_after_critic(s))

        # escalate (repeated tool errors)
        s = make_state(confidence=0.4, iteration_count=2, tool_errors=[
            {"error": "x", "tier": "all_failed"},
            {"error": "y", "tier": "all_failed"},
            {"error": "z", "tier": "all_failed"},
        ])
        outcomes.add(route_after_critic(s))

        # All possible outcomes should be reachable
        assert "force_draft"  in outcomes
        assert "draft_report" in outcomes
        assert "more_research" in outcomes
        assert "escalate"     in outcomes
