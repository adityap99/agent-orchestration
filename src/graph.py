"""
graph.py — Production LangGraph definition for the research agent.

Implements the complete graph topology from the design doc §3.7:
  intent → search_plan → search_exec → critic → draft → verify → approve → publish
                              ↑__________________|            |
                                                              → escalate (failures)

All routing functions are:
  - Deterministic (no LLM calls)
  - Fast (pure state field reads)
  - Well-tested (see tests/test_routing.py)

Every cycle has three termination conditions (§3.6 / FC1.1):
  1. Quality gate: confidence ≥ threshold
  2. Hard iteration cap: MAX_ITERATIONS
  3. Cost budget: budget_usd exceeded → escalate
"""
from __future__ import annotations

from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from src.agents.approve import approve_node, escalation_node, publish_node
from src.agents.critic import critic_agent
from src.agents.executor import search_executor_node
from src.agents.intent import intent_agent
from src.agents.planner import search_planner_agent
from src.agents.retrieval_grader import retrieval_grader_node
from src.agents.synthesis import synthesis_agent
from src.agents.verifier import verifier_node
from src.config import MAX_ITERATIONS, SOFT_LIMIT
from src.guardrails import BudgetExceededError, CostTracker
from src.state import AgentState


# ── Cost-tracking node wrappers ───────────────────────────────────────────────
# We wrap nodes to accumulate cost_usd in control state.
# The CostTracker is recreated from cost_records on each node call so the
# state stays serialisable (no live objects in TypedDict).

def _accumulate_cost(state: AgentState) -> float:
    """Sum all recorded costs from cost_records."""
    return sum(r.cost_usd for r in state.get("cost_records", []))


def _check_budget(state: AgentState) -> bool:
    """Return True if budget has been exceeded."""
    config = state["config"]
    spent  = _accumulate_cost(state)
    return spent >= config.budget_usd


# ── Routing functions (deterministic, pure state) ─────────────────────────────

def route_after_intent(state: AgentState) -> str:
    """
    After intent: route to search or escalate.

    Escalates if:
    - Intent confidence is very low and clarification is needed
    - Question is genuinely unanswerable
    """
    intent = state.get("intent")
    if not intent:
        return "escalate"
    if intent.needs_clarification and intent.confidence < 0.3:
        return "escalate"
    return "search"


def route_after_search(state: AgentState) -> str:
    """
    After search executor: route to critic or escalate.

    Escalates only if ALL search tiers failed (no results AND errors).
    A partial result (some results + some errors) routes to critic.
    The critic then evaluates quality with full context.
    """
    if _check_budget(state):
        return "abort"

    results = state.get("search_results", [])
    errors  = state.get("tool_errors", [])

    if not results and errors:
        # All searches failed — route to escalate
        return "abort"
    if not results:
        return "abort"
    # Even with some errors, route to critic (it handles partial evidence)
    return "ok"


def route_after_critic(state: AgentState) -> str:
    """
    Core routing function for the research loop.

    Checks (in order):
    1. Hard iteration cap  → force_draft (never inf-loop)
    2. Budget exceeded     → escalate
    3. Confidence ≥ threshold → draft_report
    4. Repeated tool errors → escalate (FC1.2)
    5. Soft limit reached  → draft_report (acceptable quality)
    6. Default             → search_planner (more research needed)
    """
    config      = state["config"]
    iteration   = state.get("iteration_count", 0)
    confidence  = state.get("confidence", 0.0)
    tool_errors = state.get("tool_errors", [])

    # 1. Hard cap — absolute maximum, no exceptions
    if iteration >= config.max_iterations:
        return "force_draft"

    # 2. Budget guard
    if _check_budget(state):
        return "escalate"

    # 3. Quality gate — confidence above threshold
    if confidence >= config.confidence_threshold:
        return "draft_report"

    # 4. FC1.2 — repeated tool failures indicate infrastructure issue
    recent_errors = [e for e in tool_errors if e.get("tier") == "all_failed"]
    if len(recent_errors) >= 3:
        return "escalate"

    # 5. Soft limit — draft with what we have after several iterations
    if iteration >= config.soft_iteration_limit:
        return "draft_report"

    # 6. Default — request more research
    return "more_research"


def route_approval(state: AgentState) -> str:
    """
    After human/auto approval: route to publish, revision, or escalate.
    """
    decision = state.get("review_decision")
    if not decision:
        return "escalate"

    status = decision.status
    if decision.escalate_reason:
        return "escalate"

    return {
        "approved":       "publish",
        "rejected":       "search_plan",   # Returns to research with feedback
        "needs_revision": "draft",         # Redraft without new search
    }.get(status, "escalate")


def increment_iteration(state: AgentState) -> dict:
    """
    Thin node that increments the iteration counter and updates cost_usd.
    Inserted between critic → search_planner to avoid mutating routing functions.
    """
    new_count = state.get("iteration_count", 0) + 1
    new_cost  = _accumulate_cost(state)
    return {
        "iteration_count": new_count,
        "cost_usd":        new_cost,
    }


def set_abort_reason_budget(state: AgentState) -> dict:
    spent = _accumulate_cost(state)
    return {"abort_reason": f"Budget exceeded: ${spent:.4f} >= ${state['config'].budget_usd:.4f}"}


def set_abort_reason_search(state: AgentState) -> dict:
    errors = state.get("tool_errors", [])
    return {"abort_reason": f"All search tiers failed. {len(errors)} errors recorded."}


def set_abort_reason_intent(state: AgentState) -> dict:
    intent = state.get("intent")
    reason = intent.clarification_reason if intent else "Intent classification failed"
    return {"abort_reason": f"Intent needs clarification: {reason}"}


# ── Graph construction ────────────────────────────────────────────────────────

def build_graph(checkpointer=None):
    """
    Build and compile the production research agent graph.

    Args:
        checkpointer: LangGraph checkpointer. Defaults to MemorySaver (dev).
                      Use PostgresSaver for production.

    Returns:
        Compiled StateGraph app.
    """
    graph = StateGraph(AgentState)

    # ── Nodes ─────────────────────────────────────────────────────────────────
    graph.add_node("intent",            intent_agent)
    graph.add_node("search_plan",       search_planner_agent)
    graph.add_node("search_exec",       search_executor_node)
    graph.add_node("retrieval_grade",   retrieval_grader_node)  # CRAG
    graph.add_node("tick",              increment_iteration)         # Counter node
    graph.add_node("critic",            critic_agent)
    graph.add_node("draft",             synthesis_agent)
    graph.add_node("verify",            verifier_node)
    graph.add_node("approve",           approve_node)
    graph.add_node("publish",           publish_node)
    graph.add_node("escalate",          escalation_node)

    # Abort helper nodes (set abort_reason before routing to escalate)
    graph.add_node("abort_budget", set_abort_reason_budget)
    graph.add_node("abort_search", set_abort_reason_search)
    graph.add_node("abort_intent", set_abort_reason_intent)

    # ── Edges ─────────────────────────────────────────────────────────────────
    graph.add_edge(START, "intent")

    graph.add_conditional_edges(
        "intent",
        route_after_intent,
        {"search": "search_plan", "escalate": "abort_intent"},
    )
    graph.add_edge("abort_intent", "escalate")

    graph.add_edge("search_plan", "search_exec")

    graph.add_conditional_edges(
        "search_exec",
        route_after_search,
        {"ok": "retrieval_grade", "fallback": "retrieval_grade", "abort": "abort_search"},
    )
    graph.add_edge("abort_search", "escalate")

    graph.add_edge("retrieval_grade", "tick")
    graph.add_edge("tick", "critic")

    graph.add_conditional_edges(
        "critic",
        route_after_critic,
        {
            "draft_report":  "draft",
            "force_draft":   "draft",
            "more_research": "search_plan",
            "escalate":      "escalate",
        },
    )

    graph.add_edge("draft", "verify")
    graph.add_edge("verify", "approve")

    graph.add_conditional_edges(
        "approve",
        route_approval,
        {
            "publish":    "publish",
            "search_plan":"search_plan",  # Rejected → back to research with feedback
            "draft":      "draft",        # Needs revision → redraft
            "escalate":   "escalate",
        },
    )

    graph.add_edge("publish",  END)
    graph.add_edge("escalate", END)

    # ── Compile ───────────────────────────────────────────────────────────────
    if checkpointer is None:
        # Explicitly register our custom Pydantic/dataclass types so LangGraph
        # checkpointing doesn't emit "Deserializing unregistered type" warnings.
        _allowed: set[tuple[str, str]] = {
            ("src.config", "RunConfig"),
            ("src.state", "SearchResult"),
            ("src.state", "SearchQuery"),
            ("src.state", "SearchPlan"),
            ("src.state", "IntentClassification"),
            ("src.state", "ReportSchema"),
            ("src.state", "ReportSection"),
            ("src.state", "VerificationResult"),
            ("src.state", "ReviewDecision"),
            ("src.state", "CostRecord"),
        }
        serde = JsonPlusSerializer(allowed_msgpack_modules=_allowed)
        checkpointer = MemorySaver(serde=serde)

    app = graph.compile(
        checkpointer=checkpointer,
        interrupt_before=["approve"],   # HITL interrupt point
    )
    return app


# Module-level compiled app (MemorySaver for dev; swap checkpoint for prod)
app = build_graph()
