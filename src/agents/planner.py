"""
agents/planner.py — Search planning agent.

Decides which search queries to run, in what order, with what priority.
Outputs a typed SearchPlan. Checks prior query history to avoid near-duplicate
queries (FC3.2 — Step Repetition prevention).

Model: claude-3.5-sonnet (needs good reasoning to choose effective queries)
Tools: none — pure reasoning
"""
from __future__ import annotations

import json
import time
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.config import COST_PER_TOKEN
from src.llm import get_llm
from src.state import AgentState, CostRecord, SearchPlan, SearchQuery


SYSTEM_PROMPT = """You are a search planning agent for a research system.
Your job is to produce an optimal set of search queries to answer the user's research question.

You will receive:
  - The original research question
  - The topic and disambiguation from the intent agent
  - A history of already-executed queries (DO NOT repeat them or produce near-duplicates)
  - Feedback from the critic agent (if this is a refinement iteration)

Your output MUST be a JSON object matching this schema — no markdown, no explanation:
{
  "queries": [
    {
      "query": "<the exact search string to use>",
      "rationale": "<why this query will help answer the question>",
      "priority": <integer 1–5, where 5=highest priority>
    }
  ],
  "max_results_per_query": <integer 1–10>
}

RULES:
1. Generate 2–5 queries. More is not better — targeted queries outperform broad ones.
2. Each query must be MEANINGFULLY different from all prior queries. Do not rephrase.
3. Vary query types: some factual, some for recent developments, some for expert analysis.
4. Higher priority queries should be more directly relevant to the core question.
5. If the critic provided specific areas to investigate, generate queries targeting those gaps.
6. Prefer authoritative sources: target government, academic, or institutional content.

BAD (near-duplicate): 'fusion energy research' → 'fusion energy studies' → 'research on fusion'
GOOD: 'ITER fusion reactor progress 2025' → 'private fusion companies funding 2025' → 'nuclear fusion plasma confinement breakthrough'"""


def search_planner_agent(state: AgentState) -> dict[str, Any]:
    """
    Node: search_plan

    Reads: question, intent, messages (for prior query history)
    Writes: search_plan, messages, cost_records
    """
    config = state["config"]
    api_key = state.get("openrouter_key", "")
    llm     = get_llm(config.model_planner, temperature=0.1, api_key=api_key)

    # Build context of already-executed queries from search_results
    executed_queries = list({r.query for r in state.get("search_results", []) if r.query})

    # Pull the latest critic feedback from messages (if any)
    critic_feedback = ""
    for msg in reversed(state.get("messages", [])):
        name    = getattr(msg, "name", None) or ""
        content = getattr(msg, "content", "") or ""
        if "critic" in name.lower():
            critic_feedback = f"\nCritic feedback:\n{content}"
            break
        # Also look for HumanMessage injected feedback
        if "Revision instructions" in content:
            critic_feedback = f"\nRevision context:\n{content}"
            break

    user_prompt = f"""Research question: {state['question']}
Topic: {state['intent'].topic if state.get('intent') else 'unknown'}
Interpretation: {state['intent'].disambiguation if state.get('intent') else 'generic research'}
Iteration: {state.get('iteration_count', 0) + 1}

Already-executed queries (DO NOT repeat or rephrase these):
{chr(10).join(f'  - {q}' for q in executed_queries) if executed_queries else '  (none yet — this is the first iteration)'}
{critic_feedback}

Generate the next set of search queries."""

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]

    start    = time.monotonic()
    response = llm.invoke(messages)
    latency_ms = (time.monotonic() - start) * 1000

    raw = response.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    try:
        data = json.loads(raw)
        plan = SearchPlan(
            queries=[SearchQuery(**q) for q in data["queries"]],
            max_results_per_query=min(
                data.get("max_results_per_query", config.max_results_per_query),
                config.max_results_per_query,
            ),
        )
    except Exception as exc:
        # Fallback: single broad query
        plan = SearchPlan(
            queries=[SearchQuery(
                query=state["question"][:100],
                rationale=f"Fallback query due to parse error: {exc}",
                priority=3,
            )],
            max_results_per_query=config.max_results_per_query,
        )

    usage    = response.response_metadata.get("token_usage", {})
    in_tok   = usage.get("prompt_tokens", 0)
    out_tok  = usage.get("completion_tokens", 0)
    rates    = COST_PER_TOKEN.get(config.model_planner, {"input": 0.0, "output": 0.0})
    cost_rec = CostRecord(
        node="search_plan",
        model=config.model_planner,
        input_tokens=in_tok,
        output_tokens=out_tok,
        cost_usd=in_tok * rates["input"] + out_tok * rates["output"],
    )

    return {
        "search_plan":  plan,
        "messages":     [AIMessage(content=f"Search plan generated: {len(plan.queries)} queries", name="planner")],
        "cost_records": [cost_rec],
    }
