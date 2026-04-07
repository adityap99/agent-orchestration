"""
agents/critic.py — Evidence evaluation agent (quality gate).

Evaluates whether accumulated search results are sufficient to answer the
research question with the required confidence. Either approves for drafting
or requests specific follow-up queries.

Implements FC3.1 (Premature Termination) prevention with multi-factor
evaluation checklists from the design doc.

Model: claude-3.5-sonnet (quality gate — worth the cost)
Tools: none — pure reasoning
"""
from __future__ import annotations

import json
import time
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.config import COST_PER_TOKEN
from src.llm import get_llm
from src.memory.context_window import ContextWindowManager
from src.state import AgentState, CostRecord


SYSTEM_PROMPT = """You are a research quality critic. Your job is to evaluate whether
accumulated search results are sufficient to produce a high-quality research report.

You will receive the research question, all retrieved results with credibility scores,
and any prior feedback. Evaluate using this checklist:

1. SOURCE DIVERSITY: Are there at least 3 independent sources (different domains)?
2. CREDIBILITY: Is the average credibility score above 0.6?  
3. COVERAGE: Are at least 2 distinct subtopics/aspects covered?
4. RECENCY: Is there at least 1 source from the past 12 months (if applicable)?
5. DEPTH: Do the results contain enough detail to write a substantive report?

Respond ONLY with a JSON object — no markdown, no explanation:
{
  "verdict": "<approve | request_more>",
  "confidence": <float 0.0–1.0, your confidence that existing evidence is sufficient>,
  "reasoning": "<1-2 sentence explanation of your verdict>",
  "checklist": {
    "source_diversity": <true|false>,
    "credibility_ok": <true|false>,
    "coverage_ok": <true|false>,
    "has_recent_sources": <true|false>,
    "depth_ok": <true|false>
  },
  "gaps": ["<gap1>", "<gap2>"],
  "suggested_focus": "<if request_more: what area to focus the next search on>"
}

Be strict. A report built on insufficient evidence will receive poor human review.
If 3+ checklist items fail, always return request_more."""


def critic_agent(state: AgentState) -> dict[str, Any]:
    """
    Node: critic

    Reads: question, search_results, tool_errors, iteration_count, config
    Writes: confidence (control field), messages, cost_records
    """
    config  = state["config"]
    api_key = state.get("openrouter_key", "")
    llm     = get_llm(config.model_critic, temperature=0.0, api_key=api_key)
    results = state.get("search_results", [])
    errors  = state.get("tool_errors", [])

    # Build evidence summary using ContextWindowManager (dedup by URL, top 25)
    evidence_text = ContextWindowManager.pack_for_critic(state, max_results=25)

    avg_cred = (
        sum(
            r.credibility_score if hasattr(r, "credibility_score") else r.get("credibility_score", 0.5)
            for r in results
        ) / len(results)
        if results else 0.0
    )

    # Inject any known common gaps from procedural memory
    memory_gap_hint = ""
    hints = state.get("memory_hints") or {}
    if hints.get("common_gaps"):
        memory_gap_hint = (
            "\n\nKnown common gaps for this topic type (from prior runs): "
            + ", ".join(hints["common_gaps"])
        )

    user_prompt = f"""Research question: {state['question']}
Iteration: {state.get('iteration_count', 0) + 1} of {config.max_iterations}
Confidence threshold required: {config.confidence_threshold}

Search errors: {len(errors)} failures
Average source credibility: {avg_cred:.2f}
Total results retrieved: {len(results)}

Evidence:
{evidence_text}{memory_gap_hint}"""

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
        data       = json.loads(raw)
        verdict    = data.get("verdict", "request_more")
        confidence = float(data.get("confidence", 0.4))
        reasoning  = data.get("reasoning", "")
        gaps       = data.get("gaps", [])
        focus      = data.get("suggested_focus", "")
        checklist  = data.get("checklist", {})
    except Exception as exc:
        verdict    = "request_more"
        confidence = 0.4
        reasoning  = f"Critic parse error: {exc}"
        gaps       = []
        focus      = ""
        checklist  = {}

    # Build message injecting critic reasoning for next planner iteration
    feedback_msg = AIMessage(
        content=(
            f"Critic verdict: {verdict} (confidence: {confidence:.2f})\n"
            f"Reasoning: {reasoning}\n"
            + (f"Gaps: {', '.join(gaps)}\n" if gaps else "")
            + (f"Focus next search on: {focus}" if focus else "")
        ),
        name="critic",
    )

    usage    = response.response_metadata.get("token_usage", {})
    in_tok   = usage.get("prompt_tokens", 0)
    out_tok  = usage.get("completion_tokens", 0)
    rates    = COST_PER_TOKEN.get(config.model_critic, {"input": 0.0, "output": 0.0})
    cost_rec = CostRecord(
        node="critic",
        model=config.model_critic,
        input_tokens=in_tok,
        output_tokens=out_tok,
        cost_usd=in_tok * rates["input"] + out_tok * rates["output"],
    )

    critic_eval = {
        "verdict":         verdict,
        "confidence":      confidence,
        "reasoning":       reasoning,
        "checklist":       checklist,
        "gaps":            gaps,
        "suggested_focus": focus,
        "iteration":       state.get("iteration_count", 0),
    }

    return {
        "confidence":   confidence,
        "critic_evals": [critic_eval],
        "messages":     [feedback_msg],
        "cost_records": [cost_rec],
    }
