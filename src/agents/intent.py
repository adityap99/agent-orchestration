"""
agents/intent.py — Intent classification agent.

Parses the user's question, classifies the task type, identifies
information needs, and flags whether the question needs clarification before
a search plan can be formed.

Model: claude-3-haiku (cheap classification task, minimal reasoning)
Tools: none — pure reasoning
"""
from __future__ import annotations

import json
import time
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from src.config import COST_PER_TOKEN
from src.llm import get_llm
from src.state import AgentState, CostRecord, IntentClassification


SYSTEM_PROMPT = """You are an intent classification agent for a research system.
Your ONLY job is to parse the user's question and output a structured JSON object.

Respond ONLY with a JSON object matching this schema — no markdown, no explanation:
{
  "topic": "<the main topic being researched>",
  "task_type": "<one of: research, factcheck, summarize, other>",
  "disambiguation": "<how you interpreted the question — be specific>",
  "confidence": <float 0.0–1.0, how confident you are in the interpretation>,
  "needs_clarification": <true|false>,
  "clarification_reason": "<if needs_clarification=true, what's ambiguous>"
}

Classification guidelines:
- research: broad question requiring synthesis of multiple sources
- factcheck: a specific factual claim to verify
- summarize: user has content they want summarised
- other: anything not falling into the above

Be specific in disambiguation: if 'fusion energy' could mean nuclear fusion OR 
stellar fusion, pick the most likely interpretation and say so explicitly.
If the question is genuinely ambiguous with no clear dominant interpretation,
set needs_clarification=true."""


def intent_agent(state: AgentState) -> dict[str, Any]:
    """
    Node: intent

    Reads: question
    Writes: intent (IntentClassification), messages, cost_records
    """
    config  = state["config"]
    api_key = state.get("openrouter_key", "")
    llm     = get_llm(config.model_intent, temperature=0.0, api_key=api_key)

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"Question: {state['question']}"),
    ]

    start = time.monotonic()
    response = llm.invoke(messages)
    latency_ms = (time.monotonic() - start) * 1000

    # Parse JSON from response
    raw = response.content.strip()
    # Strip markdown code fences if present
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    try:
        data   = json.loads(raw)
        intent = IntentClassification(**data)
    except Exception as exc:
        # Fallback: assume generic research task
        intent = IntentClassification(
            topic=state["question"][:100],
            task_type="research",
            disambiguation=f"Defaulted to 'research' due to parse error: {exc}",
            confidence=0.5,
            needs_clarification=False,
        )

    # Record cost
    usage     = response.response_metadata.get("token_usage", {})
    in_tok    = usage.get("prompt_tokens", 0)
    out_tok   = usage.get("completion_tokens", 0)
    rates     = COST_PER_TOKEN.get(config.model_intent, {"input": 0.0, "output": 0.0})
    cost_rec  = CostRecord(
        node="intent",
        model=config.model_intent,
        input_tokens=in_tok,
        output_tokens=out_tok,
        cost_usd=in_tok * rates["input"] + out_tok * rates["output"],
    )

    return {
        "intent":       intent,
        "memory_hints": _load_memory_hints(intent),
        "messages":     [HumanMessage(content=f"Question: {state['question']}")],
        "cost_records": [cost_rec],
    }


def _load_memory_hints(intent: IntentClassification) -> dict:
    """Load procedural hints for the topic cluster. Returns {} on any failure."""
    try:
        from src.memory import procedural_store
        pattern = procedural_store.get_pattern(intent.task_type)
        if not pattern:
            return {}
        return {
            "topic_cluster":  intent.task_type,
            "avg_iterations": pattern["avg_iterations"],
            "common_gaps":    pattern["common_gaps"],
            "search_hints":   pattern["search_hints"],
        }
    except Exception:
        return {}
