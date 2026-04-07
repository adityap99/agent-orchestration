"""
agents/synthesis.py — Report drafting agent.

Writes the structured research report from validated search results.
Outputs a typed ReportSchema. Has NO tool access — pure reasoning over
provided evidence.

Model: claude-3-opus (best writing quality; cost justified by output value)
Tools: none — synthesis only, no external calls
"""
from __future__ import annotations

import json
import time
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.config import COST_PER_TOKEN
from src.llm import get_llm
from src.state import AgentState, CostRecord, ReportSchema, ReportSection


SYSTEM_PROMPT = """You are a research synthesis agent. Your job is to write a
comprehensive, well-structured research report based ONLY on the provided search results.

RULES:
1. Use ONLY information from the provided sources — do not add outside knowledge.
2. Cite every factual claim with the exact source URL it came from.
3. Prioritise high-credibility sources (credibility score ≥ 0.7).
4. If the human reviewer provided revision instructions, follow them precisely.
5. Be objective and balanced — represent multiple perspectives where they exist.

Respond ONLY with a JSON object — no markdown fences, no explanation — matching:
{
  "title": "<descriptive title for this research>",
  "summary": "<Executive summary in 2-4 sentences. MAX 1000 characters.>",
  "sections": [
    {
      "title": "<section heading>",
      "content": "<section body — detailed, well-written, 100-300 words>",
      "sources": ["<url1>", "<url2>"]
    }
  ],
  "sources": ["<all unique URLs cited in the report>"],
  "confidence": <float 0.0–1.0, your own assessment of report quality>,
  "word_count": <integer, approximate total word count>
}

Structure the report with 3-5 sections covering distinct aspects of the topic.
Each section must cite at least one source. Every URL in sources must appear in
at least one section's sources list."""


def synthesis_agent(state: AgentState) -> dict[str, Any]:
    """
    Node: draft (synthesis)

    Reads: question, search_results, messages (for revision feedback), config
    Writes: report (ReportSchema), messages, cost_records
    """
    config  = state["config"]
    api_key = state.get("openrouter_key", "")
    llm     = get_llm(config.model_synthesis, temperature=0.2, api_key=api_key)
    results = state.get("search_results", [])

    # Sort by credibility descending; take top 20 to avoid context overflow
    sorted_results = sorted(results, key=lambda r: r.credibility_score, reverse=True)[:20]

    # Build evidence block
    evidence_parts = []
    for i, r in enumerate(sorted_results, 1):
        evidence_parts.append(
            f"[Source {i}]\nURL: {r.url}\n"
            f"Credibility: {r.credibility_score:.2f}\n"
            f"Content: {r.content[:800]}\n"
        )
    evidence_text = "\n---\n".join(evidence_parts)

    # Check for revision feedback from human reviewer or critic
    revision_context = ""
    for msg in reversed(state.get("messages", [])):
        content = getattr(msg, "content", "")
        if "Revision instructions" in content or "rejected" in content.lower():
            revision_context = f"\n\nIMPORTANT — Reviewer feedback to incorporate:\n{content}"
            break

    user_prompt = f"""Research question: {state['question']}

Sources to use (sorted by credibility):
{evidence_text}
{revision_context}

Write a comprehensive research report using ONLY the sources above."""

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
        data    = json.loads(raw)
        sections = [ReportSection(**s) for s in data.get("sections", [])]
        report   = ReportSchema(
            title=data.get("title", "Research Report"),
            summary=data.get("summary", "")[:1000],
            sections=sections,
            sources=data.get("sources", []),
            confidence=float(data.get("confidence", 0.5)),
            word_count=int(data.get("word_count", 0)),
        )
    except Exception as exc:
        # Fallback: minimal valid report from raw response
        report = ReportSchema(
            title=f"Research: {state['question'][:80]}",
            summary=f"Report generated with parse error: {exc}. Raw content available.",
            sections=[ReportSection(
                title="Findings",
                content=raw[:2000],
                sources=[r.url for r in sorted_results[:3]],
            )],
            sources=[r.url for r in sorted_results[:5]],
            confidence=0.4,
            word_count=len(raw.split()),
        )

    usage    = response.response_metadata.get("token_usage", {})
    in_tok   = usage.get("prompt_tokens", 0)
    out_tok  = usage.get("completion_tokens", 0)
    rates    = COST_PER_TOKEN.get(config.model_synthesis, {"input": 0.0, "output": 0.0})
    cost_rec = CostRecord(
        node="draft",
        model=config.model_synthesis,
        input_tokens=in_tok,
        output_tokens=out_tok,
        cost_usd=in_tok * rates["input"] + out_tok * rates["output"],
    )

    return {
        "report":       report,
        "messages":     [AIMessage(content=f"Draft complete: '{report.title}' ({report.word_count} words, confidence {report.confidence:.2f})", name="synthesis")],
        "cost_records": [cost_rec],
    }
