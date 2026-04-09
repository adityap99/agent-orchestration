"""
agents/retrieval_grader.py — CRAG-style relevance grading node.

Implements Corrective RAG (CRAG): after each search execution, an LLM grades
every unique retrieved document as relevant / partial / irrelevant with respect
to the original research question.

Why this matters
────────────────
Without document-level grading the critic and synthesis agents receive all
retrieved results — relevant or not.  Noise dilutes LLM attention, inflates
token costs, and degrades report quality.  The grader solves this by producing
a URL → relevance_score map that ContextWindowManager uses to downweight
irrelevant documents when building LLM evidence prompts.

Design decisions
────────────────
- Irrelevant docs are NOT removed from search_results — they remain in the
  audit trail and can still inform the critic's meta-evaluation.
- Grades replace previous grades each iteration (latest-write-wins field).
- Grading is batched into a single LLM call to minimise latency and cost.
- Model: same as model_intent (claude-3-haiku tier) — cheap classification.
- Max 25 unique URLs are graded per iteration (content truncated to 400 chars).
- Any failure degrades gracefully: empty grades → ContextWindowManager falls
  back to credibility-only ranking (pre-grader behaviour).

CRAG flow:
  search_exec → retrieval_grade → tick → critic
                     ↑
          grades injected into state
          ContextWindowManager reads them when packing evidence
"""
from __future__ import annotations

import json
import time
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from src.config import COST_PER_TOKEN
from src.llm import get_llm
from src.state import AgentState, CostRecord


SYSTEM_PROMPT = """You are a document relevance grader for a research retrieval system.

Your ONLY job: decide whether each retrieved document is relevant to the research question.

Grading criteria:
  "yes"     — Document directly addresses the research question (facts, analysis, data)
  "partial" — Document is tangentially related; contains at most 1-2 useful facts
  "no"      — Document is unrelated, off-topic, spam, or an error page

Respond ONLY with this JSON object — no markdown, no explanation:
{
  "grades": [
    {"idx": <int>, "relevance": "<yes|partial|no>", "reason": "<one sentence max>"}
  ]
}

Be strict. A document that mentions the topic only in passing is "partial", not "yes".
Erring toward "no" prevents noise from degrading downstream report quality."""


def retrieval_grader_node(state: AgentState) -> dict[str, Any]:
    """
    Node: retrieval_grade

    Reads: question, search_results, config
    Writes: retrieval_grades, messages, cost_records

    Deduplicates search_results by URL, grades each unique source once,
    and stores the grades for ContextWindowManager to consume.
    """
    config   = state["config"]
    api_key  = state.get("openrouter_key", "")
    results  = state.get("search_results", [])
    question = state["question"]

    if not results:
        return {
            "retrieval_grades": [],
            "messages":         [AIMessage(content="Retrieval grader: no results to grade.", name="retrieval_grader")],
            "cost_records":     [],
        }

    # ── Deduplicate by URL; keep highest-credibility copy ─────────────────────
    seen: dict[str, tuple[Any, float]] = {}
    for r in results:
        url  = r.url  if hasattr(r, "url")               else r.get("url", "")
        cred = float(r.credibility_score if hasattr(r, "credibility_score") else r.get("credibility_score", 0.5))
        if url and (url not in seen or cred > seen[url][1]):
            seen[url] = (r, cred)

    unique = list(seen.values())[:25]  # Cap to control prompt size

    llm = get_llm(config.model_intent, temperature=0.0, api_key=api_key)

    # ── Build compact document list ───────────────────────────────────────────
    doc_lines: list[str]   = []
    idx_to_url: dict[int, str] = {}
    for i, (r, _) in enumerate(unique):
        url     = r.url     if hasattr(r, "url")     else r.get("url", "")
        content = r.content if hasattr(r, "content") else r.get("content", "")
        idx_to_url[i] = url
        doc_lines.append(
            f"[{i}] URL: {url}\n"
            f"     Snippet: {str(content)[:400]}"
        )

    user_prompt = (
        f"Research question: {question}\n\n"
        "Grade each document below:\n\n"
        + "\n\n".join(doc_lines)
    )

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]

    start = time.monotonic()
    response = llm.invoke(messages)
    latency_ms = (time.monotonic() - start) * 1000

    raw = response.content.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
    raw = raw.strip()

    grades: list[dict] = []
    try:
        data = json.loads(raw)
        for g in data.get("grades", []):
            idx = int(g.get("idx", -1))
            rel = g.get("relevance", "partial")
            url = idx_to_url.get(idx, "")
            if not url:
                continue
            score = {"yes": 1.0, "partial": 0.5, "no": 0.0}.get(rel, 0.5)
            grades.append({
                "url":       url,
                "relevance": rel,
                "score":     score,
                "reason":    g.get("reason", ""),
            })
    except Exception:
        pass  # grades stays [] — ContextWindowManager uses credibility-only fallback

    # ── Summary stats ─────────────────────────────────────────────────────────
    n_yes     = sum(1 for g in grades if g["relevance"] == "yes")
    n_partial = sum(1 for g in grades if g["relevance"] == "partial")
    n_no      = sum(1 for g in grades if g["relevance"] == "no")
    n_total   = len(grades)

    # Corrective signal: if most docs irrelevant, hint the next planner iteration
    relevance_ratio = (n_yes + 0.5 * n_partial) / n_total if n_total else 1.0

    usage    = response.response_metadata.get("token_usage", {})
    in_tok   = usage.get("prompt_tokens", 0)
    out_tok  = usage.get("completion_tokens", 0)
    rates    = COST_PER_TOKEN.get(config.model_intent, {"input": 0.0, "output": 0.0})
    cost_rec = CostRecord(
        node="retrieval_grade",
        model=config.model_intent,
        input_tokens=in_tok,
        output_tokens=out_tok,
        cost_usd=in_tok * rates["input"] + out_tok * rates["output"],
    )

    correction_note = (
        " ⚠️ Most results off-topic — next search should use more specific queries."
        if relevance_ratio < 0.4 and n_total >= 3
        else ""
    )
    summary = (
        f"CRAG graded {n_total} unique sources: "
        f"{n_yes} relevant, {n_partial} partial, {n_no} irrelevant "
        f"(relevance ratio: {relevance_ratio:.0%}).{correction_note}"
    )

    return {
        "retrieval_grades": grades,
        "messages":         [AIMessage(content=summary, name="retrieval_grader")],
        "cost_records":     [cost_rec],
    }
