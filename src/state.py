"""
state.py — Production state schema for the research agent.

Organised into four field categories as per the blackboard pattern:
  • Input fields    — immutable after init
  • Working memory  — append-only (operator.add reducers)
  • Output fields   — typed Pydantic models, latest-write wins
  • Control fields  — routing signals, latest-write wins
"""
from __future__ import annotations

import operator
from datetime import datetime
from typing import Annotated, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator
from typing_extensions import TypedDict

from src.config import RunConfig


# ── Pydantic contracts between nodes ─────────────────────────────────────────

class SearchQuery(BaseModel):
    query:     str
    rationale: str
    priority:  int = Field(ge=1, le=5)


class SearchPlan(BaseModel):
    queries:               List[SearchQuery]
    max_results_per_query: int = Field(default=5, ge=1, le=20)


class SearchResult(BaseModel):
    url:               str
    content:           str
    credibility_score: float = Field(ge=0.0, le=1.0)
    retrieved_at:      datetime = Field(default_factory=datetime.utcnow)
    query:             str = ""   # Which query produced this result


class IntentClassification(BaseModel):
    topic:              str
    task_type:          Literal["research", "factcheck", "summarize", "other"]
    disambiguation:     str       # How we interpreted the question
    confidence:         float = Field(ge=0.0, le=1.0)
    needs_clarification: bool = False
    clarification_reason: Optional[str] = None


class ReportSection(BaseModel):
    title:   str
    content: str
    sources: List[str]


class ReportSchema(BaseModel):
    title:      str
    summary:    str = Field(max_length=1000)
    sections:   List[ReportSection]
    sources:    List[str]
    confidence: float = Field(ge=0.0, le=1.0)
    word_count: int = 0

    @field_validator("confidence")
    @classmethod
    def confidence_above_threshold(cls, v: float) -> float:
        if v < 0.3:
            raise ValueError(f"Confidence {v:.2f} too low to publish (min 0.30)")
        return v

    def model_post_init(self, __context: object) -> None:
        # Auto-compute word count if not provided
        if self.word_count == 0:
            all_text = self.summary + " ".join(
                s.content for s in self.sections
            )
            object.__setattr__(self, "word_count", len(all_text.split()))


class VerificationResult(BaseModel):
    claim:          str
    source_url:     str
    verified:       bool
    confidence:     float = Field(ge=0.0, le=1.0)
    failure_reason: Optional[str] = None


class ReviewDecision(BaseModel):
    status:               Literal["approved", "rejected", "needs_revision"]
    reviewer_id:          str
    reviewed_at:          datetime = Field(default_factory=datetime.utcnow)
    revision_instructions: Optional[str]       = None
    required_sources:     Optional[List[str]]  = None
    remove_claims:        Optional[List[str]]  = None
    confidence_override:  Optional[float]      = None
    escalate_reason:      Optional[str]        = None


class CostRecord(BaseModel):
    node:          str
    model:         str
    input_tokens:  int
    output_tokens: int
    cost_usd:      float


# ── Main agent state (TypedDict with reducer annotations) ─────────────────────

class AgentState(TypedDict):
    # ── INPUT (immutable after init) ──────────────────────────────────────────
    run_id:   str
    user_id:  str
    question: str
    config:   RunConfig

    # ── WORKING MEMORY (append-only — operator.add merges across node returns) ─
    messages:       Annotated[list, operator.add]
    search_results: Annotated[List[SearchResult], operator.add]
    tool_errors:    Annotated[list, operator.add]
    cost_records:   Annotated[List[CostRecord], operator.add]
    critic_evals:   Annotated[list, operator.add]  # structured critic JSON per iteration

    # ── OUTPUT (latest-write wins; always Pydantic models, never raw strings) ──
    intent:          Optional[IntentClassification]
    search_plan:     Optional[SearchPlan]
    report:          Optional[ReportSchema]
    verification:    Optional[List[VerificationResult]]
    review_decision: Optional[ReviewDecision]

    # ── CONTROL (routing signals) ─────────────────────────────────────────────
    iteration_count:  int
    confidence:       float
    cost_usd:         float
    abort_reason:     Optional[str]
    openrouter_key:   str   # per-request key supplied by the UI (never stored long-term)

    # ── MEMORY (populated at run start; read-only inside nodes) ──────────────
    memory_hints:  Optional[dict]  # procedural + semantic hints for planner/critic
    user_profile:  Optional[dict]  # loaded user preference profile
