# Agent Orchestration — Production LangGraph Research Agent

A full-stack, production-grade AI research agent built with **LangGraph**, **FastAPI**, and **Vue 3**. Given any research question, the agent autonomously plans searches, evaluates source quality, synthesizes a structured report, and routes it through a configurable human-in-the-loop approval workflow — all streamed to the browser in real time via Server-Sent Events (SSE).

---

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Agent Graph Topology](#agent-graph-topology)
- [Feature Highlights](#feature-highlights)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the Server](#running-the-server)
- [API Reference](#api-reference)
- [Human-in-the-Loop (HITL) Workflow](#human-in-the-loop-hitl-workflow)
- [Autonomy Levels](#autonomy-levels)
- [Guardrails & Safety](#guardrails--safety)
- [Observability](#observability)
- [Running Tests](#running-tests)
- [Configuration](#configuration)
- [Security](#security)
- [Deployment Notes](#deployment-notes)

---

## Architecture Overview

```
Browser (Vue 3 SPA)
    │  POST /api/research        → create run, get thread_id
    │  GET  /api/research/{id}/events  → SSE stream of agent events
    │  POST /api/research/{id}/review  → inject human decision & resume
    ▼
FastAPI (server.py)
    │  Background thread per run
    │  asyncio.Queue bridges thread → SSE generator
    ▼
LangGraph (src/graph.py)
    │  Checkpointed state machine (MemorySaver)
    │  Interrupt-before="approve" for HITL
    ▼
Agent Nodes (src/agents/)
    │  intent → search_plan → search_exec → critic → draft → verify → approve → publish
    ▼
OpenRouter API (Claude models via langchain-openai)
```

Each research run lives in an **in-memory run store** keyed by `thread_id`. The LangGraph checkpoint (`MemorySaver`) preserves full state across the HITL pause so the graph can be resumed exactly where it left off after a human review.

---

## Agent Graph Topology

```
START
  └─► intent ──────────────────────────────────► escalate (low confidence / ambiguous)
        └─► search_plan
              └─► search_exec ──────────────────► escalate (all search tiers failed)
                    └─► critic
                          ├─► search_plan  (iterate — confidence below threshold)
                          ├─► abort_budget (cost limit exceeded)
                          ├─► abort_search (max iterations exceeded)
                          └─► draft
                                └─► verify
                                      └─► approve  ◄── HITL interrupt point
                                            ├─► publish   (approved)
                                            ├─► search_plan (needs revision)
                                            └─► escalate   (rejected)
END
```

**Termination conditions** (checked deterministically in routing functions — no LLM calls):
1. **Quality gate** — critic confidence ≥ threshold → proceed to draft
2. **Hard iteration cap** — `MAX_ITERATIONS` exceeded → abort
3. **Cost budget** — accumulated cost ≥ `budget_usd` → escalate

---

## Feature Highlights

| Feature | Details |
|---|---|
| **Real-time SSE streaming** | Every node transition, cost update, and result is pushed to the browser as a typed JSON event |
| **Configurable HITL** | Four autonomy levels from "always ask" to "fully automatic" |
| **Per-request API key** | OpenRouter key supplied by the UI — never stored server-side |
| **Prompt injection protection** | 12 regex patterns sanitize all web content before it enters the LLM context |
| **Cost guardrails** | Per-run USD budget enforced; cost tracked per node with real token counts |
| **Structured audit log** | `structlog`-powered 5-level trace hierarchy (Session → Trace → Span → Generation → Tool) |
| **Circuit breaker** | Three-tier search reliability (retry → fallback provider → escalate) |
| **LangGraph visualization** | Frontend renders live state trace + SVG topology of the graph |
| **54-test suite** | Unit coverage for graph routing, state schema, guardrails, and node logic |

---

## Tech Stack

| Layer | Technology |
|---|---|
| **Agent framework** | [LangGraph](https://github.com/langchain-ai/langgraph) ≥ 0.2 |
| **LLM provider** | [OpenRouter](https://openrouter.ai) (Anthropic Claude models) via `langchain-openai` |
| **Web framework** | [FastAPI](https://fastapi.tiangolo.com) + [uvicorn](https://www.uvicorn.org) |
| **Frontend** | Vue 3 (CDN), Tailwind CSS Play CDN, marked.js — single `static/index.html` file |
| **Search** | DuckDuckGo (`ddgs`) — free, no API key required |
| **Observability** | `structlog` structured logging |
| **Validation** | Pydantic v2 |
| **Resilience** | `tenacity` (retry), custom circuit breaker |
| **Testing** | `pytest` |

---

## Project Structure

```
agent-orchestration/
├── server.py                  # FastAPI app — REST + SSE endpoints
├── main.py                    # Standalone CLI entry point
├── requirements.txt
├── .env.example               # Copy to .env and fill in your key
├── .gitignore
│
├── src/
│   ├── config.py              # All runtime configuration (env vars, model IDs, cost table)
│   ├── state.py               # AgentState TypedDict + all Pydantic contracts
│   ├── graph.py               # LangGraph graph definition + routing functions
│   ├── llm.py                 # ChatOpenAI factory (lru_cache keyed on model+key)
│   ├── search.py              # Three-tier web search with retry + circuit breaker
│   ├── guardrails.py          # Injection detection, cost tracking, budget enforcement
│   ├── observability.py       # AuditLogger — structured logging
│   │
│   └── agents/
│       ├── intent.py          # Intent classification (claude-3-haiku)
│       ├── planner.py         # Search query planning (claude-3.7-sonnet)
│       ├── executor.py        # Pure tool execution — no LLM
│       ├── critic.py          # Quality evaluation + confidence scoring (claude-3.7-sonnet)
│       ├── synthesis.py       # Report drafting (claude-sonnet-4.6)
│       ├── verifier.py        # URL verification — no LLM
│       └── approve.py         # HITL interrupt node + auto-approve logic
│
├── static/
│   └── index.html             # Vue 3 SPA — complete frontend
│
└── tests/
    ├── test_graph.py          # Graph topology + node integration tests
    ├── test_routing.py        # Routing function unit tests
    ├── test_state.py          # State schema validation tests
    └── test_guardrails.py     # Guardrail unit tests
```

---

## Getting Started

### Prerequisites

- Python 3.11+
- An [OpenRouter](https://openrouter.ai) account and API key (`sk-or-v1-…`)
- Git

### Installation

```bash
git clone git@github.com:adityap99/agent-orchestration.git
cd agent-orchestration

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment (optional — key can also be entered via the UI)
cp .env.example .env
# Edit .env and set OPENROUTER_API_KEY=sk-or-v1-your-key-here
```

### Running the Server

```bash
source .venv/bin/activate
uvicorn server:web --host 0.0.0.0 --port 8000 --reload
```

Open **http://localhost:8000** in your browser.

> **No `.env` key needed** — you can leave `OPENROUTER_API_KEY` empty and enter your key directly in the UI. It is stored only in `localStorage` and sent with each request; the server forwards it to OpenRouter and never persists it.

---

## API Reference

### `POST /api/research`

Start a new research run.

**Request body:**
```json
{
  "question":       "What are the latest developments in fusion energy?",
  "openrouter_key": "sk-or-v1-...",
  "budget":         2.00,
  "max_iters":      3,
  "confidence":     0.65,
  "autonomy_level": 0,
  "user_id":        "web_user"
}
```

**Response:**
```json
{ "thread_id": "research:web_user:20260406_120000:a1b2c3d4" }
```

---

### `GET /api/research/{thread_id}/events`

Server-Sent Events stream. Each event is a JSON object:

```
data: {"type": "node_active",       "payload": {"node": "search_plan"}}
data: {"type": "intent_complete",   "payload": {"topic": "...", "task_type": "research", ...}}
data: {"type": "search_results_update", "payload": {"results": [...], "error_count": 0}}
data: {"type": "critic_complete",   "payload": {"confidence": 0.72, ...}}
data: {"type": "cost_update",       "payload": {"cost_usd": 0.0031, ...}}
data: {"type": "draft_complete",    "payload": {"title": "...", "sections": [...], ...}}
data: {"type": "hitl_required",     "payload": {"report": {...}, "confidence": 0.72}}
data: {"type": "published",         "payload": {"outcome": "published"}}
data: {"type": "stream_complete",   "payload": {}}
```

**All event types:** `connected`, `run_started`, `node_active`, `intent_complete`, `search_plan_complete`, `search_results_update`, `iteration_tick`, `critic_complete`, `draft_complete`, `verification_complete`, `hitl_required`, `review_applied`, `approve_complete`, `published`, `escalated`, `abort_detected`, `cost_update`, `error`, `ping`, `stream_complete`

---

### `POST /api/research/{thread_id}/review`

Inject a human review decision to resume a paused run.

**Request body:**
```json
{
  "status":                "approved",
  "revision_instructions": null,
  "required_sources":      null,
  "remove_claims":         null
}
```

`status` must be one of: `"approved"` | `"needs_revision"` | `"rejected"`

---

### `GET /api/research/{thread_id}/status`

Returns the current run status (`created` | `running` | `awaiting_review` | `done`).

---

## Human-in-the-Loop (HITL) Workflow

When autonomy level 0 is selected (or confidence is below threshold for level 2), the graph pauses at the `approve` node using LangGraph's `interrupt_before` mechanism. The SSE stream emits a `hitl_required` event containing the full draft report and confidence score.

The frontend displays a review modal where the human can:

- **Approve** — graph resumes, report is published
- **Request revision** — free-text instructions are injected as a `HumanMessage`; graph re-enters the search/draft cycle with the feedback in context
- **Reject** — run escalates with the rejection reason

The LangGraph `MemorySaver` checkpoints the full state, so the HTTP handler returns immediately and the SSE connection stays open. When `/review` is POSTed, the graph resumes in a new background thread on the same SSE stream.

---

## Autonomy Levels

| Level | Name | Behaviour |
|---|---|---|
| **0** | Full Review | Always pauses for human approval |
| **1** | Spot Check | 20% of runs sampled for review |
| **2** | Auto Approve | Auto-approves when confidence ≥ threshold; pauses otherwise |
| **3** | Always Auto | Never pauses; publishes automatically |

---

## Guardrails & Safety

### Prompt Injection Protection
All web content retrieved by the search executor is passed through `sanitize_tool_result()` before being inserted into the LLM context. Twelve regex patterns detect common injection attempts (instruction override, system prompt extraction, ChatML/Llama injection tokens, etc.). Suspicious content is redacted in-place and logged.

### Cost Guardrails
Each LLM-calling node records a `CostRecord` (model, input tokens, output tokens, cost USD) using a lookup table of OpenRouter/Anthropic rates. The routing function `_check_budget()` reads the running total from state before every search iteration. If the budget is exceeded, the run is routed to `escalate` rather than starting another search cycle.

### Input Validation
- `openrouter_key` is validated server-side (must start with `sk-`) before a run is created
- All state fields are typed via Pydantic v2 models; invalid LLM outputs raise `ValidationError` and are caught per-node

---

## Observability

`AuditLogger` (backed by `structlog`) produces structured JSON-compatible log lines for every significant event:

| Level | Events logged |
|---|---|
| Session | `run_start`, `run_end` |
| Trace | Node entry/exit with latency |
| Span | LLM invocation with token counts |
| Generation | Raw prompt + response (debug mode) |
| Tool | Search queries, URLs fetched, injection redactions |

All logs include `run_id` and `user_id` for correlation. The HITL decision is also recorded with reviewer identity and timestamp for audit trail compliance.

---

## Running Tests

```bash
source .venv/bin/activate
pytest tests/ -v
```

The test suite covers:
- **Graph routing** — all `route_after_*` functions with every state permutation
- **State schema** — Pydantic validation, reducer behavior, field constraints
- **Guardrails** — injection pattern detection, budget enforcement, sanitization edge cases
- **Node logic** — intent parsing, critic scoring, approval logic

---

## Configuration

All settings are read from environment variables (`.env` file via `python-dotenv`) at startup. The UI-supplied `openrouter_key` overrides the env var for that run only.

| Variable | Default | Description |
|---|---|---|
| `OPENROUTER_API_KEY` | *(empty)* | Fallback key if not supplied via UI |
| `OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1` | OpenRouter endpoint |
| `MODEL_INTENT` | `anthropic/claude-3-haiku` | Intent classification model |
| `MODEL_PLANNER` | `anthropic/claude-3.7-sonnet` | Search query planner model |
| `MODEL_EXECUTOR` | `anthropic/claude-3-haiku` | (Reserved — executor is tool-only) |
| `MODEL_CRITIC` | `anthropic/claude-3.7-sonnet` | Quality evaluation model |
| `MODEL_SYNTHESIS` | `anthropic/claude-sonnet-4.6` | Report drafting model |
| `MODEL_VERIFIER` | `anthropic/claude-3-haiku` | (Reserved — verifier is HTTP-only) |
| `RUN_BUDGET_USD` | `1.00` | Default per-run cost budget |
| `MAX_ITERATIONS` | `5` | Hard cap on search → critic cycles |
| `CONFIDENCE_THRESHOLD` | `0.65` | Default quality gate for auto-approve |

---

## Security

- **No secrets in source** — `.env` and `openrouter.txt` are gitignored; `.env.example` contains only placeholder values
- **API key never persisted server-side** — the key travels from the browser → server → OpenRouter only for the duration of the request; it is not logged or stored
- **Browser storage** — the key is saved to `localStorage` for session convenience; it is never sent to any third-party endpoint
- **Injection detection** — all external web content is sanitized before LLM injection
- **Input validation** — Pydantic models validate all request bodies; malformed requests return HTTP 422 before any graph execution begins

---

## Deployment Notes

For production deployment, consider:

- **Replace `MemorySaver`** with a persistent checkpointer (e.g., `langgraph-checkpoint-postgres`) so runs survive server restarts
- **Replace `run_store` dict** with Redis or a database for multi-process / multi-replica deployments
- **Add authentication** to the `/api/research` and `/api/research/{id}/review` endpoints
- **Rate-limit** the `/api/research` endpoint per user to control API spend
- **Set `CORS` origins** explicitly instead of `allow_origins=["*"]` for production
- **Use HTTPS** — the API key travels in the request body; TLS is mandatory in production
