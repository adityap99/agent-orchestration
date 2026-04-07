"""
server.py — FastAPI web server for the Research Agent.

Exposes REST + SSE endpoints that power the single-page frontend.

Architecture:
  POST /api/research          → create run, return thread_id
  GET  /api/research/{id}/events → SSE stream of all agent events
  POST /api/research/{id}/review → inject HITL ReviewDecision & resume
  GET  /api/research/{id}/status → current run status

All LangGraph execution runs in background daemon threads. An asyncio.Queue
bridges the threads to the async SSE generator.
"""
from __future__ import annotations

import asyncio
import json
import secrets
import threading
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Optional

import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel as PydanticModel

from src.config import RunConfig
from src.graph import app as graph_app
from src.state import ReviewDecision

# ── App setup ─────────────────────────────────────────────────────────────────

web = FastAPI(title="Research Agent API", version="1.0.0")

web.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Per-run state ─────────────────────────────────────────────────────────────

class RunState:
    """Carries all mutable state for one research run."""

    def __init__(
        self,
        thread_id: str,
        lg_config: dict,
        initial_state: dict,
        autonomy_level: int,
        confidence_threshold: float,
    ):
        self.thread_id            = thread_id
        self.lg_config            = lg_config
        self.initial_state        = initial_state
        self.autonomy_level       = autonomy_level
        self.confidence_threshold = confidence_threshold
        self.status               = "created"      # created | running | awaiting_review | done
        self.last_confidence      = 0.0
        # Set when SSE client connects
        self.queue: Optional[asyncio.Queue] = None
        self.loop:  Optional[asyncio.AbstractEventLoop] = None


# Global run store (in-memory for dev; swap with Redis for prod)
run_store: dict[str, RunState] = {}


# ── Serialization ─────────────────────────────────────────────────────────────

def _dump(obj: Any) -> Any:
    """Recursively serialize Pydantic models, dataclasses, datetimes."""
    if obj is None:
        return None
    if hasattr(obj, "model_dump"):          # Pydantic v2
        return _dump(obj.model_dump())
    if hasattr(obj, "__dataclass_fields__"): # dataclass (RunConfig)
        return _dump(asdict(obj))
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, list):
        return [_dump(i) for i in obj]
    if isinstance(obj, dict):
        return {k: _dump(v) for k, v in obj.items()}
    return obj


def _sse(event_type: str, payload: Any = None) -> str:
    """Format a server-sent event line."""
    data = json.dumps(
        {"type": event_type, "payload": payload or {}},
        default=str,
    )
    return f"data: {data}\n\n"


# ── Background graph runner ───────────────────────────────────────────────────

def _emit_to_queue(rs: RunState, event_type: str, payload: Any = None) -> None:
    """Thread-safe: post a formatted SSE string to the asyncio queue."""
    loop = rs.loop
    q    = rs.queue
    if loop is None or loop.is_closed() or q is None:
        return
    asyncio.run_coroutine_threadsafe(q.put(_sse(event_type, payload)), loop)


def _done(rs: RunState) -> None:
    """Signal SSE generator to close the stream."""
    loop = rs.loop
    q    = rs.queue
    if loop is None or loop.is_closed() or q is None:
        return
    asyncio.run_coroutine_threadsafe(q.put(None), loop)  # None = sentinel


def _emit_node_events(rs: RunState, node_name: str, update: dict) -> None:
    """Emit targeted, UI-friendly events based on which node just completed."""
    emit = lambda t, p=None: _emit_to_queue(rs, t, p)

    # Always emit cost update for any LLM node
    for cr in update.get("cost_records", []):
        emit("cost_update", _dump(cr))

    if node_name == "intent":
        intent = update.get("intent")
        if intent:
            emit("intent_complete", _dump(intent))

    elif node_name == "search_plan":
        plan = update.get("search_plan")
        if plan:
            emit("search_plan_complete", _dump(plan))

    elif node_name == "search_exec":
        results = update.get("search_results", [])
        errors  = update.get("tool_errors", [])
        emit("search_results_update", {
            "results":     [_dump(r) for r in results],
            "error_count": len(errors),
        })

    elif node_name == "tick":
        emit("iteration_tick", {
            "iteration_count": update.get("iteration_count", 0),
            "cost_usd":        update.get("cost_usd", 0.0),
        })

    elif node_name == "critic":
        evals = update.get("critic_evals", [])
        payload = evals[-1] if evals else {"confidence": update.get("confidence", 0.0)}
        emit("critic_complete", payload)

    elif node_name == "draft":
        report = update.get("report")
        if report:
            emit("draft_complete", _dump(report))

    elif node_name == "verify":
        verification = update.get("verification") or []
        emit("verification_complete", {"results": [_dump(v) for v in verification]})

    elif node_name == "approve":
        decision = update.get("review_decision")
        if decision:
            emit("approve_complete", _dump(decision))

    elif node_name == "publish":
        emit("published", {"outcome": "published"})

    elif node_name == "escalate":
        msgs   = update.get("messages", [])
        reason = getattr(msgs[-1], "content", "") if msgs else ""
        emit("escalated", {"reason": reason[:800]})

    elif node_name in ("abort_budget", "abort_search", "abort_intent"):
        emit("abort_detected", {"node": node_name})


def _stream_graph(initial_state: dict, rs: RunState) -> None:
    """
    Background thread: run the LangGraph agent and push events to the SSE queue.
    Handles the first phase (START → approve interrupt).
    """
    emit = lambda t, p=None: _emit_to_queue(rs, t, p)
    rs.status = "running"

    try:
        emit("run_started", {
            "thread_id":     rs.thread_id,
            "autonomy_level": rs.autonomy_level,
        })

        for chunk in graph_app.stream(initial_state, rs.lg_config, stream_mode="updates"):
            for node_name, update in chunk.items():
                if node_name.startswith("__") or not isinstance(update, dict):
                    continue  # skip __interrupt__ / __end__ tuples
                emit("node_active", {"node": node_name})
                _emit_node_events(rs, node_name, update)

        _handle_post_stream(rs)

    except Exception as exc:
        emit("error", {"message": str(exc)})
        rs.status = "done"
        _done(rs)


def _handle_post_stream(rs: RunState) -> None:
    """
    After ANY graph stream ends: check for HITL interrupt, handle auto-approve,
    or close the SSE stream.  Called from both _stream_graph and _resume_graph
    so that a HITL prompt is re-issued every time the approve node is reached.
    """
    emit = lambda t, p=None: _emit_to_queue(rs, t, p)
    graph_state = graph_app.get_state(rs.lg_config)
    if graph_state.next and "approve" in graph_state.next:
        rs.last_confidence = graph_state.values.get("confidence", 0.0)
        rs.status = "awaiting_review"
        if rs.autonomy_level >= 3:
            _auto_resume(rs, "approved", None)
        elif rs.autonomy_level == 2 and rs.last_confidence >= rs.confidence_threshold:
            _auto_resume(rs, "approved", None)
        else:
            report = graph_state.values.get("report")
            emit("hitl_required", {
                "report":         _dump(report),
                "confidence":     rs.last_confidence,
                "autonomy_level": rs.autonomy_level,
            })
            # SSE stays open — client will POST /review to resume
    else:
        rs.status = "done"
        _done(rs)


def _auto_resume(rs: RunState, status: str, instructions: Optional[str]) -> None:
    """Inject an auto ReviewDecision then resume the graph in the same thread."""
    emit = lambda t, p=None: _emit_to_queue(rs, t, p)
    decision = ReviewDecision(
        status=status,
        reviewer_id="system_auto",
        reviewed_at=datetime.now(timezone.utc),
        revision_instructions=instructions,
    )
    graph_app.update_state(rs.lg_config, {"review_decision": decision})
    emit("review_applied", {"status": status, "reviewer": "system_auto"})
    _resume_graph(rs)


def _resume_graph(rs: RunState) -> None:
    """Stream from the current checkpoint (after update_state injects decision)."""
    emit = lambda t, p=None: _emit_to_queue(rs, t, p)
    rs.status = "running"
    try:
        for chunk in graph_app.stream(None, rs.lg_config, stream_mode="updates"):
            for node_name, update in chunk.items():
                if node_name.startswith("__") or not isinstance(update, dict):
                    continue  # skip __interrupt__ / __end__ tuples
                emit("node_active", {"node": node_name})
                _emit_node_events(rs, node_name, update)
        _handle_post_stream(rs)
    except Exception as exc:
        emit("error", {"message": str(exc)})
        rs.status = "done"
        _done(rs)


# ── Request / response models ─────────────────────────────────────────────────

class StartResearchRequest(PydanticModel):
    question:          str
    openrouter_key:    str   # required — supplied by the UI, never stored server-side
    budget:            float = 2.00
    max_iters:         int   = 3
    confidence:        float = 0.65
    autonomy_level:    int   = 0
    user_id:           str   = "web_user"


class ReviewRequest(PydanticModel):
    status:                str               # "approved" | "rejected" | "needs_revision"
    revision_instructions: Optional[str]    = None
    required_sources:      Optional[list[str]] = None
    remove_claims:         Optional[list[str]] = None


# ── API routes ────────────────────────────────────────────────────────────────

@web.post("/api/research")
async def start_research(req: StartResearchRequest):
    """Start a new research run. Returns thread_id for the SSE endpoint."""
    if not req.openrouter_key or not req.openrouter_key.startswith("sk-"):
        raise HTTPException(status_code=422, detail="A valid OpenRouter API key is required.")

    ts        = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    suffix    = secrets.token_hex(4)
    thread_id = f"research:{req.user_id}:{ts}:{suffix}"

    config = RunConfig(
        budget_usd           = req.budget,
        max_iterations       = req.max_iters,
        confidence_threshold = req.confidence,
        autonomy_level       = req.autonomy_level,
    )

    initial_state = {
        "run_id":          thread_id,
        "user_id":         req.user_id,
        "question":        req.question,
        "config":          config,
        "messages":        [],
        "search_results":  [],
        "tool_errors":     [],
        "cost_records":    [],
        "critic_evals":    [],
        "intent":          None,
        "search_plan":     None,
        "report":          None,
        "verification":    None,
        "review_decision": None,
        "iteration_count": 0,
        "confidence":      0.0,
        "cost_usd":        0.0,
        "abort_reason":    None,
        "openrouter_key":  req.openrouter_key,
    }

    rs = RunState(
        thread_id            = thread_id,
        lg_config            = {"configurable": {"thread_id": thread_id}},
        initial_state        = initial_state,
        autonomy_level       = req.autonomy_level,
        confidence_threshold = req.confidence,
    )
    run_store[thread_id] = rs
    return {"thread_id": thread_id}


@web.get("/api/research/{thread_id}/events")
async def stream_events(thread_id: str):
    """
    SSE endpoint. Opens a long-lived stream and starts the graph in a
    background thread on first connection.
    """
    if thread_id not in run_store:
        raise HTTPException(status_code=404, detail="Run not found")

    rs   = run_store[thread_id]
    loop = asyncio.get_event_loop()
    q: asyncio.Queue = asyncio.Queue()

    rs.queue = q
    rs.loop  = loop

    # Start background thread (only once per run)
    thread = threading.Thread(
        target  = _stream_graph,
        args    = (rs.initial_state, rs),
        daemon  = True,
        name    = f"agent-{thread_id[:8]}",
    )
    thread.start()

    async def generate():
        # Heartbeat while waiting
        yield _sse("connected", {"thread_id": thread_id})
        while True:
            try:
                item = await asyncio.wait_for(q.get(), timeout=90.0)
                if item is None:   # sentinel → close stream
                    yield _sse("stream_complete", {})
                    break
                yield item
            except asyncio.TimeoutError:
                yield _sse("ping", {})  # keep-alive

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":    "no-cache",
            "X-Accel-Buffering": "no",
            "Connection":       "keep-alive",
        },
    )


@web.post("/api/research/{thread_id}/review")
async def submit_review(thread_id: str, req: ReviewRequest):
    """
    Inject a human ReviewDecision into the checkpointed graph and resume
    execution. The existing SSE connection receives the continued events.
    """
    if thread_id not in run_store:
        raise HTTPException(status_code=404, detail="Run not found")

    rs = run_store[thread_id]
    if rs.status != "awaiting_review":
        raise HTTPException(
            status_code=400,
            detail=f"Run is not awaiting review (status={rs.status})"
        )

    decision = ReviewDecision(
        status                = req.status,
        reviewer_id           = "human:web",
        reviewed_at           = datetime.now(timezone.utc),
        revision_instructions = req.revision_instructions,
        required_sources      = req.required_sources,
        remove_claims         = req.remove_claims,
    )
    graph_app.update_state(rs.lg_config, {"review_decision": decision})

    _emit_to_queue(rs, "review_applied", {
        "status":       req.status,
        "reviewer":     "human:web",
        "instructions": req.revision_instructions,
    })

    # Resume in a new daemon thread so the HTTP handler returns immediately
    thread = threading.Thread(
        target = _resume_graph,
        args   = (rs,),
        daemon = True,
        name   = f"resume-{thread_id[:8]}",
    )
    thread.start()

    return {"status": "resumed", "decision": req.status}


@web.get("/api/research/{thread_id}/status")
async def get_run_status(thread_id: str):
    """Quick status check (polling fallback)."""
    if thread_id not in run_store:
        raise HTTPException(status_code=404, detail="Run not found")
    rs = run_store[thread_id]
    return {
        "thread_id":  thread_id,
        "status":     rs.status,
        "confidence": rs.last_confidence,
    }


# ── Static file serving ───────────────────────────────────────────────────────

@web.get("/")
async def serve_index():
    return FileResponse("static/index.html")


web.mount("/static", StaticFiles(directory="static"), name="static")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    uvicorn.run(
        "server:web",
        host    = "0.0.0.0",
        port    = 8000,
        reload  = False,
        workers = 1,   # Single worker: MemorySaver state is in-process
    )
