"""
main.py — Production Research Agent entry point.

Provides two modes:
  1. Interactive CLI: run a full research session with HITL review
  2. Single-shot: run with auto-approval (autonomy_level=2) for testing

Usage:
  # Interactive (full HITL review):
  python main.py "What are the latest developments in fusion energy?"

  # Auto-approve mode (for testing):
  python main.py "What are the latest developments in fusion energy?" --auto

  # Resume a paused run (after human review injection):
  python main.py --resume <thread_id> --decision approved
"""
from __future__ import annotations

import argparse
import json
import secrets
import sys
import uuid
from datetime import datetime, timezone
from typing import Optional

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from src.config import RunConfig
from src.graph import app, build_graph
from src.observability import AuditLogger
from src.state import AgentState, ReviewDecision

console = Console()


# ── Thread ID factory ─────────────────────────────────────────────────────────

def make_thread_id(user_id: str, task_type: str = "research") -> str:
    """Structured thread ID encoding task type, user, timestamp, and random suffix."""
    ts     = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    suffix = secrets.token_hex(4)
    return f"{task_type}:{user_id}:{ts}:{suffix}"


# ── Display helpers ───────────────────────────────────────────────────────────

def display_report(state: dict) -> None:
    """Render the research report in the terminal."""
    report = state.get("report")
    if not report:
        console.print("[yellow]No report generated.[/yellow]")
        return

    console.print()
    console.print(Panel(
        f"[bold cyan]{report.title}[/bold cyan]",
        subtitle=f"Confidence: {report.confidence:.0%} | Words: {report.word_count}",
        border_style="cyan",
    ))
    console.print()
    console.print(Panel(report.summary, title="Summary", border_style="green"))

    for section in report.sections:
        console.print()
        console.print(f"[bold]{section.title}[/bold]")
        console.print(section.content)
        if section.sources:
            console.print(f"[dim]Sources: {', '.join(section.sources[:3])}[/dim]")

    # Cost summary
    cost_usd = state.get("cost_usd", 0.0)
    iters    = state.get("iteration_count", 0)
    console.print()
    console.print(f"[dim]Cost: ${cost_usd:.4f} | Iterations: {iters}[/dim]")


def display_verification(state: dict) -> None:
    """Show verification results table."""
    verification = state.get("verification")
    if not verification:
        return

    table = Table(title="Source Verification", show_lines=True)
    table.add_column("Section", style="cyan", no_wrap=True)
    table.add_column("URL", style="blue", max_width=50)
    table.add_column("Verified", justify="center")
    table.add_column("Confidence", justify="center")

    for v in verification:
        icon   = "✓" if v.verified else "✗"
        colour = "green" if v.verified else "red"
        table.add_row(
            v.claim[:40],
            v.source_url[:50],
            f"[{colour}]{icon}[/{colour}]",
            f"{v.confidence:.0%}",
        )
    console.print(table)


def collect_human_review(state: dict) -> ReviewDecision:
    """
    Prompt the terminal user for a structured review decision.
    Used in interactive (Level 0) mode.
    """
    console.print()
    console.print(Panel(
        "[bold yellow]HUMAN REVIEW REQUIRED[/bold yellow]\n\n"
        "Options:\n"
        "  [green]approve[/green]        — Publish the report as-is\n"
        "  [yellow]revise[/yellow]        — Request specific revisions (no new search)\n"
        "  [red]reject[/red]         — Reject and restart research with feedback\n"
        "  [red]escalate[/red]      — Escalate for manual handling",
        border_style="yellow",
    ))

    while True:
        choice = console.input("[bold]Decision[/bold] (approve/revise/reject/escalate): ").strip().lower()
        if choice in ("approve", "revise", "reject", "escalate"):
            break
        console.print("[red]Invalid choice. Enter: approve, revise, reject, or escalate[/red]")

    if choice == "approve":
        return ReviewDecision(
            status="approved",
            reviewer_id=f"human:{uuid.uuid4().hex[:8]}",
        )
    elif choice == "escalate":
        reason = console.input("Escalation reason: ").strip()
        return ReviewDecision(
            status="needs_revision",
            reviewer_id=f"human:{uuid.uuid4().hex[:8]}",
            escalate_reason=reason or "Manual escalation",
        )
    else:
        instructions = console.input("Revision instructions: ").strip()
        required_src = console.input("Required sources to add (comma-separated URLs, or blank): ").strip()
        remove_claims = console.input("Claims to remove (comma-separated, or blank): ").strip()

        status = "needs_revision" if choice == "revise" else "rejected"
        return ReviewDecision(
            status=status,
            reviewer_id=f"human:{uuid.uuid4().hex[:8]}",
            revision_instructions=instructions or None,
            required_sources=[s.strip() for s in required_src.split(",") if s.strip()] or None,
            remove_claims=[c.strip() for c in remove_claims.split(",") if c.strip()] or None,
        )


# ── Core run loop ─────────────────────────────────────────────────────────────

def run_research(
    question: str,
    user_id: str = "cli_user",
    auto_approve: bool = False,
    config: Optional[RunConfig] = None,
) -> dict:
    """
    Execute a full research run with optional HITL.

    Returns the final state dict.
    """
    if config is None:
        config = RunConfig()

    if auto_approve:
        config = RunConfig(
            **{k: v for k, v in config.__dict__.items() if k != "autonomy_level"},
            autonomy_level=2,   # Confidence-gated auto-approval
        )

    thread_id = make_thread_id(user_id)
    run_id    = thread_id
    run_cfg   = {"configurable": {"thread_id": thread_id}}

    audit = AuditLogger(run_id=run_id, user_id=user_id)

    console.print(Panel(
        f"[bold green]Research Agent Starting[/bold green]\n\n"
        f"Question: [cyan]{question}[/cyan]\n"
        f"Thread:   [dim]{thread_id}[/dim]\n"
        f"Budget:   ${config.budget_usd:.2f} | Max iters: {config.max_iterations}\n"
        f"Mode:     {'Auto-approve (Level 2)' if auto_approve else 'Full HITL (Level 0)'}",
        border_style="green",
    ))

    initial_state: AgentState = {
        "run_id":          run_id,
        "user_id":         user_id,
        "question":        question,
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
    }

    audit.log_run_start(question, config)
    start_time = datetime.now(timezone.utc)

    review_cycles = 0
    final_state   = {}

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
        console=console,
    ) as progress:
        task = progress.add_task("Running research agent...", total=None)

        # ── Phase 1: Run to interrupt ─────────────────────────────────────────
        for event in app.stream(initial_state, run_cfg, stream_mode="values"):
            final_state = event
            # Show node progress
            msgs = event.get("messages", [])
            if msgs:
                last = msgs[-1]
                content = getattr(last, "content", "")[:80]
                progress.update(task, description=f"[cyan]{content}[/cyan]")

    # ── HITL loop ─────────────────────────────────────────────────────────────
    while True:
        # Check if we hit the interrupt (approve node)
        current = app.get_state(run_cfg)
        next_nodes = current.next

        if not next_nodes:
            # Graph completed
            break

        if "approve" in next_nodes:
            review_cycles += 1
            console.print()
            console.print(f"[bold]Review cycle {review_cycles}[/bold]")
            display_report(current.values)
            display_verification(current.values)

            if auto_approve:
                # In auto mode, decision is made inside approve_node
                console.print("[dim]Auto-approving (confidence-gated)...[/dim]")
                decision = None  # Let approve_node handle it
            else:
                decision = collect_human_review(current.values)

            if decision is not None:
                # Inject human decision into state
                app.update_state(
                    run_cfg,
                    {"review_decision": decision},
                    as_node="approve",
                )

            # ── Phase 4: Informed resume ──────────────────────────────────────
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                transient=True,
                console=console,
            ) as progress2:
                task2 = progress2.add_task("Resuming...", total=None)
                for event in app.stream(None, run_cfg, stream_mode="values"):
                    final_state = event
                    msgs = event.get("messages", [])
                    if msgs:
                        content = getattr(msgs[-1], "content", "")[:80]
                        progress2.update(task2, description=f"[cyan]{content}[/cyan]")
        else:
            # Some other interrupt or completion
            break

    # ── Final output ──────────────────────────────────────────────────────────
    end_time   = datetime.now(timezone.utc)
    duration_s = (end_time - start_time).total_seconds()

    console.print()
    outcome = "published" if final_state.get("review_decision", None) and \
              getattr(final_state.get("review_decision"), "status", "") == "approved" \
              else "escalated" if final_state.get("abort_reason") else "completed"

    console.print(Panel(
        f"[bold]Run Complete[/bold]\n"
        f"Outcome:       {outcome}\n"
        f"Duration:      {duration_s:.1f}s\n"
        f"Cost:          ${final_state.get('cost_usd', 0.0):.4f}\n"
        f"Iterations:    {final_state.get('iteration_count', 0)}\n"
        f"Review cycles: {review_cycles}\n"
        f"Search results:{len(final_state.get('search_results', []))}",
        border_style="blue" if outcome == "published" else "red",
    ))

    if not auto_approve:
        display_report(final_state)

    audit.log_run_end(
        outcome=outcome,
        total_cost_usd=final_state.get("cost_usd", 0.0),
        total_tokens=sum(
            r.input_tokens + r.output_tokens
            for r in final_state.get("cost_records", [])
        ),
        duration_ms=duration_s * 1000,
        review_cycles=review_cycles,
    )

    return final_state


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Production Research Agent — powered by LangGraph + OpenRouter"
    )
    parser.add_argument("question", nargs="?", help="Research question")
    parser.add_argument("--auto", action="store_true", help="Auto-approve mode (no HITL)")
    parser.add_argument("--user", default="cli_user", help="User ID (for thread namespacing)")
    parser.add_argument("--budget", type=float, default=1.00, help="Budget in USD")
    parser.add_argument("--max-iters", type=int, default=5, help="Max search iterations")
    parser.add_argument("--confidence", type=float, default=0.65, help="Confidence threshold")

    args = parser.parse_args()

    if not args.question:
        # Default demo question from the design doc
        args.question = "What are the latest developments in fusion energy?"
        console.print(f"[dim]No question provided. Using demo: {args.question}[/dim]\n")

    config = RunConfig(
        budget_usd=args.budget,
        max_iterations=args.max_iters,
        confidence_threshold=args.confidence,
    )

    run_research(
        question=args.question,
        user_id=args.user,
        auto_approve=args.auto,
        config=config,
    )


if __name__ == "__main__":
    main()
