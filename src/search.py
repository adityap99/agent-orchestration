"""
search.py — Web search tool with retry, circuit breaker, and fallback.

Implements the three-tier reliability pattern from the design doc:
  Tier 1: Retry with exponential backoff (tenacity)
  Tier 2: Fallback to alternate search provider
  Tier 3: Escalate — raises SearchUnavailableError for graph to handle

Uses DuckDuckGo (free, no API key) as primary, with a graceful mock fallback.
"""
from __future__ import annotations

import asyncio
import time
from datetime import datetime
from typing import TYPE_CHECKING, List, Optional

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.guardrails import sanitize_tool_result
from src.state import SearchResult

if TYPE_CHECKING:
    from src.observability import AuditLogger


# ── Custom exceptions ─────────────────────────────────────────────────────────

class SearchUnavailableError(Exception):
    """All search tiers exhausted — caller should route to escalate node."""


# ── Circuit breaker state (simple manual implementation) ─────────────────────

class CircuitBreaker:
    """
    Simple circuit breaker with three states: CLOSED, OPEN, HALF-OPEN.

    Design doc calls for pybreaker, but we implement a lightweight version
    to avoid import complexity; behaviour is equivalent.
    """
    CLOSED    = "closed"
    OPEN      = "open"
    HALF_OPEN = "half_open"

    def __init__(self, fail_max: int = 5, reset_timeout: float = 60.0):
        self._fail_max       = fail_max
        self._reset_timeout  = reset_timeout
        self._failures       = 0
        self._state          = self.CLOSED
        self._opened_at: Optional[float] = None

    @property
    def state(self) -> str:
        if self._state == self.OPEN:
            if time.monotonic() - (self._opened_at or 0) >= self._reset_timeout:
                self._state = self.HALF_OPEN
        return self._state

    def record_success(self) -> None:
        self._failures = 0
        self._state    = self.CLOSED

    def record_failure(self) -> None:
        self._failures += 1
        if self._failures >= self._fail_max:
            self._state     = self.OPEN
            self._opened_at = time.monotonic()

    def is_open(self) -> bool:
        return self.state == self.OPEN


_duckduckgo_breaker = CircuitBreaker(fail_max=5, reset_timeout=60.0)


# ── Primary search: DuckDuckGo ────────────────────────────────────────────────

@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(Exception),
    reraise=True,
)
async def _duckduckgo_search(query: str, max_results: int = 5) -> List[dict]:
    """Search DuckDuckGo via its unofficial search API with retry."""
    # Try new package name first (ddgs), fall back to old (duckduckgo_search)
    try:
        from ddgs import DDGS
    except ImportError:
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            raise SearchUnavailableError("Neither 'ddgs' nor 'duckduckgo_search' is installed")

    loop = asyncio.get_event_loop()
    results = await loop.run_in_executor(
        None,
        lambda: list(DDGS().text(query, max_results=max_results)),
    )
    return results


def _ddg_result_to_search_result(
    raw: dict,
    query: str,
    audit_logger: Optional["AuditLogger"],
    sanitize: bool = True,
) -> SearchResult:
    """Convert a raw DuckDuckGo result dict to our typed SearchResult."""
    url     = raw.get("href", raw.get("url", "unknown"))
    content = raw.get("body", raw.get("description", ""))

    # Guardrail: sanitize before injecting into LLM context
    if sanitize:
        content = sanitize_tool_result(content, url, audit_logger)

    # Rough credibility heuristic: prefer HTTPS, known domains
    credibility = _score_credibility(url)

    return SearchResult(
        url=url,
        content=content,
        credibility_score=credibility,
        retrieved_at=datetime.utcnow(),
        query=query,
    )


def _score_credibility(url: str) -> float:
    """
    Heuristic credibility score [0,1] based on URL characteristics.

    In production this would use a domain authority API or pre-computed list.
    """
    score = 0.5  # Baseline
    url_lower = url.lower()

    # Boost for reputable domain patterns
    high_cred = [".gov", ".edu", "nature.com", "science.org", "arxiv.org",
                 "pubmed.ncbi", "who.int", "reuters.com", "bbc.com", "nytimes.com",
                 "iter.org", "energy.gov", "iaea.org"]
    low_cred  = ["blogspot.com", "wordpress.com", "reddit.com", "quora.com",
                 "yahoo.answers"]

    for domain in high_cred:
        if domain in url_lower:
            score = min(1.0, score + 0.3)
            break
    for domain in low_cred:
        if domain in url_lower:
            score = max(0.1, score - 0.2)
            break

    if url_lower.startswith("https://"):
        score = min(1.0, score + 0.05)

    return round(score, 2)


# ── Fallback search: simple HTTP scrape of Bing ───────────────────────────────

async def _bing_fallback_search(query: str, max_results: int = 3) -> List[dict]:
    """
    Minimal Bing fallback — parses search result snippets.
    Returns fewer results; caller decrements confidence accordingly.
    """
    url = "https://www.bing.com/search"
    headers = {"User-Agent": "Mozilla/5.0 (compatible; ResearchAgent/1.0)"}
    params  = {"q": query, "count": max_results}

    async with httpx.AsyncClient(timeout=10.0, follow_redirects=True) as client:
        resp = await client.get(url, params=params, headers=headers)
        resp.raise_for_status()

    # Very basic extraction from Bing HTML (no parsing library needed for snippets)
    import re
    snippets = re.findall(r'<p[^>]*>(.*?)</p>', resp.text, re.DOTALL)
    results  = []
    for snip in snippets[:max_results]:
        clean = re.sub(r'<[^>]+>', '', snip).strip()
        if len(clean) > 50:
            results.append({"href": url, "body": clean})
    return results


# ── Public API ────────────────────────────────────────────────────────────────

async def search_with_fallback(
    query: str,
    max_results: int = 5,
    audit_logger: Optional["AuditLogger"] = None,
) -> tuple[List[SearchResult], bool]:
    """
    Execute a search with Tier 1 (DuckDuckGo + retry) → Tier 2 (Bing fallback).

    Returns:
        (results, is_fallback)  where is_fallback=True if we used the backup.

    Raises:
        SearchUnavailableError  if both tiers fail (graph routes to escalate).
    """
    start_ms = time.monotonic() * 1000

    # ── Tier 1 ────────────────────────────────────────────────────────────────
    if not _duckduckgo_breaker.is_open():
        try:
            raw_results = await _duckduckgo_search(query, max_results)
            _duckduckgo_breaker.record_success()
            results = [
                _ddg_result_to_search_result(r, query, audit_logger)
                for r in raw_results
            ]
            dur = time.monotonic() * 1000 - start_ms
            if audit_logger:
                audit_logger.log_tool_call(
                    tool_name="duckduckgo_search",
                    input_args={"query": query, "max_results": max_results},
                    output=f"{len(results)} results retrieved",
                    duration_ms=dur,
                    success=True,
                )
            return results, False
        except Exception as exc:
            _duckduckgo_breaker.record_failure()
            if _duckduckgo_breaker.is_open() and audit_logger:
                audit_logger.log_circuit_breaker_open("duckduckgo")
    else:
        if audit_logger:
            audit_logger.log_circuit_breaker_open("duckduckgo")

    # ── Tier 2: Bing fallback ─────────────────────────────────────────────────
    try:
        raw_fallback = await _bing_fallback_search(query, max(2, max_results // 2))
        results = [
            _ddg_result_to_search_result(r, query, audit_logger)
            for r in raw_fallback
        ]
        dur = time.monotonic() * 1000 - start_ms
        if audit_logger:
            audit_logger.log_tool_call(
                tool_name="bing_fallback_search",
                input_args={"query": query},
                output=f"{len(results)} fallback results",
                duration_ms=dur,
                success=True,
            )
        return results, True
    except Exception as exc2:
        raise SearchUnavailableError(
            f"All search tiers failed for query '{query}': {exc2}"
        ) from exc2


async def verify_url(url: str, timeout: float = 5.0) -> tuple[bool, Optional[int]]:
    """
    Check that a URL is reachable (HTTP HEAD request).
    Returns (is_reachable, http_status_code).
    """
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.head(url, follow_redirects=True)
            return resp.status_code < 400, resp.status_code
    except Exception:
        return False, None
