"""
memory/semantic.py — ChromaDB-backed semantic memory.

Stores and retrieves:
  1. source_chunks   — search result content indexed by embedding
  2. report_sections — prior report sections for synthesis context
  3. verified_claims — URL → verification outcome cache (skip HTTP re-checks)

ChromaDB is OPTIONAL. If not installed (or fails to initialise) all methods
silently return empty results so the system continues to work normally.
"""
from __future__ import annotations

import hashlib
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from src.config import SEMANTIC_STORE_DIR

_WRITE_LOCK = threading.Lock()


def _chromadb_available() -> bool:
    try:
        import chromadb  # noqa: F401
        return True
    except ImportError:
        return False


class SemanticStore:
    """
    Vector-search semantic memory backed by ChromaDB.
    Gracefully degrades to no-ops when chromadb is not installed.
    """

    def __init__(self, persist_dir: str = SEMANTIC_STORE_DIR) -> None:
        self.available = False
        self._client: Any   = None
        self._sources: Any  = None
        self._sections: Any = None
        self._claims: Any   = None

        if not _chromadb_available():
            return

        try:
            import chromadb  # noqa: F811

            Path(persist_dir).mkdir(parents=True, exist_ok=True)
            self._client   = chromadb.PersistentClient(path=str(persist_dir))
            self._sources  = self._client.get_or_create_collection("source_chunks")
            self._sections = self._client.get_or_create_collection("report_sections")
            self._claims   = self._client.get_or_create_collection("verified_claims")
            self.available = True
        except Exception:
            pass  # Leave available=False — all methods return empty

    # ── Query ─────────────────────────────────────────────────────────────────

    def search_sources(self, query: str, n: int = 8) -> list[dict]:
        """Find semantically similar cached source chunks for a query string."""
        if not self.available:
            return []
        try:
            count = self._sources.count()
            if count == 0:
                return []
            results = self._sources.query(
                query_texts=[query],
                n_results=min(n, count),
            )
            items: list[dict] = []
            for doc, meta in zip(
                results["documents"][0], results["metadatas"][0]
            ):
                items.append(
                    {
                        "url":               meta.get("url", ""),
                        "content":           doc,
                        "credibility_score": float(meta.get("credibility_score", 0.5)),
                        "query":             meta.get("source_query", query),
                    }
                )
            return items
        except Exception:
            return []

    def search_reports(self, query: str, n: int = 4) -> list[dict]:
        """Find related prior report sections for synthesis context."""
        if not self.available:
            return []
        try:
            count = self._sections.count()
            if count == 0:
                return []
            results = self._sections.query(
                query_texts=[query],
                n_results=min(n, count),
            )
            items: list[dict] = []
            for doc, meta in zip(
                results["documents"][0], results["metadatas"][0]
            ):
                items.append(
                    {
                        "content": doc,
                        "title":   meta.get("title", ""),
                        "run_id":  meta.get("run_id", ""),
                    }
                )
            return items
        except Exception:
            return []

    def is_claim_cached(
        self, url: str, max_age_days: int = 30
    ) -> Optional[bool]:
        """
        Check if a URL's verification result is cached.
        Returns True/False if cached and fresh, None if not cached or expired.
        """
        if not self.available:
            return None
        try:
            claim_id = "claim:" + hashlib.sha256(url.encode()).hexdigest()[:16]
            results  = self._claims.get(ids=[claim_id], include=["metadatas"])
            if not results["ids"]:
                return None
            meta          = results["metadatas"][0]
            cached_at_str = meta.get("cached_at", "")
            if cached_at_str:
                cached_at = datetime.fromisoformat(cached_at_str)
                # Ensure both datetimes are timezone-aware for comparison
                now = datetime.now(timezone.utc)
                if cached_at.tzinfo is None:
                    cached_at = cached_at.replace(tzinfo=timezone.utc)
                age_days = (now - cached_at).days
                if age_days > max_age_days:
                    return None
            return bool(meta.get("verified", False))
        except Exception:
            return None

    # ── Index ─────────────────────────────────────────────────────────────────

    def index_results(self, run_id: str, results: list) -> None:
        """Index SearchResult objects (or compatible dicts) from a completed run."""
        if not self.available:
            return
        with _WRITE_LOCK:
            try:
                docs: list[str]  = []
                metas: list[dict] = []
                ids: list[str]   = []

                for i, r in enumerate(results):
                    url  = r.url  if hasattr(r, "url")               else r.get("url", "")
                    cont = r.content if hasattr(r, "content")         else r.get("content", "")
                    cred = r.credibility_score if hasattr(r, "credibility_score") else r.get("credibility_score", 0.5)
                    qry  = r.query if hasattr(r, "query")             else r.get("query", "")

                    chunk_id = f"{run_id}:src:{i}"
                    docs.append(cont[:2000])
                    metas.append(
                        {
                            "url":               url,
                            "run_id":            run_id,
                            "credibility_score": float(cred),
                            "source_query":      qry,
                        }
                    )
                    ids.append(chunk_id)

                if docs:
                    self._sources.upsert(documents=docs, metadatas=metas, ids=ids)
            except Exception:
                pass

    def index_report(self, run_id: str, report: Any) -> None:
        """Index report sections from a completed run."""
        if not self.available:
            return
        with _WRITE_LOCK:
            try:
                sections = getattr(report, "sections", [])
                docs: list[str]   = []
                metas: list[dict] = []
                ids: list[str]    = []

                for i, sec in enumerate(sections):
                    title   = getattr(sec, "title",   f"Section {i}")
                    content = getattr(sec, "content", "")
                    sources = getattr(sec, "sources", [])
                    sec_id  = f"{run_id}:sec:{i}"

                    docs.append(f"{title}\n{content[:2000]}")
                    metas.append(
                        {
                            "run_id":       run_id,
                            "title":        title,
                            "sources_json": json.dumps(sources[:5]),
                        }
                    )
                    ids.append(sec_id)

                if docs:
                    self._sections.upsert(documents=docs, metadatas=metas, ids=ids)
            except Exception:
                pass

    def cache_verification(
        self, url: str, verified: bool, confidence: float = 0.9
    ) -> None:
        """Cache the HTTP verification outcome for a URL (30-day TTL by default)."""
        if not self.available:
            return
        with _WRITE_LOCK:
            try:
                claim_id = "claim:" + hashlib.sha256(url.encode()).hexdigest()[:16]
                self._claims.upsert(
                    documents=[url],
                    metadatas=[
                        {
                            "url":        url,
                            "verified":   verified,
                            "confidence": float(confidence),
                            "cached_at":  datetime.now(timezone.utc).isoformat(),
                        }
                    ],
                    ids=[claim_id],
                )
            except Exception:
                pass
