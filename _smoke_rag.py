"""Smoke test for Agentic RAG components."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ── 1. Graph builds with new node ─────────────────────────────────────────────
from src.graph import build_graph
app = build_graph()
nodes = sorted(app.get_graph().nodes.keys())
assert "retrieval_grade" in nodes, f"retrieval_grade missing from nodes: {nodes}"
print("1. Graph nodes:", nodes)

# ── 2. retrieval_grader imports ───────────────────────────────────────────────
from src.agents.retrieval_grader import retrieval_grader_node
print("2. retrieval_grader_node imported OK")

# ── 3. RRF merge ──────────────────────────────────────────────────────────────
from src.agents.executor import _rrf_merge
from src.state import SearchResult

r1 = SearchResult(url="http://a.com", content="a", credibility_score=0.9, query="q1")
r2 = SearchResult(url="http://b.com", content="b", credibility_score=0.7, query="q1")
r3 = SearchResult(url="http://b.com", content="b", credibility_score=0.8, query="q2")  # dup
r4 = SearchResult(url="http://c.com", content="c", credibility_score=0.5, query="q2")

merged = _rrf_merge([[r1, r2], [r3, r4]])
urls = [r.url for r in merged]
print(f"3. RRF merged: {urls}")
# b.com appears in both lists so gets boosted
assert "http://b.com" in urls[:2], f"b.com should be in top-2, got {urls}"
# highest-credibility copy of b.com should be kept (0.8 from q2)
b_copy = next(r for r in merged if r.url == "http://b.com")
assert b_copy.credibility_score == 0.8, f"Expected best copy credibility=0.8, got {b_copy.credibility_score}"
print("   RRF assertions passed")

# ── 4. Grade map + ContextWindowManager ──────────────────────────────────────
from src.memory.context_window import ContextWindowManager, _grade_map

state_with_grades = {
    "search_results": [r1, r2],
    "retrieval_grades": [
        {"url": "http://a.com", "relevance": "yes",  "score": 1.0},
        {"url": "http://b.com", "relevance": "no",   "score": 0.0},
    ],
}
grade_map = _grade_map(state_with_grades)
assert grade_map["http://a.com"] == 1.0
assert grade_map["http://b.com"] == 0.0
print(f"4. Grade map: {grade_map}")

# Check that grade tag appears and a.com (relevant) is ranked above b.com (irrelevant)
text = ContextWindowManager.pack_for_critic(state_with_grades)
assert "relevant" in text, "Grade tag should appear in evidence text"
# a.com should be listed as [1] since it's weighted higher
assert text.index("a.com") < text.index("b.com"), "a.com (relevant) should rank above b.com (irrelevant)"
print("   ContextWindowManager grade ranking OK")

# ── 5. _crag_signal in planner ────────────────────────────────────────────────
from src.agents.planner import _crag_signal

state_mostly_irrelevant = {
    "retrieval_grades": [
        {"relevance": "no",      "score": 0.0},
        {"relevance": "no",      "score": 0.0},
        {"relevance": "no",      "score": 0.0},
        {"relevance": "partial", "score": 0.5},
    ]
}
signal = _crag_signal(state_mostly_irrelevant)
assert "CRAG SIGNAL" in signal, f"Expected CRAG SIGNAL, got: {signal!r}"
print(f"5. CRAG corrective signal: {signal.strip()}")

state_mostly_relevant = {
    "retrieval_grades": [
        {"relevance": "yes", "score": 1.0},
        {"relevance": "yes", "score": 1.0},
    ]
}
assert _crag_signal(state_mostly_relevant) == "", "No signal needed for mostly-relevant results"
print("6. No corrective signal when results are mostly relevant: OK")

print()
print("All Agentic RAG smoke tests passed ✓")
