"""Corrective RAG (CRAG): grade retrieved hits for relevance to the query, drop clear junk, and —
when the whole pool is weak — trigger ONE corrective re-retrieval with a rewritten query. Grading
is a local-model call wired by the Searcher; the pure filter + decision logic lives here and is
unit-tested. CRAG is a PRECISION technique (trades a little recall for cleaner context) except its
weak-pool fallback, which recovers recall on queries the first pass missed.
"""
from __future__ import annotations

DROP_THRESHOLD = 0.25   # drop hits graded below this (clearly off-topic) — conservative
FALLBACK_FLOOR = 0.5    # if the best hit grades below this, the pool is weak -> corrective action


def parse_grades(raw: object, n: int) -> list[float]:
    """Model JSON -> n relevance grades clamped to [0,1]. Fails SAFE: a bad/short response yields
    neutral 0.5s so CRAG neither drops a hit nor falsely triggers the fallback on a model hiccup."""
    gs = raw.get("grades") if isinstance(raw, dict) else raw
    if not isinstance(gs, list):
        return [0.5] * n
    out: list[float] = []
    for i in range(n):
        try:
            out.append(max(0.0, min(1.0, float(gs[i]))))
        except (IndexError, TypeError, ValueError):
            out.append(0.5)
    return out


def apply_grades(hits: list[dict], grades: list[float],
                 drop: float = DROP_THRESHOLD) -> tuple[list[dict], float]:
    """Attach each grade, drop hits below `drop`, preserve order. Returns (kept, best_grade)."""
    kept = [dict(h, crag_grade=g) for h, g in zip(hits, grades) if g >= drop]
    best = max(grades) if grades else 0.0
    return kept, best


def is_weak(best_grade: float, floor: float = FALLBACK_FLOOR) -> bool:
    """Pool is weak (nothing clears the floor) -> a corrective re-retrieval is warranted."""
    return best_grade < floor
