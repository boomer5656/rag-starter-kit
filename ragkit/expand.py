"""Multi-query expansion for retrieval recall: paraphrase the query, retrieve for each, and
Reciprocal-Rank-Fuse the results. Pure helpers here (parse the model's JSON, fuse ranked
hit lists); the LLM call is wired by the Searcher via ollama.generate_json.
"""
from __future__ import annotations

_RRF_K = 60


def parse_expansions(raw: object, original: str, n: int) -> list[str]:
    """Up to n distinct paraphrases from the model's JSON, excluding the original query."""
    qs = raw.get("queries") if isinstance(raw, dict) else raw
    if not isinstance(qs, list):
        return []
    out: list[str] = []
    seen = {original.strip().lower()}
    for q in qs:
        s = str(q).strip()
        low = s.lower()
        if s and low not in seen:
            seen.add(low)
            out.append(s)
        if len(out) >= n:
            break
    return out


def rrf_fuse(hit_lists: list[list[dict]], k: int = _RRF_K) -> list[dict]:
    """RRF-fuse ranked hit lists by point id: score(id) = Σ 1/(k + rank). Returns unique hits
    ordered by fused score, keeping the first-seen payload per id."""
    scores: dict = {}
    keep: dict = {}
    for hits in hit_lists:
        for rank, h in enumerate(hits, start=1):
            hid = h.get("id")
            if hid is None:
                continue
            scores[hid] = scores.get(hid, 0.0) + 1.0 / (k + rank)
            keep.setdefault(hid, h)
    return [keep[i] for i in sorted(scores, key=lambda i: -scores[i])]
