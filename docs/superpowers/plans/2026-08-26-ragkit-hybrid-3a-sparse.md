# ragkit hybrid 3a — sparse encoding + corpus stats

> **For agentic workers:** One task, TDD. Stay on branch `ragkit-hybrid-3a-sparse`. Do NOT merge/push/deploy. Steps use `- [ ]`.

**Goal:** Pure-Python BM25 sparse encoding + corpus statistics — the non-breaking foundation of hybrid retrieval. No schema change, no search change, no ingest wiring yet (those are 3b/3c). Pure logic, fully unit-tested network-free.

**Design:** BM25 factorizes so the **query** vector carries IDF and the **doc** vector carries TF-saturation + doc-length norm; `dot(query, doc)` reconstructs BM25. Doc vectors therefore hold no corpus-wide IDF and don't go stale as the corpus grows (only avgdl drifts; a re-ingest refreshes it). `CorpusStats` (document frequencies, N docs, total length) is the small ingest-time state, persisted as JSON per collection. Terms hash to u32 indices (Qdrant sparse-vector indices) — collisions are rare and tolerable.

**Tech Stack:** stdlib only (`re`, `hashlib`, `math`, `json`). pytest. No new deps.

## Global Constraints
- stdlib only; plain dataclass; no deps. LF endings. Tests network-free.
- This stage is additive and unused by the rest of ragkit until 3b — it must not change any existing behavior.

---

### Task 1: `ragkit/sparse.py` + tests

**Files:** Create `ragkit/sparse.py`; Test `tests/test_sparse.py`.
**Interfaces — Produces:** `tokenize(text)->list[str]`, `term_index(term)->int`, `CorpusStats(n_docs,total_len,df)` with `.avgdl`, `.add_doc(text)`, `.idf(ti)`, `.save(path)`, `.load(path)`, `doc_sparse(text, stats, *, k1=1.5, b=0.75)->dict[int,float]`, `query_sparse(text, stats)->dict[int,float]`, `to_qdrant(sparse)->{"indices":[],"values":[]}`.

- [ ] **Step 1: write `tests/test_sparse.py`**

```python
from __future__ import annotations

from ragkit.sparse import (
    CorpusStats, doc_sparse, query_sparse, term_index, to_qdrant, tokenize,
)


def test_tokenize_lowercases_filters_short_and_punct():
    assert tokenize("The Cap-Rate is 12% on Unit A!") == ["the", "cap", "rate", "is", "12", "on", "unit"]
    # single-char tokens ("a") and punctuation dropped; digits kept.


def test_term_index_stable_and_u32():
    a, b = term_index("balance"), term_index("balance")
    assert a == b and 0 <= a < 2**32
    assert term_index("balance") != term_index("escrow")


def test_add_doc_counts_df_once_per_doc():
    s = CorpusStats()
    s.add_doc("rent rent rent")          # 3 tokens, but df for 'rent' is +1
    s.add_doc("rent escrow")
    assert s.n_docs == 2
    assert s.total_len == 5
    assert s.df[term_index("rent")] == 2
    assert s.df[term_index("escrow")] == 1
    assert s.avgdl == 2.5


def test_idf_rarer_term_scores_higher():
    s = CorpusStats()
    for _ in range(9):
        s.add_doc("common")
    s.add_doc("common rare")
    assert s.idf(term_index("rare")) > s.idf(term_index("common"))


def test_doc_sparse_saturates_with_tf():
    s = CorpusStats()
    s.add_doc("a b c d")   # avgdl context
    one = doc_sparse("x", s)[term_index("x")]
    many = doc_sparse("x x x x x x", s)[term_index("x")]
    assert many > one              # more occurrences -> higher weight
    assert many < 6 * one          # ...but saturates (not linear)


def test_query_sparse_is_idf_valued():
    s = CorpusStats()
    s.add_doc("alpha beta")
    q = query_sparse("alpha alpha", s)     # dedup to unique terms
    assert list(q.keys()) == [term_index("alpha")]
    assert q[term_index("alpha")] == s.idf(term_index("alpha"))


def test_bm25_dot_rewards_matching_rare_term():
    s = CorpusStats()
    for _ in range(20):
        s.add_doc("filler text here")
    s.add_doc("the escrow balance rose")
    qv = query_sparse("escrow", s)
    dv_match = doc_sparse("the escrow balance rose", s)
    dv_miss = doc_sparse("filler text here", s)
    dot = lambda q, d: sum(v * d.get(k, 0.0) for k, v in q.items())
    assert dot(qv, dv_match) > 0
    assert dot(qv, dv_miss) == 0


def test_to_qdrant_shape_and_empty():
    out = to_qdrant({5: 0.5, 9: 0.25})
    assert set(out.keys()) == {"indices", "values"}
    assert len(out["indices"]) == len(out["values"]) == 2
    assert to_qdrant({}) == {"indices": [], "values": []}


def test_corpus_stats_roundtrip(tmp_path):
    s = CorpusStats()
    s.add_doc("alpha beta gamma")
    s.add_doc("alpha delta")
    p = str(tmp_path / "stats.json")
    s.save(p)
    r = CorpusStats.load(p)
    assert r.n_docs == s.n_docs and r.total_len == s.total_len and r.df == s.df
    assert CorpusStats.load(str(tmp_path / "missing.json")).n_docs == 0
```

- [ ] **Step 2: run → fails** (`ModuleNotFoundError: ragkit.sparse`).
- [ ] **Step 3: implement `ragkit/sparse.py`**

```python
"""Pure-Python BM25 sparse encoding for hybrid retrieval.

No model, no service. BM25 factorizes so IDF lives on the QUERY vector and the
TF-saturation (with doc-length norm) lives on the DOC vector; their dot product
reconstructs BM25. Stored doc vectors therefore carry no corpus-wide IDF and
don't go stale as the corpus grows (only avgdl drifts — a re-ingest refreshes
it). CorpusStats is the small ingest-time state, persisted as JSON per collection.
Terms hash to u32 indices (Qdrant sparse indices); collisions are rare/tolerable.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass, field

_TOKEN = re.compile(r"[a-z0-9]+")
K1 = 1.5
B = 0.75


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall((text or "").lower()) if len(t) >= 2]


def term_index(term: str) -> int:
    """Stable u32 index for a term (blake2b hash; collisions rare and tolerable)."""
    return int(hashlib.blake2b(term.encode("utf-8"), digest_size=4).hexdigest(), 16)


@dataclass
class CorpusStats:
    n_docs: int = 0
    total_len: int = 0
    df: dict[int, int] = field(default_factory=dict)   # term_index -> document frequency

    @property
    def avgdl(self) -> float:
        return self.total_len / self.n_docs if self.n_docs else 0.0

    def add_doc(self, text: str) -> None:
        toks = tokenize(text)
        self.n_docs += 1
        self.total_len += len(toks)
        for ti in {term_index(t) for t in toks}:      # unique per doc -> document frequency
            self.df[ti] = self.df.get(ti, 0) + 1

    def idf(self, ti: int) -> float:
        # BM25 idf, +1 inside log to stay non-negative; unseen term (df 0) -> max idf.
        df = self.df.get(ti, 0)
        return math.log(1 + (self.n_docs - df + 0.5) / (df + 0.5))

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            json.dump({"n_docs": self.n_docs, "total_len": self.total_len,
                       "df": {str(k): v for k, v in self.df.items()}}, f)

    @staticmethod
    def load(path: str) -> "CorpusStats":
        if not os.path.exists(path):
            return CorpusStats()
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return CorpusStats(n_docs=d["n_docs"], total_len=d["total_len"],
                           df={int(k): v for k, v in d["df"].items()})


def doc_sparse(text: str, stats: CorpusStats, *, k1: float = K1, b: float = B) -> dict[int, float]:
    """Doc-side BM25 vector: TF-saturation + length norm, NO idf (applied at query)."""
    toks = tokenize(text)
    dl = len(toks)
    avgdl = stats.avgdl or dl or 1.0
    tf: dict[int, int] = {}
    for t in toks:
        ti = term_index(t)
        tf[ti] = tf.get(ti, 0) + 1
    denom_norm = k1 * (1 - b + b * dl / avgdl)
    return {ti: f * (k1 + 1) / (f + denom_norm) for ti, f in tf.items()}


def query_sparse(text: str, stats: CorpusStats) -> dict[int, float]:
    """Query-side vector: idf per unique query term."""
    return {ti: stats.idf(ti) for ti in {term_index(t) for t in tokenize(text)}}


def to_qdrant(sparse: dict[int, float]) -> dict:
    """{index: value} -> Qdrant sparse vector {indices, values}."""
    if not sparse:
        return {"indices": [], "values": []}
    idx, val = zip(*sparse.items())
    return {"indices": list(idx), "values": list(val)}
```

- [ ] **Step 4: run → pass** (`pytest tests/test_sparse.py -q`).
- [ ] **Step 5: full suite** — `pytest -q` green (nothing else touched).
- [ ] **Step 6: commit** — `git add ragkit/sparse.py tests/test_sparse.py && git commit -m "feat(sparse): pure-Python BM25 encoding + corpus stats (hybrid 3a)"`

---

## Acceptance
1. `pytest -q` green (existing + test_sparse), network-free.
2. `python -c "import ragkit.sparse"` OK.
3. No existing file modified — additive only.
4. Committed on branch `ragkit-hybrid-3a-sparse`; master untouched.

## Next (not this stage)
- 3b: named dense+sparse collection schema, ingest stores both, corpus stats updated at ingest (breaking; re-ingest required).
- 3c: dual-leg search + RRF fusion via Qdrant Query API; `--hybrid` flag.
