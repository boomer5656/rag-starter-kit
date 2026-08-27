# ragkit multi-query expansion — Implementation Plan

> **For agentic workers:** TDD, task by task. Stay on branch `multi-query-expansion`. Do NOT merge/push. Steps use `- [ ]`.

**Goal:** A `--multi N` query-expansion path: paraphrase the query with the local model into N variants, retrieve for each (reusing the dense/hybrid path), RRF-fuse across queries, then rerank against the ORIGINAL query. Improves recall on vague/synonym-heavy queries. Default off; measurable with the eval harness (`ragkit eval --multi N`).

**Architecture:** Pure helpers in `ragkit/expand.py` (`parse_expansions`, `rrf_fuse`) — unit-tested. `Searcher` gains a `_retrieve_hits` helper (the existing per-query dense/hybrid retrieval, extracted) and a `multi` path that expands via `ollama.generate_json`, fuses, and reranks. Config `multi_query` block + `--multi` CLI flag.

**Tech Stack:** stdlib + httpx; reuses `ollama.generate_json` (think:false + cap) and the RRF idea from hybrid. No new deps.

## Global Constraints
- stdlib + httpx; plain dataclasses; config single-source; thin CLI.
- Best-effort: expansion failure ⇒ fall back to the single-query path (never throw).
- Default OFF (`multi=0`) ⇒ byte-for-byte current `search()` behavior.
- Rerank against the ORIGINAL query (the paraphrases are a recall device, not the intent).
- Tests network-free (pure helpers; the LLM call is not unit-tested).

---

### Task 1: `ragkit/expand.py` (parse + fuse) + tests

**Files:** Create `ragkit/expand.py`, `tests/test_expand.py`.
**Interfaces — Produces:** `parse_expansions(raw, original, n) -> list[str]`, `rrf_fuse(hit_lists, k=60) -> list[dict]`.

- [ ] **Step 1: `tests/test_expand.py`**

```python
from __future__ import annotations

from ragkit.expand import parse_expansions, rrf_fuse


def test_parse_expansions_dedups_and_caps():
    raw = {"queries": ["how much rent", "HOW MUCH RENT", "monthly rent due", "rent amount", "lease cost"]}
    out = parse_expansions(raw, original="how much rent", n=2)
    assert out == ["monthly rent due", "rent amount"]   # original + dup dropped, capped to 2


def test_parse_expansions_accepts_bare_list_and_handles_junk():
    assert parse_expansions(["a query here"], "orig", 3) == ["a query here"]
    assert parse_expansions({"nope": 1}, "orig", 3) == []
    assert parse_expansions("garbage", "orig", 3) == []


def test_rrf_fuse_ranks_by_reciprocal_rank_across_lists():
    a = [{"id": 1, "payload": {"t": "a1"}}, {"id": 2}, {"id": 3}]
    b = [{"id": 2, "payload": {"t": "b2"}}, {"id": 4}, {"id": 1}]
    fused = rrf_fuse([a, b], k=60)
    ids = [h["id"] for h in fused]
    # 2 appears at ranks 2 and 1; 1 at ranks 1 and 3 -> both high; unique; 3 and 4 lower
    assert set(ids) == {1, 2, 3, 4}
    assert ids[0] in (1, 2) and ids[1] in (1, 2)
    # first-seen payload kept
    assert next(h for h in fused if h["id"] == 1)["payload"] == {"t": "a1"}


def test_rrf_fuse_skips_hits_without_id():
    fused = rrf_fuse([[{"score": 1}, {"id": 5}]])
    assert [h["id"] for h in fused] == [5]
```

- [ ] **Step 2: run → fails.**
- [ ] **Step 3: create `ragkit/expand.py`**

```python
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
```

- [ ] **Step 4: run → pass.**
- [ ] **Step 5: commit** — `git add ragkit/expand.py tests/test_expand.py && git commit -m "feat(expand): multi-query parse + RRF fuse helpers"`

---

### Task 2: `Searcher` multi-query path

**Files:** Modify `ragkit/search.py`.
**Interfaces — Consumes:** `ragkit.expand`, `ragkit.ollama.generate_json`. Produces: `Searcher.search(..., multi: int = 0)`.

- [ ] **Step 1: imports + expand client.** Add to `search.py`:

```python
import httpx

from .expand import parse_expansions, rrf_fuse
from .ollama import generate_json
```

In `__init__`, after `self._stats = ...`:

```python
        self._expand_client = httpx.Client(timeout=120.0)
```

In `close()`, add `self._expand_client.close()`.

- [ ] **Step 2: add the retrieval helper + expander** (methods on `Searcher`, above `search`):

```python
    _EXPAND_SYSTEM = (
        "You rewrite a search query into alternative phrasings for a retrieval system — "
        "vary vocabulary, specificity, and synonyms while preserving intent. Output ONLY JSON: "
        "{\"queries\": [\"...\"]}."
    )

    def _retrieve_hits(self, store: Store, query: str, fetch_k: int, hybrid: bool,
                       coll: str, query_filter: dict | None) -> list[dict]:
        """One query's raw hits (dense, or hybrid when stats exist) — no rerank."""
        vector = self.embedder.embed(query)
        stats = self._stats if coll == self.cfg.collection else CorpusStats.load(stats_path(coll))
        if hybrid and stats.n_docs > 0:
            sparse = to_qdrant(query_sparse(query, stats))
            return store.query_hybrid(vector, sparse, top_k=fetch_k, query_filter=query_filter)
        return store.search(vector, top_k=fetch_k, query_filter=query_filter)

    def _expand(self, query: str, n: int) -> list[str]:
        """n LLM paraphrases (best-effort → [] on any failure)."""
        model = self.cfg.multi_query.model or self.cfg.ollama.gate_model
        prompt = f"Query: {query}\n\nReturn JSON: {{\"queries\": [...]}} with {n} distinct rephrasings."
        try:
            raw = generate_json(self._expand_client, self.cfg.ollama.url, model,
                                system=self._EXPAND_SYSTEM, prompt=prompt)
            return parse_expansions(raw, query, n)
        except Exception:
            return []
```

- [ ] **Step 3: replace `search()`** with the multi-aware version (fetches per query, fuses, reranks the fused pool against the ORIGINAL query):

```python
    def search(self, query: str, *, top_k: int = 5, rerank: bool = False, hybrid: bool = False,
               multi: int = 0, query_filter: dict | None = None,
               collection: str | None = None) -> list[dict]:
        store = self.store
        opened = False
        coll = collection if collection is not None else self.cfg.collection
        if collection is not None and collection != self.store.collection:
            store = Store(self.cfg.qdrant, collection)
            opened = True
        try:
            want_rerank = rerank and bool(self.cfg.reranker.url)
            fetch_k = top_k * _RERANK_FANOUT if want_rerank else top_k
            if multi > 0:
                queries = [query] + self._expand(query, multi)
                hits = rrf_fuse([self._retrieve_hits(store, q, fetch_k, hybrid, coll, query_filter)
                                 for q in queries])
            else:
                hits = self._retrieve_hits(store, query, fetch_k, hybrid, coll, query_filter)
            if want_rerank:
                hits = self.reranker.rerank(query, hits, top_k=top_k)   # rerank vs ORIGINAL query
            else:
                hits = hits[:top_k]
            return [self._to_result(h) for h in hits]
        finally:
            if opened:
                store.close()
```

- [ ] **Step 4: verify** — `python -c "import ragkit.search"`; `pytest -q` green (multi defaults 0 ⇒ existing behavior; no unit tests for the network path).
- [ ] **Step 5: commit** — `git add ragkit/search.py && git commit -m "feat(search): multi-query expansion + RRF fusion (default off)"`

---

### Task 3: config + `--multi` CLI flag

**Files:** Modify `ragkit/config.py`, `ragkit/cli.py`, `ragkit.example.yaml`.

- [ ] **Step 1: config.** In `config.py`, add after `EvalConfig` (or near it):

```python
@dataclass
class MultiQueryConfig:
    model: Optional[str] = None    # None -> ollama.gate_model; use a CAPABLE instruct model
    n: int = 3                     # default paraphrase count for `--multi` with no number
```

Add field `multi_query: MultiQueryConfig = field(default_factory=MultiQueryConfig)` to `Config`, and wire `multi_query=MultiQueryConfig(**(data.get("multi_query") or {}))` into `Config.load`.

- [ ] **Step 2: `cmd_search`** — pass multi, add the flag. In `cmd_search`, change the call to `searcher.search(args.query, top_k=args.top_k, rerank=args.rerank, hybrid=args.hybrid, multi=args.multi)`; add to `p_search`:

```python
    p_search.add_argument("--multi", type=int, nargs="?", const=None, default=0,
                          help="expand into N paraphrases + RRF-fuse (N optional; default from config)")
```

Because `const=None` (bare `--multi` → None → use config n), resolve it in `cmd_search` before the call:

```python
    multi = cfg.multi_query.n if args.multi is None else args.multi
```

and pass `multi=multi`.

- [ ] **Step 3: `cmd_eval`** — same flag + thread it through `search_fn`. Add to `p_eval` the identical `--multi` arg; in `cmd_eval` resolve `multi = cfg.multi_query.n if args.multi is None else args.multi`; change `search_fn` to `searcher.search(query, top_k=top_k * _DISTINCT_FANOUT, rerank=args.rerank, hybrid=args.hybrid, multi=multi)`.

- [ ] **Step 4: document in `ragkit.example.yaml`** (append):

```yaml

# Multi-query expansion (ragkit search/eval --multi N). Paraphrase the query, retrieve each,
# RRF-fuse. Helps vague/synonym-heavy queries. Off unless --multi is passed.
multi_query:
  # model: null            # null -> ollama.gate_model; use a CAPABLE instruct model
  n: 3
```

- [ ] **Step 5: verify** — `python -m ragkit.cli search --help` and `eval --help` show `--multi`; `python -c "from ragkit.config import Config; print(Config().multi_query.n)"` → 3; `pytest -q` green.
- [ ] **Step 6: commit** — `git add ragkit/config.py ragkit/cli.py ragkit.example.yaml && git commit -m "feat(cli): --multi query-expansion flag on search and eval"`

---

## Acceptance
1. `pytest -q` green (existing + test_expand), network-free.
2. `python -c "import ragkit.expand, ragkit.search, ragkit.cli"` OK.
3. `--multi` unset / `multi=0` ⇒ byte-for-byte current search behavior; best-effort fallback on expansion failure.
4. Committed on `multi-query-expansion`; master untouched.

## Notes for the smoke (not build steps)
Live A/B: `ragkit eval <c> --hybrid --k 1,5,10` (baseline) vs `--hybrid --multi 3` — expect equal-or-better recall, biggest gains on vague/synonym queries.
