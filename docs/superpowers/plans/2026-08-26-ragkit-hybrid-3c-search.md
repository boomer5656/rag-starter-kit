# ragkit hybrid 3c — dual-leg search + RRF fusion

> **For agentic workers:** TDD, task by task. Stay on branch `ragkit-hybrid-3c-search`. Do NOT merge/push/deploy. Steps use `- [ ]`.

**Goal:** Activate hybrid retrieval — a `--hybrid` query flag that runs a dense leg + a BM25 sparse leg and fuses them with **RRF server-side** (Qdrant Query API). Graceful: if a collection has no corpus stats, `--hybrid` falls back to dense-only. Reranking (when configured) still applies after fusion.

**Design:** Query-time, `query_sparse` (3a) builds the sparse query from the collection's persisted `CorpusStats`; Qdrant's `/points/query` prefetches both legs (`using: "dense"` / `using: "sparse"`) and fuses with `{"fusion": "rrf"}`. A `stats_path(collection)` helper (3a module) DRYs the path shared by ingest and search.

**Tech Stack:** stdlib + httpx; uses `ragkit.sparse` (3a) + the named schema (3b). No new deps.

## Global Constraints
- stdlib + httpx; LF endings; tests network-free (fake client / injected stats).
- `--hybrid` is per-query (like `--rerank`); default off ⇒ existing dense behavior unchanged.
- Missing/empty stats ⇒ dense-only fallback, never a crash.

---

### Task 1: `stats_path` helper + `Store.query_hybrid`

**Files:** Modify `ragkit/sparse.py`, `ragkit/store.py`; extend `tests/test_sparse.py`, `tests/test_store.py`.

- [ ] **Step 1: tests.** Append to `tests/test_sparse.py`:

```python
def test_stats_path_shape():
    from ragkit.sparse import stats_path
    assert stats_path("mycoll").replace("\\", "/") == ".ragkit/sparse/mycoll.json"
```

Append to `tests/test_store.py`:

```python
def test_query_hybrid_body_has_both_legs_and_rrf():
    s = _store(_FakeClient())
    # _FakeClient.post returns {"result": []}; query api reads result.points -> tolerate missing
    s._client.post = lambda url, json=None: (s._client.calls.append(("POST", url, json))
                                             or _Resp(200, {"result": {"points": []}}))
    s.query_hybrid([0.1, 0.2], {"indices": [7], "values": [0.5]}, top_k=4, prefetch=20)
    body = next(c[2] for c in s._client.calls if c[0] == "POST" and c[1].endswith("/points/query"))
    legs = {leg["using"]: leg for leg in body["prefetch"]}
    assert set(legs) == {"dense", "sparse"}
    assert legs["dense"]["query"] == [0.1, 0.2] and legs["dense"]["limit"] == 20
    assert legs["sparse"]["query"] == {"indices": [7], "values": [0.5]}
    assert body["query"] == {"fusion": "rrf"}
    assert body["limit"] == 4
```

- [ ] **Step 2: run → fails.**
- [ ] **Step 3: add `stats_path` to `ragkit/sparse.py`** (after `to_qdrant`, `os` is already imported):

```python
def stats_path(collection: str) -> str:
    """Where a collection's BM25 corpus stats are persisted."""
    return os.path.join(".ragkit", "sparse", f"{collection}.json")
```

- [ ] **Step 4: add `Store.query_hybrid` to `ragkit/store.py`** (before `close`):

```python
    def query_hybrid(self, dense_vector: list[float], sparse_vector: dict,
                     top_k: int = 5, query_filter: dict | None = None,
                     prefetch: int = 50) -> list[dict]:
        """Dense + sparse legs fused server-side with RRF (Qdrant Query API)."""
        legs = [
            {"query": dense_vector, "using": "dense", "limit": prefetch},
            {"query": {"indices": sparse_vector.get("indices", []),
                       "values": sparse_vector.get("values", [])},
             "using": "sparse", "limit": prefetch},
        ]
        if query_filter:
            for leg in legs:
                leg["filter"] = query_filter
        body = {"prefetch": legs, "query": {"fusion": "rrf"},
                "limit": top_k, "with_payload": True}
        r = self._client.post(self._url("/points/query"), json=body)
        r.raise_for_status()
        return r.json().get("result", {}).get("points", [])
```

- [ ] **Step 5: run → pass** (`pytest tests/test_sparse.py tests/test_store.py -q`).
- [ ] **Step 6: commit** — `git add ragkit/sparse.py ragkit/store.py tests/test_sparse.py tests/test_store.py && git commit -m "feat(store): query_hybrid (RRF) + stats_path helper (hybrid 3c)"`

---

### Task 2: `Searcher` hybrid routing

**Files:** Modify `ragkit/search.py`.
**Interfaces — Produces:** `Searcher.search(query, *, top_k=5, rerank=False, hybrid=False, query_filter=None, collection=None)`.

- [ ] **Step 1: imports.** In `search.py`, add:

```python
from .sparse import CorpusStats, query_sparse, stats_path, to_qdrant
```

- [ ] **Step 2: load stats in `__init__`.** After `self.reranker = Reranker(cfg.reranker)`:

```python
        self._stats = CorpusStats.load(stats_path(cfg.collection))
```

- [ ] **Step 3: replace `search()`** with the hybrid-aware version:

```python
    def search(self, query: str, *, top_k: int = 5, rerank: bool = False, hybrid: bool = False,
               query_filter: dict | None = None, collection: str | None = None) -> list[dict]:
        store = self.store
        opened = False
        coll = collection if collection is not None else self.cfg.collection
        if collection is not None and collection != self.store.collection:
            store = Store(self.cfg.qdrant, collection)
            opened = True
        try:
            vector = self.embedder.embed(query)
            want_rerank = rerank and bool(self.cfg.reranker.url)
            fetch_k = top_k * _RERANK_FANOUT if want_rerank else top_k
            stats = self._stats if coll == self.cfg.collection else CorpusStats.load(stats_path(coll))
            if hybrid and stats.n_docs > 0:
                sparse = to_qdrant(query_sparse(query, stats))
                hits = store.query_hybrid(vector, sparse, top_k=fetch_k, query_filter=query_filter)
            else:
                hits = store.search(vector, top_k=fetch_k, query_filter=query_filter)
            if want_rerank:
                hits = self.reranker.rerank(query, hits, top_k=top_k)
            else:
                hits = hits[:top_k]
            return [self._to_result(h) for h in hits]
        finally:
            if opened:
                store.close()
```

- [ ] **Step 4: verify** — `python -c "import ragkit.search"` OK; `pytest -q` green (search has no unit tests; existing suite unaffected).
- [ ] **Step 5: commit** — `git add ragkit/search.py && git commit -m "feat(search): hybrid dense+sparse routing with dense fallback"`

---

### Task 3: `--hybrid` CLI flags + DRY the ingest stats path

**Files:** Modify `ragkit/cli.py`.

- [ ] **Step 1: ingest uses the helper.** In `cmd_ingest`, replace
`sparse_stats_path = os.path.join(".ragkit", "sparse", f"{collection}.json")` with:

```python
    from .sparse import stats_path
    sparse_stats_path = stats_path(collection)
```

- [ ] **Step 2: search command.** In `cmd_search`, change the `searcher.search(...)` call to pass `hybrid=args.hybrid`:

```python
        hits = searcher.search(args.query, top_k=args.top_k, rerank=args.rerank, hybrid=args.hybrid)
```

and add to `p_search` in `build_parser()` (next to `--rerank`):

```python
    p_search.add_argument("--hybrid", action="store_true", help="dense + BM25 sparse, RRF-fused")
```

- [ ] **Step 3: eval command.** In `cmd_eval`, change `search_fn` to pass hybrid:

```python
        def search_fn(query: str, top_k: int) -> list[str]:
            hits = searcher.search(query, top_k=top_k * _DISTINCT_FANOUT,
                                   rerank=args.rerank, hybrid=args.hybrid)
            return [h["source_uri"] for h in hits]
```

and add to `p_eval`:

```python
    p_eval.add_argument("--hybrid", action="store_true", help="dense + BM25 sparse, RRF-fused")
```

- [ ] **Step 4: verify** — `python -m ragkit.cli search --help` and `eval --help` show `--hybrid`; `pytest -q` green.
- [ ] **Step 5: commit** — `git add ragkit/cli.py && git commit -m "feat(cli): --hybrid flag on search and eval"`

---

## Acceptance
1. `pytest -q` green (existing + new query_hybrid/stats_path tests), network-free.
2. `python -c "import ragkit.search, ragkit.store, ragkit.sparse, ragkit.cli"` OK.
3. `search --help` / `eval --help` show `--hybrid`; default off ⇒ dense behavior unchanged.
4. Committed on branch `ragkit-hybrid-3c-search`; master untouched.

## Notes for the smoke (not build steps)
- Live A/B: ingest the sample corpus (fresh collection), then `ragkit eval <c> --k 1,3,5 --out dense.json` and `ragkit eval <c> --hybrid --baseline dense.json`. On 4 docs expect ~parity; the real check is that `--hybrid` runs the two-leg query without error and a rare-exact-term query (e.g. a specific token) surfaces its doc.
- MCP `rag_search` hybrid exposure is out of scope (future).
