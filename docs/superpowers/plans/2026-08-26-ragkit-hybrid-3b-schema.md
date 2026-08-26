# ragkit hybrid 3b — named dense+sparse schema + ingest

> **For agentic workers:** TDD, task by task. Stay on branch `ragkit-hybrid-3b-schema`. Do NOT merge/push/deploy. Steps use `- [ ]`.

**Goal:** Move ragkit collections to Qdrant **named vectors** (`dense`) + a **`sparse`** vector, store both at ingest (dense = embedder, sparse = BM25 from 3a), persist corpus stats per collection, and keep search working **dense-only** (hybrid fusion is 3c). This is a **breaking** schema change — existing collections must be re-ingested.

**Design:** BM25 "document" = a chunk (the retrieval unit). Corpus stats are rebuilt **fresh each ingest run** from that run's chunks (ragkit re-ingests whole folders idempotently, so a fresh rebuild avoids double-counting df on re-runs) and saved to `.ragkit/sparse/{collection}.json`. Two sub-passes in the embed phase: fold all batch chunks into stats → save → embed dense + encode sparse + store both. A pre-3b (unnamed-vector) collection raises a clear "recreate" error instead of failing cryptically on upsert.

**Tech Stack:** stdlib + httpx. Uses `ragkit.sparse` (3a). No new deps.

## Global Constraints
- stdlib + httpx; fail-loud per-doc isolation preserved; LF endings; tests network-free (fake httpx client).
- Sparse is always stored now (the schema requires it); pure-Python, cheap.
- `delete_source` + stable ids keep re-ingest idempotent.

---

### Task 1: `store.py` — named dense + sparse schema

**Files:** Modify `ragkit/store.py`; Test `tests/test_store.py` (new, fake-client).
**Interfaces — Produces:** `ensure_collection(dim, index_fields=None)` (named+sparse), `current_dim()` (reads named dense), `upsert(chunks, dense_vectors, sparse_vectors, source_type)`, `search(vector, top_k=5, query_filter=None)` (named "dense").

- [ ] **Step 1: write `tests/test_store.py`**

```python
from __future__ import annotations

import pytest

from ragkit.config import QdrantConfig
from ragkit.models import Chunk
from ragkit.store import Store


class _Resp:
    def __init__(self, status=200, payload=None):
        self.status_code = status
        self._p = payload or {}
    def raise_for_status(self): pass
    def json(self): return self._p


class _FakeClient:
    def __init__(self, get_status=404, get_payload=None):
        self.calls = []
        self._gs, self._gp = get_status, get_payload
    def get(self, url):
        self.calls.append(("GET", url, None)); return _Resp(self._gs, self._gp)
    def put(self, url, json=None):
        self.calls.append(("PUT", url, json)); return _Resp(200)
    def post(self, url, json=None):
        self.calls.append(("POST", url, json)); return _Resp(200, {"result": []})
    def close(self): pass


def _store(client):
    s = Store(QdrantConfig(), "c")
    s._client = client
    return s


def test_ensure_collection_creates_named_dense_and_sparse():
    s = _store(_FakeClient(get_status=404))
    s.ensure_collection(1024)
    body = next(c[2] for c in s._client.calls if c[0] == "PUT" and c[1].endswith("/collections/c"))
    assert body["vectors"]["dense"]["size"] == 1024
    assert body["vectors"]["dense"]["distance"] == "Cosine"
    assert "sparse" in body["sparse_vectors"]


def test_ensure_collection_rejects_legacy_unnamed_schema():
    legacy = {"result": {"config": {"params": {"vectors": {"size": 768, "distance": "Cosine"}}}}}
    s = _store(_FakeClient(get_status=200, get_payload=legacy))
    with pytest.raises(Exception):
        s.ensure_collection(1024)


def test_ensure_collection_noops_when_dense_dim_matches():
    named = {"result": {"config": {"params": {"vectors": {"dense": {"size": 1024, "distance": "Cosine"}}}}}}
    s = _store(_FakeClient(get_status=200, get_payload=named))
    s.ensure_collection(1024)
    assert not any(c[0] == "PUT" for c in s._client.calls)   # no create


def test_upsert_writes_named_dense_and_sparse():
    s = _store(_FakeClient())
    s.upsert([Chunk("u", 0, "t")], [[0.1, 0.2, 0.3, 0.4]],
             [{"indices": [7], "values": [0.5]}], source_type="files")
    pt = next(c[2] for c in s._client.calls if c[0] == "PUT" and "/points" in c[1])["points"][0]
    assert pt["vector"]["dense"] == [0.1, 0.2, 0.3, 0.4]
    assert pt["vector"]["sparse"] == {"indices": [7], "values": [0.5]}


def test_search_uses_named_dense_vector():
    s = _store(_FakeClient())
    s.search([0.1, 0.2], top_k=3)
    body = next(c[2] for c in s._client.calls if c[0] == "POST" and c[1].endswith("/points/search"))
    assert body["vector"] == {"name": "dense", "vector": [0.1, 0.2]}
    assert body["limit"] == 3
```

- [ ] **Step 2: run → fails** (import ok, assertions fail on old unnamed schema).
- [ ] **Step 3: edit `store.py`.** Replace `current_dim` and `ensure_collection` with:

```python
    def _vectors_cfg(self):
        r = self._client.get(self._url())
        if r.status_code != 200:
            return None
        return r.json().get("result", {}).get("config", {}).get("params", {}).get("vectors", {})

    def current_dim(self) -> int | None:
        v = self._vectors_cfg()
        if not isinstance(v, dict):
            return None
        dense = v.get("dense")
        if isinstance(dense, dict):
            return dense.get("size")
        return v.get("size")  # legacy unnamed collection (pre-hybrid)

    def ensure_collection(self, dim: int, index_fields: dict[str, str] | None = None) -> None:
        """Create a named-vector (dense) + sparse collection if absent; else assert its
        dense dim matches. A pre-hybrid unnamed collection is rejected — recreate it."""
        v = self._vectors_cfg()
        if v is not None:
            dense = v.get("dense") if isinstance(v, dict) else None
            if not isinstance(dense, dict):
                raise DimensionMismatch(
                    f"collection '{self.collection}' predates the hybrid (named-vector) schema. "
                    f"Delete it or use a fresh collection, then re-ingest."
                )
            if dense.get("size") != dim:
                raise DimensionMismatch(
                    f"collection '{self.collection}' is dim={dense.get('size')} but the embedder "
                    f"produces dim={dim}. Use a fresh collection or re-embed."
                )
            return
        self._client.put(
            self._url(),
            json={
                "vectors": {"dense": {"size": dim, "distance": self.cfg.distance}},
                "sparse_vectors": {"sparse": {}},
                "on_disk_payload": True,
            },
        ).raise_for_status()
        for field_name, schema in (index_fields or {}).items():
            self._client.put(self._url("/index"),
                             json={"field_name": field_name, "field_schema": schema})
```

- [ ] **Step 4: edit `store.py` `upsert`** — add sparse param + named vectors:

```python
    def upsert(self, chunks: list[Chunk], dense_vectors: list[list[float]],
               sparse_vectors: list[dict], source_type: str) -> None:
        assert len(chunks) == len(dense_vectors) == len(sparse_vectors)
        points = []
        for ch, dv, sv in zip(chunks, dense_vectors, sparse_vectors):
            payload = {
                "text": ch.text,
                "source_uri": ch.source_uri,
                "ordinal": ch.ordinal,
                "source_type": source_type,
                **ch.meta,
            }
            points.append({"id": ch.id, "vector": {"dense": dv, "sparse": sv}, "payload": payload})
        if points:
            self._client.put(self._url("/points?wait=true"),
                             json={"points": points}).raise_for_status()
```

- [ ] **Step 5: edit `store.py` `search`** — query the named dense vector:

```python
    def search(self, vector: list[float], top_k: int = 5,
               query_filter: dict | None = None) -> list[dict]:
        body: dict = {"vector": {"name": "dense", "vector": vector},
                      "limit": top_k, "with_payload": True}
        if query_filter:
            body["filter"] = query_filter
        r = self._client.post(self._url("/points/search"), json=body)
        r.raise_for_status()
        return r.json().get("result", [])
```

- [ ] **Step 6: run tests** — `pytest tests/test_store.py -q` → pass.
- [ ] **Step 7: commit** — `git add ragkit/store.py tests/test_store.py && git commit -m "feat(store): named dense + sparse collection schema (hybrid 3b)"`

---

### Task 2: `pipeline.py` — store sparse + persist corpus stats

**Files:** Modify `ragkit/pipeline.py`.
**Interfaces — Consumes:** `ragkit.sparse` (3a), the new `store.upsert` (Task 1). Produces: `Pipeline(..., sparse_stats_path: str | None = None)`.

- [ ] **Step 1: imports.** Add at the top of `pipeline.py` with the other `from .` imports:

```python
from .sparse import CorpusStats, doc_sparse, to_qdrant
```

- [ ] **Step 2: constructor param.** Add `sparse_stats_path: str | None = None` to `Pipeline.__init__` (after `contextualizer`), and `self.sparse_stats_path = sparse_stats_path` in the body.

- [ ] **Step 3: replace the embed+store section** of `run()` (from the `# --- embed + store ...` comment through the end of that `for` loop) with a stats pass + a store pass:

```python
        # --- corpus stats for BM25 (fresh per run; a "document" is a chunk) ---
        stats = CorpusStats()
        for _d, _chunks in chunked:
            for c in _chunks:
                stats.add_doc(c.text)
        if self.sparse_stats_path:
            stats.save(self.sparse_stats_path)

        # --- embed + store (dense from the embedder, sparse from BM25) ---
        for d, chunks in chunked:
            try:
                texts = [c.text for c in chunks]
                if self.contextualizer is not None:
                    ctxs = self.contextualizer.contextualize(d.text or "", chunks)
                    texts = [f"{ctx}\n{c.text}" if ctx else c.text
                             for ctx, c in zip(ctxs, chunks)]
                vectors = self.embedder.embed_batch(texts)
                bad = next((v for v in vectors if len(v) != dim), None)
                if bad is not None:
                    raise ValueError(f"embedding dim {len(bad)} != expected {dim}")
                sparse = [to_qdrant(doc_sparse(c.text, stats)) for c in chunks]
                self.state.set(d.uri, State.EMBEDDED, dim=dim, n_chunks=len(chunks))
                res.embedded += len(chunks)
                self.store.delete_source(d.uri)
                self.store.upsert(chunks, vectors, sparse, source_type=source_type)
                self.state.set(d.uri, State.STORED, dim=dim, n_chunks=len(chunks))
                res.stored += len(chunks)
            except Exception as e:  # noqa: BLE001
                self._err(res, d.uri, f"embed/store: {e}")

        return res
```

(The `dim = self.embedder.dim` / `ensure_collection` block just above is unchanged.)

- [ ] **Step 4: verify** — `python -c "import ragkit.pipeline"` OK; `pytest -q` green (existing suite unaffected — pipeline has no unit tests).
- [ ] **Step 5: commit** — `git add ragkit/pipeline.py && git commit -m "feat(pipeline): store sparse vectors + persist corpus stats"`

---

### Task 3: `cli.py cmd_ingest` — pass the stats path

**Files:** Modify `ragkit/cli.py`.

- [ ] **Step 1: build the path + pass it.** In `cmd_ingest`, after the `contextualizer = None ...` block, add:

```python
    sparse_stats_path = os.path.join(".ragkit", "sparse", f"{collection}.json")
```

and add `sparse_stats_path=sparse_stats_path,` to the `Pipeline(...)` constructor call.

- [ ] **Step 2: verify** — `python -m ragkit.cli ingest --help` exit 0; `pytest -q` green.
- [ ] **Step 3: commit** — `git add ragkit/cli.py && git commit -m "feat(cli): wire sparse corpus-stats path into ingest"`

---

## Acceptance
1. `pytest -q` green (existing + test_store), network-free.
2. `python -c "import ragkit.store, ragkit.pipeline, ragkit.cli"` OK.
3. store.py: fresh collections are named `dense` + `sparse`; legacy unnamed collections raise a clear recreate error.
4. All work committed on branch `ragkit-hybrid-3b-schema`; master untouched.

## Notes for the reviewer/smoke (not build steps)
- Live smoke (after review) needs a FRESH collection (the schema changed) — ingest into a new name, confirm dense search still returns results and `.ragkit/sparse/{collection}.json` is written with df/n_docs.
- Search stays dense-only here; hybrid fusion is 3c.
