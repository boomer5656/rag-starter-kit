# ragkit CRAG (corrective retrieval) — Implementation Plan

> **For agentic workers:** TDD, task by task. Stay on branch `crag-corrective`. Do NOT merge/push. Steps use `- [ ]`.

**Goal:** A `--crag` path: grade the retrieved pool for relevance with a local model, drop clearly off-topic hits, and — when the whole pool grades weak — trigger ONE corrective re-retrieval with a rewritten query (reusing `_expand`). Default off.

**Architecture:** Pure decision logic in `ragkit/crag.py` (`parse_grades`, `apply_grades`, `is_weak`) — unit-tested. `Searcher` gains a `_grade` (batched local-LLM relevance call via `ollama.generate_json`) and a `_crag_correct` step inserted between retrieval and rerank. Config `crag` block + `--crag` flag.

**Tech Stack:** stdlib + httpx; reuses `ollama.generate_json` + `expand.rrf_fuse` + the `_expand` rewrite + `_retrieve_hits`. No new deps.

## Honesty note (bake into the design)
CRAG is a **precision** technique. Its filter can drop a low-graded-but-relevant doc, so on the recall@k harness the filter is neutral-to-slightly-negative; the **weak-pool fallback** is the recall-measurable win (it rescues queries the first pass missed). So: the filter is CONSERVATIVE (low drop threshold), grading fails **safe** (parse failure → neutral 0.5 → no drop, no false trigger), and CRAG defaults OFF.

## Global Constraints
- stdlib + httpx; config single-source; thin CLI; best-effort (grading/expansion failure ⇒ degrade gracefully, never throw).
- `crag=False` (default) ⇒ byte-for-byte current `search()` behavior.
- Tests network-free (pure logic; the LLM grade call is not unit-tested).

---

### Task 1: `ragkit/crag.py` + tests

**Files:** Create `ragkit/crag.py`, `tests/test_crag.py`.
**Interfaces — Produces:** `parse_grades(raw, n)->list[float]`, `apply_grades(hits, grades, drop=DROP_THRESHOLD)->(list[dict], float)`, `is_weak(best, floor=FALLBACK_FLOOR)->bool`, `DROP_THRESHOLD`, `FALLBACK_FLOOR`.

- [ ] **Step 1: `tests/test_crag.py`**

```python
from __future__ import annotations

from ragkit.crag import DROP_THRESHOLD, apply_grades, is_weak, parse_grades


def test_parse_grades_clamps_and_orders():
    assert parse_grades({"grades": [0.9, -1, 2, "x"]}, 4) == [0.9, 0.0, 1.0, 0.5]


def test_parse_grades_failsafe_neutral_on_junk():
    assert parse_grades({"nope": 1}, 3) == [0.5, 0.5, 0.5]
    assert parse_grades("garbage", 2) == [0.5, 0.5]
    assert parse_grades({"grades": [0.8]}, 3) == [0.8, 0.5, 0.5]   # short -> padded neutral


def test_apply_grades_drops_below_threshold_keeps_order_and_best():
    hits = [{"id": 1}, {"id": 2}, {"id": 3}]
    kept, best = apply_grades(hits, [0.9, 0.1, 0.6], drop=0.25)
    assert [h["id"] for h in kept] == [1, 3]      # id 2 (0.1) dropped, order preserved
    assert kept[0]["crag_grade"] == 0.9           # grade attached
    assert best == 0.9


def test_is_weak_trigger():
    assert is_weak(0.3, floor=0.5) is True
    assert is_weak(0.7, floor=0.5) is False
    assert is_weak(0.0) is True
```

- [ ] **Step 2: run → fails.**
- [ ] **Step 3: create `ragkit/crag.py`**

```python
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
```

- [ ] **Step 4: run → pass.**
- [ ] **Step 5: commit** — `git add ragkit/crag.py tests/test_crag.py && git commit -m "feat(crag): grade/filter/weak-pool decision helpers"`

---

### Task 2: `Searcher` CRAG step

**Files:** Modify `ragkit/search.py`.
**Interfaces — Consumes:** `ragkit.crag`, existing `_expand`/`_retrieve_hits`/`rrf_fuse`/`generate_json`. Produces: `Searcher.search(..., crag: bool = False)`.

- [ ] **Step 1: import + constant.** Add to `search.py`:

```python
from .crag import apply_grades, is_weak, parse_grades
```

and near the other module constants (e.g. after `_SNIPPET_CHARS`):

```python
_CRAG_POOL = 10   # cap on passages sent to the grader per query (one batched LLM call)
```

- [ ] **Step 2: add `_GRADE_SYSTEM` + `_grade` + `_crag_correct`** (methods on `Searcher`, near `_expand`):

```python
    _GRADE_SYSTEM = (
        "You grade how relevant each numbered passage is to the user's query, from 0.0 (irrelevant) "
        "to 1.0 (directly answers it). Output ONLY JSON: {\"grades\": [0.0, ...]} — one number per "
        "passage, in order."
    )

    def _grade(self, query: str, hits: list[dict]) -> list[float]:
        """Relevance grade in [0,1] for each hit (one batched LLM call). Fails safe → neutral 0.5s."""
        if not hits:
            return []
        numbered = "\n\n".join(
            f"[{i}] {((h.get('payload') or {}).get('text', ''))[:600]}" for i, h in enumerate(hits))
        prompt = (f"Query: {query}\n\nPassages:\n{numbered}\n\n"
                  f"Return JSON: {{\"grades\": [...]}} with {len(hits)} numbers, in order.")
        try:
            model = self.cfg.crag.model or self.cfg.ollama.gate_model
            raw = generate_json(self._expand_client, self.cfg.ollama.url, model,
                                system=self._GRADE_SYSTEM, prompt=prompt)
            return parse_grades(raw, len(hits))
        except Exception:
            return [0.5] * len(hits)

    def _crag_correct(self, query: str, hits: list[dict], store: Store, fetch_k: int,
                      hybrid: bool, coll: str, query_filter: dict | None) -> list[dict]:
        """Grade the top pool, drop junk, and re-retrieve with a rewritten query if the pool is weak."""
        if not hits:
            return hits
        pool, tail = hits[:_CRAG_POOL], hits[_CRAG_POOL:]
        kept, best = apply_grades(pool, self._grade(query, pool))
        kept = kept + tail
        if is_weak(best):
            rewrites = self._expand(query, 1)
            if rewrites:
                extra = self._retrieve_hits(store, rewrites[0], fetch_k, hybrid, coll, query_filter)
                extra_kept, _ = apply_grades(extra[:_CRAG_POOL], self._grade(query, extra[:_CRAG_POOL]))
                kept = rrf_fuse([kept, extra_kept]) if kept else extra_kept
        return kept
```

- [ ] **Step 3: wire into `search()`.** Add `crag: bool = False` to the signature (after `multi`), and insert the CRAG step between the retrieve/fuse block and the rerank block:

```python
            if crag:
                hits = self._crag_correct(query, hits, store, fetch_k, hybrid, coll, query_filter)
            if want_rerank:
                hits = self.reranker.rerank(query, hits, top_k=top_k)
```

(i.e. CRAG transforms `hits` before the existing rerank/slice — unchanged otherwise.)

- [ ] **Step 4: verify** — `python -c "import ragkit.search"`; `pytest -q` green (crag defaults False ⇒ existing behavior).
- [ ] **Step 5: commit** — `git add ragkit/search.py && git commit -m "feat(search): CRAG grade/filter/corrective-refetch step (default off)"`

---

### Task 3: config + `--crag` CLI flag

**Files:** Modify `ragkit/config.py`, `ragkit/cli.py`, `ragkit.example.yaml`.

- [ ] **Step 1: config.** In `config.py`, add (near `MultiQueryConfig`):

```python
@dataclass
class CragConfig:
    model: Optional[str] = None          # None -> ollama.gate_model; a capable instruct model grades better
    drop_threshold: float = 0.25
    fallback_floor: float = 0.5
```

Add `crag: CragConfig = field(default_factory=CragConfig)` to `Config` and wire `crag=CragConfig(**(data.get("crag") or {}))` into `Config.load`.

- [ ] **Step 2: `cmd_search`** — add `--crag` (store_true) to `p_search`; pass `crag=args.crag` in the `searcher.search(...)` call.

- [ ] **Step 3: `cmd_eval`** — add `--crag` (store_true) to `p_eval`; thread `crag=args.crag` into the `search_fn`'s `searcher.search(...)` call.

- [ ] **Step 4: document in `ragkit.example.yaml`** (append):

```yaml

# Corrective RAG (ragkit search/eval --crag). Grade retrieved passages, drop off-topic ones, and
# re-retrieve with a rewritten query when the whole pool is weak. Precision-oriented; off by default.
crag:
  # model: null            # null -> ollama.gate_model; a capable instruct model grades better
  drop_threshold: 0.25
  fallback_floor: 0.5
```

- [ ] **Step 5: verify** — `python -m ragkit.cli search --help` and `eval --help` show `--crag`; `python -c "from ragkit.config import Config; print(Config().crag.drop_threshold)"` → 0.25; `pytest -q` green.
- [ ] **Step 6: commit** — `git add ragkit/config.py ragkit/cli.py ragkit.example.yaml && git commit -m "feat(cli): --crag flag on search and eval"`

---

## Acceptance
1. `pytest -q` green (existing + test_crag), network-free.
2. `python -c "import ragkit.crag, ragkit.search, ragkit.cli"` OK.
3. `--crag` unset / `crag=False` ⇒ byte-for-byte current search behavior; grading fails safe (neutral) on any model failure.
4. Committed on `crag-corrective`; master untouched.

## Notes for the smoke (not build steps)
Live: `ragkit eval <c> --hybrid` vs `--hybrid --crag`. Expect roughly-neutral recall (filter cost ≈ fallback gain on a clean corpus); the value shows on noisy corpora / weak-retrieval queries where the fallback fires. Grade model should be capable (≥~9B).
