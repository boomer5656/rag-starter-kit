# ragkit eval harness — Implementation Plan

> **For agentic workers:** Implement task-by-task, TDD. Steps use checkbox (`- [ ]`) syntax. Stay on branch `ragkit-eval-harness`. Do NOT merge to master, do NOT deploy, do NOT touch the tower.

**Goal:** Add a retrieval eval harness to ragkit — a local-synthetic golden set plus `ragkit eval-gen`/`ragkit eval` commands scoring recall@k + MRR at doc level, with before/after compare.

**Architecture:** One new core module `ragkit/eval.py` holding pure metric/generation logic (dependency-injected via `gen_fn`/`search_fn` callables so tests stay network-free), a `Store.scroll_points` addition to read stored docs back, `EvalConfig` in config, and two thin CLI commands that wire the real Ollama client + `Searcher`.

**Tech Stack:** Python 3, stdlib + `httpx` only. pytest. No new dependencies. No ORM/pydantic in core.

## Global Constraints (verbatim from spec)

- stdlib + `httpx`/`pyyaml` only in core; **no ORM, no pydantic**; plain dataclasses.
- Config single-source: everything through `ragkit.yaml`; never hardcode endpoints/IPs.
- Fail-loud: never silently drop; per-item errors isolated + surfaced, run never aborts.
- Relevance is **doc-level** (`source_uri`). Metrics: recall@k + MRR@k only (no nDCG in v1).
- Tests are **network-free** (no Ollama/Qdrant/Tika); LF line endings; mirror `tests/test_chunk.py` style.
- Thin CLI: metric/generation logic lives in `eval.py`, not `cli.py`.
- A query/doc that errors counts as a **0** (miss) and is surfaced by count — never dropped.

---

### Task 1: Metric + golden/report logic in `ragkit/eval.py`

**Files:**
- Create: `ragkit/eval.py`
- Test: `tests/test_eval.py`

**Interfaces:**
- Produces: `GoldenItem(query, relevant_uris)`, `EvalReport(...)`, `distinct_in_order(uris)->list[str]`, `recall_at_k(retrieved_distinct, relevant, k)->float`, `first_relevant_rank(retrieved_distinct, relevant)->int|None`, `generate_golden(docs, gen_fn, per_doc_n, *, min_query_len=15)->(list[GoldenItem], int)`, `evaluate(golden, search_fn, k_values)->(dict[int,float], float, list[dict], int)`, `compare(baseline, current)->dict`, `save_golden/load_golden`, `save_report/load_report`.

- [ ] **Step 1: Write `tests/test_eval.py`** (network-free, fakes only)

```python
from __future__ import annotations

import pytest

from ragkit.eval import (
    EvalReport, GoldenItem, compare, distinct_in_order, evaluate,
    first_relevant_rank, generate_golden, load_golden, load_report,
    recall_at_k, save_golden, save_report,
)


def test_distinct_in_order_collapses_and_preserves_first():
    assert distinct_in_order(["a", "b", "a", "c", ""]) == ["a", "b", "c"]


def test_recall_at_k_hit_and_miss():
    assert recall_at_k(["a", "b", "c"], ["b"], 2) == 1.0
    assert recall_at_k(["a", "b", "c"], ["z"], 3) == 0.0
    assert recall_at_k(["a", "b", "c"], ["a", "z"], 3) == 0.5   # multi-relevant partial


def test_first_relevant_rank():
    assert first_relevant_rank(["a", "b", "c"], ["b"]) == 2
    assert first_relevant_rank(["a", "b"], ["z"]) is None


def test_generate_golden_respects_n_and_filters():
    def gen(text, n):
        return ["What is the cap rate for this property?", "hi", "What is the cap rate for this property?"]
    items, skipped = generate_golden([("file:///a.pdf", "body")], gen, per_doc_n=3)
    assert skipped == 0
    # "hi" dropped (too short), dup dropped -> one item
    assert [i.query for i in items] == ["What is the cap rate for this property?"]
    assert items[0].relevant_uris == ["file:///a.pdf"]


def test_generate_golden_drops_filename_leak():
    def gen(text, n):
        return ["What does a1234.pdf say about the roof replacement?"]
    items, _ = generate_golden([("file:///a1234.pdf", "body")], gen, per_doc_n=1)
    assert items == []


def test_generate_golden_isolates_failing_doc():
    def gen(text, n):
        if text == "boom":
            raise RuntimeError("model down")
        return ["What is the total monthly rent collected across units?"]
    docs = [("file:///a.pdf", "boom"), ("file:///b.pdf", "ok")]
    items, skipped = generate_golden(docs, gen, per_doc_n=1)
    assert skipped == 1
    assert [i.relevant_uris for i in items] == [["file:///b.pdf"]]


def test_evaluate_metrics_and_error_isolation():
    golden = [
        GoldenItem("q1", ["a"]),   # a at rank 1
        GoldenItem("q2", ["b"]),   # b at rank 2 (after dup collapse)
        GoldenItem("q3", ["z"]),   # boom -> errored, miss
    ]

    def search_fn(query, top_k):
        if query == "q1":
            return ["a", "a", "x"]
        if query == "q2":
            return ["x", "b", "y"]
        raise RuntimeError("search down")

    recall, mrr, per_query, errored = evaluate(golden, search_fn, [1, 3])
    assert errored == 1
    assert recall[1] == pytest.approx(1 / 3)     # only q1 hits at k=1
    assert recall[3] == pytest.approx(2 / 3)     # q1,q2 hit within 3; q3 miss
    assert mrr == pytest.approx((1.0 + 0.5 + 0.0) / 3)
    assert per_query[2]["first_rank"] is None and per_query[2]["retrieved_uris"] == []


def test_compare_deltas_and_warnings():
    base = EvalReport("c", 3, [1, 3], False, "m", {1: 0.3, 3: 0.6}, 0.4, [])
    cur = EvalReport("c", 3, [1, 3], True, "m", {1: 0.5, 3: 0.7}, 0.55, [])
    out = compare(base, cur)
    assert out["recall_at_k"][1] == (0.3, 0.5, pytest.approx(0.2))
    assert out["mrr"][2] == pytest.approx(0.15)
    assert out["warnings"] == []
    mism = compare(base, EvalReport("d", 4, [1], True, "m", {1: 0.5}, 0.5, []))
    assert any("collection differs" in w for w in mism["warnings"])


def test_golden_and_report_roundtrip(tmp_path):
    items = [GoldenItem("q?", ["file:///a.pdf"])]
    gp = str(tmp_path / "g.jsonl")
    save_golden(items, gp)
    assert load_golden(gp) == items
    rep = EvalReport("c", 1, [1, 5], False, None, {1: 0.0, 5: 1.0}, 0.5, [{"query": "q?"}])
    rp = str(tmp_path / "r.json")
    save_report(rep, rp)
    assert load_report(rp) == rep
```

- [ ] **Step 2: Run, verify it fails** — `pytest tests/test_eval.py -q` → FAIL (ModuleNotFoundError: ragkit.eval).

- [ ] **Step 3: Create `ragkit/eval.py`**

```python
"""Retrieval eval harness: build a local-synthetic golden set, score a
collection with recall@k + MRR at document granularity, and compare runs.

Pure logic here. `generate_golden` and `evaluate` take injected callables
(`gen_fn` / `search_fn`) so the metric math is unit-tested without touching
Ollama or Qdrant — the CLI wires the real Ollama client and Searcher.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Callable, Iterable, Optional


@dataclass
class GoldenItem:
    query: str
    relevant_uris: list[str]        # doc-level; usually length 1


@dataclass
class EvalReport:
    collection: str
    n_queries: int
    k_values: list[int]
    rerank: bool
    gen_model: Optional[str]
    recall_at_k: dict[int, float]   # k -> mean recall@k
    mrr: float
    per_query: list[dict] = field(default_factory=list)


GenFn = Callable[[str, int], list[str]]     # (doc_text, n) -> questions
SearchFn = Callable[[str, int], list[str]]  # (query, top_k) -> ranked source_uris (dups ok)


def distinct_in_order(uris: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for u in uris:
        if u and u not in seen:
            seen.add(u)
            out.append(u)
    return out


def recall_at_k(retrieved_distinct: list[str], relevant: list[str], k: int) -> float:
    if not relevant:
        return 0.0
    topk = set(retrieved_distinct[:k])
    return len(topk & set(relevant)) / len(relevant)


def first_relevant_rank(retrieved_distinct: list[str], relevant: list[str]) -> Optional[int]:
    rel = set(relevant)
    for i, u in enumerate(retrieved_distinct):
        if u in rel:
            return i + 1
    return None


def generate_golden(docs: Iterable[tuple[str, str]], gen_fn: GenFn, per_doc_n: int,
                    *, min_query_len: int = 15) -> tuple[list[GoldenItem], int]:
    """Return (items, skipped_docs). One GoldenItem per accepted question, relevance =
    the doc it came from. A gen_fn that raises for a doc skips only that doc."""
    items: list[GoldenItem] = []
    skipped = 0
    for uri, text in docs:
        try:
            questions = gen_fn(text, per_doc_n)
        except Exception:
            skipped += 1
            continue
        tail = uri.rsplit("/", 1)[-1].lower()
        seen: set[str] = set()
        for q in questions:
            q = (q or "").strip()
            low = q.lower()
            if len(q) < min_query_len or low in seen:
                continue
            if tail and tail in low:      # drop questions that leak the filename
                continue
            seen.add(low)
            items.append(GoldenItem(query=q, relevant_uris=[uri]))
    return items, skipped


def evaluate(golden: list[GoldenItem], search_fn: SearchFn,
             k_values: list[int]) -> tuple[dict[int, float], float, list[dict], int]:
    """Return (recall_at_k, mrr, per_query, errored). An errored query counts as a 0
    and is surfaced via `errored`, never dropped."""
    maxk = max(k_values)
    recall_sums = {k: 0.0 for k in k_values}
    rr_sum = 0.0
    errored = 0
    per_query: list[dict] = []
    for item in golden:
        try:
            retrieved = distinct_in_order(search_fn(item.query, maxk))
        except Exception:
            retrieved = []
            errored += 1
        rank = first_relevant_rank(retrieved, item.relevant_uris)
        rr_sum += (1.0 / rank) if rank else 0.0
        for k in k_values:
            recall_sums[k] += recall_at_k(retrieved, item.relevant_uris, k)
        per_query.append({
            "query": item.query,
            "relevant_uris": item.relevant_uris,
            "retrieved_uris": retrieved[:maxk],
            "first_rank": rank,
        })
    n = len(golden) or 1
    return {k: recall_sums[k] / n for k in k_values}, rr_sum / n, per_query, errored


def compare(baseline: EvalReport, current: EvalReport) -> dict:
    warnings: list[str] = []
    if baseline.collection != current.collection:
        warnings.append(f"collection differs: {baseline.collection!r} vs {current.collection!r}")
    if baseline.k_values != current.k_values:
        warnings.append(f"k_values differ: {baseline.k_values} vs {current.k_values}")
    if baseline.n_queries != current.n_queries:
        warnings.append(f"n_queries differ: {baseline.n_queries} vs {current.n_queries}")
    ks = [k for k in current.k_values if k in baseline.recall_at_k and k in current.recall_at_k]
    recall = {k: (baseline.recall_at_k[k], current.recall_at_k[k],
                  current.recall_at_k[k] - baseline.recall_at_k[k]) for k in ks}
    mrr = (baseline.mrr, current.mrr, current.mrr - baseline.mrr)
    return {"recall_at_k": recall, "mrr": mrr, "warnings": warnings}


def save_golden(items: list[GoldenItem], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for it in items:
            f.write(json.dumps({"query": it.query, "relevant_uris": it.relevant_uris}) + "\n")


def load_golden(path: str) -> list[GoldenItem]:
    items: list[GoldenItem] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            items.append(GoldenItem(query=d["query"], relevant_uris=list(d["relevant_uris"])))
    return items


def save_report(report: EvalReport, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    d = asdict(report)
    d["recall_at_k"] = {str(k): v for k, v in report.recall_at_k.items()}  # JSON keys are strings
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(d, f, indent=2)


def load_report(path: str) -> EvalReport:
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    return EvalReport(
        collection=d["collection"], n_queries=d["n_queries"],
        k_values=[int(k) for k in d["k_values"]], rerank=d["rerank"],
        gen_model=d.get("gen_model"),
        recall_at_k={int(k): v for k, v in d["recall_at_k"].items()},
        mrr=d["mrr"], per_query=d.get("per_query", []),
    )
```

- [ ] **Step 4: Run, verify pass** — `pytest tests/test_eval.py -q` → all pass.
- [ ] **Step 5: Commit** — `git add ragkit/eval.py tests/test_eval.py && git commit -m "feat(eval): metric + golden/report core (network-free, injected fns)"`

---

### Task 2: `Store.scroll_points` to read stored docs back

**Files:**
- Modify: `ragkit/store.py` (add method after `search`, before `close`)
- Test: `tests/test_eval.py` is network-free, so `scroll_points` (network) is verified manually, not unit-tested — matches the existing convention where `Store.search`/`upsert` have no unit tests.

**Interfaces:**
- Produces: `Store.scroll_points(page: int = 256) -> Iterator[dict]` yielding raw Qdrant points (`{id, payload, ...}`).

- [ ] **Step 1: Add the method** to `ragkit/store.py` (insert immediately before `def close`):

```python
    def scroll_points(self, page: int = 256):
        """Yield every stored point (payload only) via Qdrant's scroll API.
        Used by the eval harness to reconstruct source docs from the collection."""
        offset = None
        while True:
            body: dict = {"limit": page, "with_payload": True, "with_vector": False}
            if offset is not None:
                body["offset"] = offset
            r = self._client.post(self._url("/points/scroll"), json=body)
            r.raise_for_status()
            result = r.json().get("result", {})
            points = result.get("points", [])
            for p in points:
                yield p
            offset = result.get("next_page_offset")
            if offset is None or not points:
                break
```

- [ ] **Step 2: Verify import compiles** — `python -c "import ragkit.store"` → no error.
- [ ] **Step 3: Commit** — `git add ragkit/store.py && git commit -m "feat(store): scroll_points for reading a collection back"`

---

### Task 3: `EvalConfig` in `ragkit/config.py`

**Files:**
- Modify: `ragkit/config.py` (add dataclass, add field to `Config`, wire into `Config.load`)
- Modify: `ragkit.example.yaml` (document the block)

**Interfaces:**
- Produces: `EvalConfig(golden_path, gen_model, per_doc_n, k_values)`; `Config.eval`.

- [ ] **Step 1: Add `EvalConfig`** after `ChunkConfig` (before `class Config`):

```python
@dataclass
class EvalConfig:
    golden_path: str = ".ragkit/eval/{collection}.golden.jsonl"   # {collection} templated at use
    gen_model: Optional[str] = None    # None -> falls back to ollama.gate_model
    per_doc_n: int = 3
    k_values: list[int] = field(default_factory=lambda: [1, 3, 5, 10])
```

- [ ] **Step 2: Add the field** to `Config` (after `chunk: ChunkConfig = ...`, before `gate_enabled`):

```python
    eval: EvalConfig = field(default_factory=EvalConfig)
```

- [ ] **Step 3: Wire into `Config.load`** — add to the `Config(...)` constructor call (after the `chunk=` line):

```python
            eval=EvalConfig(**(data.get("eval") or {})),
```

- [ ] **Step 4: Document in `ragkit.example.yaml`** — append (match the file's existing commented style):

```yaml

# Eval harness (ragkit eval-gen / ragkit eval). Doc-level recall@k + MRR.
eval:
  golden_path: ".ragkit/eval/{collection}.golden.jsonl"
  # gen_model: null            # null -> uses ollama.gate_model to write synthetic questions
  per_doc_n: 3                  # questions generated per document
  k_values: [1, 3, 5, 10]
```

- [ ] **Step 5: Verify** — `python -c "from ragkit.config import Config; c=Config(); print(c.eval.k_values)"` → `[1, 3, 5, 10]`. Then `pytest -q` → existing suite still green.
- [ ] **Step 6: Commit** — `git add ragkit/config.py ragkit.example.yaml && git commit -m "feat(config): EvalConfig block"`

---

### Task 4: `eval-gen` + `eval` CLI commands

**Files:**
- Modify: `ragkit/cli.py` (imports, two helper wirings, two `cmd_*`, two subparsers)

**Interfaces:**
- Consumes: `ragkit.eval` (Task 1), `Store.scroll_points` (Task 2), `Config.eval` (Task 3), existing `Searcher`, `Store`.

- [ ] **Step 1: Add imports** near the top of `cli.py` (with the other `from .` imports):

```python
import httpx

from .eval import (
    EvalReport, compare, evaluate, generate_golden, load_golden, load_report,
    save_golden, save_report,
)
```

- [ ] **Step 2: Add wiring helpers + commands** (place after `cmd_search`, before `cmd_status`):

```python
_DISTINCT_FANOUT = 8   # over-fetch chunk hits so we get enough *distinct* docs for recall@maxk


def _docs_from_store(store: Store, char_cap: int = 6000) -> dict[str, str]:
    """Reconstruct {source_uri: text} from stored chunks, in ordinal order, capped."""
    by_uri: dict[str, list[tuple[int, str]]] = {}
    for p in store.scroll_points():
        pl = p.get("payload") or {}
        uri = pl.get("source_uri")
        if not uri:
            continue
        by_uri.setdefault(uri, []).append((pl.get("ordinal", 0), pl.get("text", "")))
    docs: dict[str, str] = {}
    for uri, parts in by_uri.items():
        parts.sort(key=lambda t: t[0])
        docs[uri] = "\n".join(t[1] for t in parts)[:char_cap]
    return docs


def _make_gen_fn(client: httpx.Client, ollama_url: str, model: str):
    system = (
        "You write evaluation questions for a document retrieval system. "
        "Output ONLY JSON: {\"questions\": [\"...\"]}. Each question must be answerable "
        "from the document, realistic, and MUST NOT mention the document, its title, or filename."
    )

    def gen(doc_text: str, n: int) -> list[str]:
        prompt = (f"Write {n} distinct questions a user would ask that this document answers.\n\n"
                  f"Document:\n{doc_text}\n\nReturn JSON: {{\"questions\": [...]}}")
        r = client.post(f"{ollama_url}/api/generate", json={
            "model": model, "system": system, "prompt": prompt,
            "stream": False, "format": "json", "options": {"temperature": 0.2},
        })
        r.raise_for_status()
        raw = r.json().get("response")
        if not raw:
            raise RuntimeError(f"gen: empty response from {model}")
        import json as _json
        data = _json.loads(raw)
        qs = data.get("questions") if isinstance(data, dict) else data
        return [str(q) for q in (qs or [])][:n]

    return gen


def cmd_eval_gen(args: argparse.Namespace) -> int:
    cfg = Config.load(args.config)
    if args.collection:
        cfg.collection = args.collection
    model = args.model or cfg.eval.gen_model or cfg.ollama.gate_model
    out = args.out or cfg.eval.golden_path.format(collection=cfg.collection)
    store = Store(cfg.qdrant, cfg.collection)
    client = httpx.Client(timeout=120.0)
    try:
        docs = _docs_from_store(store)
        if not docs:
            print(f"no stored docs in collection '{cfg.collection}' — ingest first")
            return 1
        items, skipped = generate_golden(docs.items(), _make_gen_fn(client, cfg.ollama.url, model),
                                          args.per_doc)
        save_golden(items, out)
        print(f"generated {len(items)} questions from {len(docs)} docs "
              f"({skipped} skipped) -> {out}")
        return 0
    finally:
        client.close()
        store.close()


def cmd_eval(args: argparse.Namespace) -> int:
    cfg = Config.load(args.config)
    if args.collection:
        cfg.collection = args.collection
    k_values = [int(x) for x in args.k.split(",")] if args.k else cfg.eval.k_values
    golden_path = args.golden or cfg.eval.golden_path.format(collection=cfg.collection)
    golden = load_golden(golden_path)
    searcher = Searcher(cfg)
    try:
        def search_fn(query: str, top_k: int) -> list[str]:
            hits = searcher.search(query, top_k=top_k * _DISTINCT_FANOUT, rerank=args.rerank)
            return [h["source_uri"] for h in hits]

        recall, mrr, per_query, errored = evaluate(golden, search_fn, k_values)
        report = EvalReport(
            collection=cfg.collection, n_queries=len(golden), k_values=k_values,
            rerank=args.rerank, gen_model=None, recall_at_k=recall, mrr=mrr, per_query=per_query,
        )
        print(f"collection: {report.collection}   queries: {report.n_queries}   "
              f"rerank: {report.rerank}   errored: {errored}")
        for k in k_values:
            print(f"  recall@{k:<3} {report.recall_at_k[k]:.3f}")
        print(f"  MRR      {report.mrr:.3f}")
        if args.out:
            save_report(report, args.out)
            print(f"report -> {args.out}")
        if args.baseline:
            cmp = compare(load_report(args.baseline), report)
            print("\nvs baseline:")
            for w in cmp["warnings"]:
                print(f"  ! {w}")
            for k, (b, c, dlt) in cmp["recall_at_k"].items():
                print(f"  recall@{k:<3} {b:.3f} -> {c:.3f} ({dlt:+.3f})")
            b, c, dlt = cmp["mrr"]
            print(f"  MRR      {b:.3f} -> {c:.3f} ({dlt:+.3f})")
        return 0
    finally:
        searcher.close()
```

- [ ] **Step 3: Register subparsers** in `build_parser()` (after the `p_search` block, before `p_status`):

```python
    p_eval_gen = sub.add_parser("eval-gen", help="generate a synthetic golden set from a collection")
    p_eval_gen.add_argument("collection", nargs="?", default=None)
    p_eval_gen.add_argument("--per-doc", type=int, default=3, dest="per_doc")
    p_eval_gen.add_argument("--out", default=None)
    p_eval_gen.add_argument("--model", default=None)
    p_eval_gen.add_argument("--config", default="ragkit.yaml")
    p_eval_gen.set_defaults(func=cmd_eval_gen)

    p_eval = sub.add_parser("eval", help="score retrieval (recall@k + MRR) against a golden set")
    p_eval.add_argument("collection", nargs="?", default=None)
    p_eval.add_argument("--golden", default=None)
    p_eval.add_argument("--k", default=None, help="comma-separated, e.g. 1,3,5,10")
    p_eval.add_argument("--rerank", action="store_true")
    p_eval.add_argument("--out", default=None)
    p_eval.add_argument("--baseline", default=None)
    p_eval.add_argument("--config", default="ragkit.yaml")
    p_eval.set_defaults(func=cmd_eval)
```

Note: `collection` is a positional consumed into `args.collection`; the `cmd_*` fall back to `cfg.collection` when it's `None`, matching the `--collection` pattern used elsewhere.

- [ ] **Step 4: Verify wiring** — `python -m ragkit.cli eval --help` and `... eval-gen --help` print without error; `pytest -q` still green.
- [ ] **Step 5: Commit** — `git add ragkit/cli.py && git commit -m "feat(cli): eval-gen and eval commands"`

---

## Acceptance criteria (verify at end)

1. `pytest -q` → all pass (existing + new `test_eval.py`), network-free.
2. `python -m ragkit.cli eval --help` / `eval-gen --help` succeed.
3. `python -c "import ragkit.eval, ragkit.store, ragkit.cli"` → no import errors.
4. No changes to ingest/search/store behavior beyond the additive `scroll_points`.
5. All work committed on branch `ragkit-eval-harness`; master untouched; nothing deployed.

## Out of scope (do NOT touch)
- The `--connector literature` CLI bug and `serve-mcp --config` forwarding bug (tracked separately).
- Contextual retrieval and hybrid retrieval (sub-projects 2 and 3).
