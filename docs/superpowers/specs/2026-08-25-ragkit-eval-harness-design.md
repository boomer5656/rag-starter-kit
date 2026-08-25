# ragkit eval harness — design spec

**Date:** 2026-08-25
**Status:** Approved (design), pending implementation plan
**Sub-project 1 of 3** in the ragkit RAG-quality track. Sequencing (approved): **eval harness → contextual retrieval → hybrid retrieval.** Eval goes first so the other two can be proven, not assumed.

## Purpose

ragkit ingests documents well but has **no way to measure retrieval quality**. Before adding contextual retrieval or hybrid search, we need a ruler: a repeatable harness that scores how well the current retriever surfaces the right documents, so each subsequent change is validated as a real lift (or reverted). This mirrors eval-driven RAG (the Anthropic cookbook's retrieval guide, and the RE dashboard's own `rag-eval` harness).

## Goals

- Generate a **golden set** (queries → known-relevant docs) automatically, using a local model, from an already-ingested collection.
- Score a collection's retrieval with **recall@k** and **MRR@k** at doc granularity.
- **Compare** two eval runs (before/after a change) and print metric deltas.
- Stay inside ragkit's conventions: stdlib + `httpx` only, plain dataclasses (no ORM/pydantic in core), config single-source, thin CLI, fail-loud, **network-free unit tests**.

## Non-goals (YAGNI)

- No nDCG / precision@k in v1 (add later only if a decision needs it).
- No chunk-level relevance labels (doc-level only — survives the re-chunking that contextual/hybrid will cause).
- No hand-labeled golden set, no human curation UI.
- No web dashboard; output is a stdout table + optional JSON report file.
- No new service/container; the harness is a CLI over the existing `Searcher`.

## Relevance granularity (decided)

Relevance is judged at **`source_uri` (doc) level**, not chunk level. Rationale: contextual retrieval and hybrid search both change chunk boundaries/ids, which would invalidate chunk-level labels; doc-level labels remain valid across those changes, so the same golden set measures all three features.

## Architecture

One new core module `ragkit/eval.py`, additions to `ragkit/cli.py` and `ragkit/config.py`, one new test file. No changes to ingest, store schema, or the existing `Searcher` behavior.

**Dependency injection is the central design choice.** The two logic functions take callables (`search_fn`, `gen_fn`) instead of constructing `Searcher`/Ollama clients themselves. The CLI wires the real implementations; tests wire fakes. This is what lets the metric/generation logic be unit-tested without touching Ollama/Qdrant, honoring `tests/conftest.py`'s network-free rule.

### `ragkit/eval.py`

Dataclasses (in `models.py` if colocating with existing models is cleaner; otherwise local to `eval.py` — implementer's call, keep consistent with repo style):

```
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
    gen_model: str | None
    recall_at_k: dict[int, float]   # k -> mean recall@k over queries
    mrr: float                      # mean reciprocal rank (first relevant doc)
    per_query: list[dict]           # {query, relevant_uris, retrieved_uris, first_rank|None}
```

Functions:

- `generate_golden(docs, gen_fn, per_doc_n, *, min_query_len=15) -> list[GoldenItem]`
  - `docs`: iterable of `(source_uri, text)` — the CLI builds this by reading stored points from Qdrant (payload has `text` + `source_uri`) and grouping by `source_uri` (concatenate or sample that doc's chunk texts, capped to a sane char budget, e.g. first ~6000 chars, to bound the prompt).
  - `gen_fn(doc_text, n) -> list[str]`: returns up to `n` questions. CLI supplies a local-Ollama implementation reusing `gate.py`'s `/api/generate` client pattern (`stream=False`, `format="json"`, `temperature=0`), prompt = "Write N distinct, realistic questions a user would ask that THIS document answers. Do not mention the document, its title, or filename." Parse a JSON array of strings.
  - Filter: drop empty / `< min_query_len` chars / questions naming the source file; dedup case-insensitively within a doc.
  - Fail-loud: if `gen_fn` raises for a doc, log + skip that doc, continue; caller reports skipped count.

- `evaluate(golden, search_fn, k_values) -> EvalReport`
  - `search_fn(query, top_k) -> list[str]`: returns **ranked chunk-hit `source_uri`s** (with duplicates, in hit order). CLI supplies an implementation calling the real `Searcher.search`, fetching enough chunk hits to yield `max(k_values)` distinct docs (fetch budget = `max(k_values) * DISTINCT_FANOUT`, `DISTINCT_FANOUT=8`, matching the spirit of search.py's rerank fanout).
  - For each item: collapse `search_fn` output to **ordered distinct `source_uri`s** (first occurrence wins the rank). Then:
    - `recall@k` = `|distinct_retrieved[:k] ∩ relevant_uris| / len(relevant_uris)` (usually 0/1 since 1 relevant).
    - reciprocal rank = `1/rank` of the first distinct retrieved uri that is in `relevant_uris`, else 0.
  - Aggregate: `recall_at_k[k]` = mean over queries; `mrr` = mean reciprocal rank.
  - Fail-loud: a query whose `search_fn` raises → record `first_rank=None`, `retrieved_uris=[]`, count it as a 0, continue; caller reports errored count.

- `compare(baseline: EvalReport, current: EvalReport) -> dict`
  - Returns `{recall_at_k: {k: (base, cur, delta)}, mrr: (base, cur, delta), warnings: [...]}`. Warn if `k_values`, `collection`, or `n_queries` differ (comparing across different golden sets is meaningless — surface it, don't hide it).

- JSONL helpers: `load_golden(path) -> list[GoldenItem]`, `save_golden(items, path)` — one JSON object per line. `load_report/save_report` — single JSON object.

### `ragkit/cli.py` (thin — no logic beyond wiring)

- `ragkit eval-gen <collection> [--per-doc N=3] [--out PATH] [--model M] [--config C]`
  - Reads stored points (scroll the Qdrant collection), groups by `source_uri`, builds the local-Ollama `gen_fn`, calls `generate_golden`, writes JSONL to `--out` (default `eval.golden_path` templated with collection). Prints count generated + docs skipped.
- `ragkit eval <collection> [--golden PATH] [--k 1,3,5,10] [--rerank] [--out PATH] [--baseline PATH] [--config C]`
  - Loads golden, builds `search_fn` from a real `Searcher(collection, rerank=...)`, calls `evaluate`, prints a metrics table (recall@k row per k, MRR) + errored-query count. If `--out`, save JSON report. If `--baseline`, load it, `compare`, print deltas (▲/▼).

Both commands registered in `build_parser()` alongside the existing `ingest`/`search`/`status`/`serve-mcp`.

### `ragkit/config.py`

Add `EvalConfig` dataclass + wire into `Config.load` (yaml) and, minimally, no env override needed (add later if asked):

```
@dataclass
class EvalConfig:
    golden_path: str = ".ragkit/eval/{collection}.golden.jsonl"   # {collection} templated at use
    gen_model: str | None = None     # None -> fall back to ollama.gate_model
    per_doc_n: int = 3
    k_values: list[int] = field(default_factory=lambda: [1, 3, 5, 10])
```

Document these keys in `ragkit.example.yaml` under a new `eval:` block, consistent with the existing commented style.

## Data formats

- **Golden set** — JSONL, one `GoldenItem` per line: `{"query": "...", "relevant_uris": ["file:///..."]}`. Lives at the configured `golden_path`; gitignore `.ragkit/` stays as-is (state.db already there).
- **Report** — single JSON: the `EvalReport` fields above. Human-diffable and machine-comparable.

## Metric definitions (precise, to remove ambiguity)

Given a query with relevant set `R` (doc uris) and the ordered distinct retrieved doc list `D`:
- **recall@k** = `|set(D[:k]) ∩ R| / |R|`.
- **reciprocal rank** = `1 / (i+1)` for the smallest `i` where `D[i] ∈ R`; `0` if none.
- **recall_at_k[k]** and **mrr** are the arithmetic means of the above across all golden queries. A query with an empty/errored retrieval contributes `0` to both (it is not dropped — dropping would hide failures).

## Error handling (fail-loud, per ARCHITECTURE.md)

- Generation: per-doc isolation; skip + count, never abort the run; never silently emit fewer than requested without reporting it.
- Evaluation: per-query isolation; errored query counts as a miss and is surfaced, never silently dropped.
- `compare`: mismatched `k_values`/`collection`/`n_queries` → explicit warning in the returned dict and printed, not a silent wrong comparison.

## Testing (`tests/test_eval.py`, pytest, network-free)

Pure-logic coverage via injected fakes:
- recall@k: relevant at rank 1, mid, absent (→0), multi-relevant partial, k larger than result set.
- MRR: first-relevant rank math, none-found → 0.
- **distinct-doc collapse**: duplicate `source_uri`s in hit order collapse to first-occurrence rank.
- `generate_golden` with a fake `gen_fn`: per-doc-n respected, degenerate/short/dup filtered, a raising `gen_fn` for one doc skips only that doc.
- golden + report JSONL/JSON round-trip.
- `compare` deltas + the mismatch warnings.
No live Ollama/Qdrant; fakes only. LF line endings; matches existing test style in `test_chunk.py`/`test_state.py`.

## Acceptance criteria

1. `ragkit eval-gen <collection>` produces a JSONL golden set from a populated collection; reports counts; degenerate questions filtered.
2. `ragkit eval <collection> --golden <file>` prints recall@1/3/5/10 + MRR and an errored-query count; `--out` writes a JSON report.
3. `ragkit eval ... --baseline old.json` prints per-metric deltas.
4. `--rerank` routes through the existing reranker seam and changes the numbers when a reranker URL is configured.
5. `pytest` passes with the new tests, all network-free.
6. No regression to existing `ingest`/`search`/`status` behavior or store schema.

## Two pre-existing bugs noted during mapping (out of scope, track separately)

- `--connector literature` is broken from the CLI (`cli.py:39-40` passes a path positionally into `LiteratureConnector(papers=...)`). Not touched here.
- `ragkit serve-mcp --config X` doesn't forward `X` to the spawned server. Not touched here.

## Build-routing note (dev-cost discipline)

Design/spec/review: Opus (done here). Implementation of this spec is well-specified mechanical work → route the build to **Sonnet** (or on-prem local coder for max token savings — ragkit source is proprietary, so **not** OmniRoute's external free pools). Reasoning-heavy metric-correctness review returns to Opus.
