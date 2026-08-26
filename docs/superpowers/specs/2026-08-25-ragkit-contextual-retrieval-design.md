# ragkit contextual retrieval — design spec

**Date:** 2026-08-25
**Status:** Approved (design), pending implementation plan
**Sub-project 2 of 3** (eval harness ✓ → **contextual retrieval** → hybrid). Proven against the sub-project-1 eval harness.

## Purpose

Standard RAG embeds each chunk in isolation, so a chunk saying "the balance rose 12%" loses whose/when. **Contextual retrieval** (Anthropic cookbook `capabilities/contextual-embeddings/guide.ipynb`) prepends an LLM-written 1-sentence situating blurb to each chunk *before embedding*, cutting retrieval-failure rate materially. ragkit is local-only by design, so the context generation runs on the operator's own Ollama — no external API, no sensitivity question.

## Decisions (approved)

- **Batch-per-doc generation:** one LLM call per document feeds the whole doc + all its chunks and returns a context line per chunk (Ollama has no cross-request prompt cache, so re-sending the doc per chunk is wasteful). Cap chunks-per-call and loop for long docs.
- **Opt-in:** a `contextual.enabled` flag, default **off**, mirroring `gate_enabled`. Adds ingest latency + GPU cost, so the operator turns it on deliberately and measures the lift first.
- **Embed augmented, store raw:** embed `context + "\n" + chunk.text`; the Qdrant payload `text` stays the **raw chunk** (context is a retrieval aid, not display content). No store-schema change.
- **Fail-loud (ragkit convention):** a context-generation failure for a doc marks it `state='error'` (visible in `ragkit status`), never silently embeds without context. Consistent with how the gate behaves.

## Rule-of-three refactor (resolves review finding #6)

There are now three callers of the same Ollama `/api/generate` + `format=json` + `think:false` + `num_predict` pattern: `gate.py`, the eval `_make_gen_fn`, and this. Extract a shared **`ragkit/ollama.py` `generate_json()`** helper and route all three through it. This DRYs the pattern AND carries the `think:false` + output-cap fix into `gate.py` (which currently lacks both and survives only because its output is a trivial 2-field object). Low-risk: `think:false`/`num_predict` only help the gate.

## Architecture

New: `ragkit/ollama.py` (shared helper), `ragkit/contextualize.py` (the Contextualizer). Modified: `gate.py` + `cli.py _make_gen_fn` (use the helper), `config.py` (ContextualConfig), `pipeline.py` (inject + apply), `cli.py cmd_ingest` (build it when enabled), `ragkit.example.yaml`.

### `ragkit/ollama.py`
```
def generate_json(client, url, model, *, system, prompt,
                  num_predict=1024, temperature=0.2) -> dict|list
```
POSTs `/api/generate` with `stream=False, format="json", think=False, options={temperature, num_predict}`; raises on empty response; returns parsed JSON. The single home for the `think:false` + cap contract.

### `ragkit/contextualize.py`
`Contextualizer(ollama_cfg, model, max_chunks_per_call=10)`, method `contextualize(doc_text: str, chunks: list[Chunk]) -> list[str]` returning one context string per chunk (aligned by index). Batches ≤`max_chunks_per_call` chunks per `generate_json` call (numbered chunk list in the prompt, expects a JSON array/`{"contexts": [...]}` aligned to the numbering); loops for more; pads/truncates defensively to exactly `len(chunks)` (missing → empty string, which degrades that chunk to raw-embed, not a crash). Model should be capable (≥~9B); default falls back to `ollama.gate_model` with a doc warning, same lesson as eval-gen.

### `pipeline.py`
Add optional `contextualizer` to `Pipeline.__init__` (like `gate`). In the embed+store loop, when present:
```
texts = [c.text for c in chunks]
ctxs = self.contextualizer.contextualize(d.text or "", chunks)   # raises -> _err(doc), fail-loud
texts = [f"{ctx}\n{c.text}" if ctx else c.text for ctx, c in zip(ctxs, chunks)]
vectors = self.embedder.embed_batch(texts)
```
`upsert(chunks, vectors, ...)` unchanged — stores raw `chunk.text`. Gated by `cfg.contextual.enabled` (built in cmd_ingest only when enabled, else `None`).

### `config.py`
```
@dataclass
class ContextualConfig:
    enabled: bool = False
    model: Optional[str] = None          # None -> ollama.gate_model (use a capable model)
    max_chunks_per_call: int = 10
```
Field `contextual: ContextualConfig` on `Config`, wired into `Config.load`; documented in `ragkit.example.yaml`.

## Measurement (the point of doing eval first)

Smoke: ingest the sample corpus twice into two collections — `plain` (contextual off) and `ctx` (on) — then `ragkit eval-gen` a shared golden set and `ragkit eval` both, comparing with `--baseline`. Contextual should be ≥ plain on recall@k/MRR. On the 4-doc sample the signal is weak (ceiling); the harness + method are what's being validated, and the same procedure applies to a real corpus.

## Non-goals (YAGNI)

- No Anthropic prompt-caching port (Ollama lacks it; batch-per-doc is the local equivalent).
- No re-embed migration tooling — `ragkit ingest` already re-embeds in place (idempotent upsert).
- No per-chunk context caching/persistence in v1.
- Hybrid retrieval is sub-project 3.

## Acceptance

1. `contextual.enabled: false` (default) → ingest behavior byte-for-byte unchanged.
2. `enabled: true` → each chunk embedded with a generated context prefix; payload `text` still raw chunk.
3. `gate.py`, `_make_gen_fn`, `contextualize.py` all call `ollama.generate_json`; gate still works with `--gate` (smoke).
4. Context-gen failure marks the doc `error` (fail-loud), never silent.
5. `pytest` green, network-free (helper + contextualizer logic mocked/injected).
6. Live smoke: plain vs ctx eval comparison runs and prints deltas.

## Build routing
Design/review = Opus. Implementation = Sonnet (mechanical, fully specced). Not OmniRoute (proprietary source).
