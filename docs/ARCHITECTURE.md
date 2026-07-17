# ragkit — Architecture & Design Spec

> Status: v0 (scaffold). This is the design contract the code is built against.
> Author-facing spec; the buyer-facing pitch lives in the top-level `README.md`.

## What this is

A self-hosted RAG **ingestion pipeline** you point at your own documents. RAG-over-your-docs
is a commodity in 2026 (Dify, Onyx/Danswer, Open WebUI, AnythingLLM all ship it free). The
part they all treat as a black box — **getting your messy documents into the vector store
well, repeatably, on hardware you control** — is what ragkit is about.

The differentiator is the pipeline, not the retrieval. Specifically:

1. **Observable state machine.** Every document moves through explicit states
   (`discovered → extracted → gated → chunked → embedded → stored`, or `error`/`skipped`).
   Nothing is ever silently dropped. You can always answer "what happened to this file?"
2. **Resource-aware, stage-batched execution.** On a single-GPU homelab the embedding model
   and any gate/summary model thrash if you load them per-document. ragkit batches by stage so
   each model stays resident. (Ported from the KIP pipeline's hard-won lesson: per-doc model
   swaps cost ~40s/doc; stage batching removes it.)
3. **Optional relevance gate.** A cheap model (e.g. `llama3.2:3b`) filters documents before the
   expensive embed step, so you don't waste GPU embedding junk. Off by default; opt-in per source.
4. **Idempotent by construction.** `point_id = stable_hash(source_uri)`. Re-running an ingest
   updates in place (Qdrant `upsert`, SQL `ON CONFLICT DO UPDATE`) — safe to re-run anytime.
5. **Correctness assertions.** Embedding dimension is asserted against the collection's dim
   before upsert; a mismatch is a loud error, not a corrupt collection. (The classic
   768-vs-1024 cross-wire bug is impossible.)
6. **Pluggable connectors.** A `Connector` yields `SourceDoc`s; the pipeline does the rest.
   Ship: `files` (a folder of PDFs/docx/md — the headline), plus two worked examples.

## Non-goals (YAGNI)

- Not a chat UI. Bring your own (Open WebUI, the MCP server, curl). ragkit fills the store.
- Not a framework. No plugin marketplace, no DSL. Python + a YAML config.
- Not multi-tenant SaaS. Single operator, self-hosted. (A hosted tier is a *later* question.)

## Component map

```
ragkit/
  config.py       # load + validate ragkit.yaml; env overrides
  models.py       # SourceDoc, Chunk, IngestState, PipelineResult (dataclasses)
  state.py        # the state machine; SQLite-backed by default, Postgres optional
  extract.py      # bytes/path -> text, via Tika (confined) or built-in for txt/md
  chunk.py        # text -> [Chunk] (token-aware, overlap, metadata carry-through)
  gate.py         # optional cheap-model relevance filter (Ollama)
  embed.py        # text -> vector, bge-m3 via Ollama; dim probe + assertion
  store.py        # Qdrant collection lifecycle, payload indexes, idempotent upsert
  rerank.py       # query + hits -> reranked hits, via the reranker sidecar
  pipeline.py     # the orchestrator: ties the above together, stage-batched
  search.py       # query path: embed -> Qdrant search -> (optional) rerank
  cli.py          # `ragkit ingest ./docs`, `ragkit search "q"`, `ragkit status`, `ragkit serve-mcp`
  connectors/
    base.py       # Connector protocol -> yields SourceDoc
    files.py      # headline: walk a directory, one SourceDoc per file
    literature.py # worked example: OpenAlex papers (adapted from KIP)
    sigma.py      # worked example: Sigma detection rules (adapted from homelab-ai)
mcp/server.py     # FastMCP: rag_search / list_collections / ingest_path (adapted, hardened)
reranker/app.py   # bge-reranker-v2-m3 FastAPI sidecar (adapted from realestate-dashboard)
```

## Data model

- **SourceDoc**: `{ uri, title, mime, text?, raw_bytes?, meta: dict }` — what a connector yields.
- **Chunk**: `{ id, source_uri, ordinal, text, meta }` — `id = hash(source_uri + ordinal)`.
- **IngestState** (per source_uri): `{ uri, state, dim?, n_chunks?, error?, updated_at }`.
- **Qdrant payload**: `{ text, source_uri, title, ordinal, source_type, **connector_meta }`.
  Connector-declared fields get keyword/integer payload indexes for filtering.

## The pipeline (stage-batched)

```
discover → [extract] → [gate?] → [chunk] → [embed] → [store]
```

Each bracketed stage runs over the whole batch before the next begins, so a resident model
(embed, or the gate model) is loaded once per stage, not once per doc. State is written after
every stage transition, so a crash resumes from the last completed stage. `error` at any stage
is terminal-for-that-doc and *visible* (counted, listed by `ragkit status`), never a silent drop.

## Backends & config

- **Vector store:** Qdrant (HTTP). Collection name, dim, distance from config.
- **Embeddings:** Ollama `bge-m3` (1024-dim, cosine) by default. Model + Ollama URL from config.
- **State store:** SQLite file by default (zero setup). Postgres if `state.backend: postgres`.
- **Extraction:** Tika sidecar for binary docs; built-in for `.txt`/`.md`.
- **Reranker:** optional sidecar; search skips it if not configured.

Everything is driven by one `ragkit.yaml` with env-var overrides. The tower defaults
(`192.168.1.100`, the homelab ports) are the *example* config, not hardcoded.

## Security posture (carried over, non-negotiable)

- Tika/file reads are **confined to an allowed root**; path traversal + non-doc extensions rejected.
  (Ported from `mcp_homelab.py`'s `tika_extract` confinement.)
- Retrieved content is **untrusted data**: the MCP `rag_search` wraps payloads in an
  untrusted-content envelope so a downstream model treats them as data, not instructions.
- No secrets in the image or in git. `ragkit.yaml` example ships placeholders only;
  real config via `.env` / env vars, gitignored.
- SQL uses bound parameters, never string interpolation of document-derived values.

## Definition of done (v0)

- `docker compose up -d` brings up Qdrant + Tika (+ optional reranker); Ollama is external
  (reuse an existing one) or included via a profile.
- `ragkit ingest ./examples/docs` ingests a sample folder end-to-end and reports per-stage counts.
- `ragkit search "..."` returns ranked hits; `ragkit status` shows the state table.
- `ragkit serve-mcp` exposes retrieval to Claude Code / any MCP client.
- Re-running `ingest` changes nothing (idempotent); a dim mismatch errors loudly.
- README quickstart works from a clean clone on a machine with Docker + an Ollama with bge-m3.
