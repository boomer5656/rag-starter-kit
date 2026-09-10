# AGENTS.md

Self-hosted RAG ingestion pipeline: extract -> gate -> chunk -> embed -> store, with an
observable per-document state machine. Targets a homelab Ollama + Qdrant, not a cloud API.

## Stack
- Python >=3.10, stdlib + `httpx`, `pyyaml`, `mcp`, `fastmcp` (see `pyproject.toml`).
- No ORM/pydantic — plain dataclasses (`ragkit/models.py`).
- Optional reranker sidecar: FastAPI + uvicorn (`reranker/`), separate Docker image.
- Vector store: Qdrant. Embedder + gate LLM: Ollama (external, bring-your-own). Extraction: Apache Tika.

## Setup
```
pip install -e .                 # core package + CLI entry point `ragkit`
pip install -e ".[dev]"          # + pytest
pip install -e ".[reranker]"     # only needed if hacking on reranker/app.py directly (it has its own Dockerfile)
cp ragkit.example.yaml ragkit.yaml   # edit URLs/models; ragkit.yaml is gitignored
```
Ollama is NOT started by this repo's compose file by default; requires `bge-m3` pulled
(`ollama pull bge-m3`) on whatever Ollama instance `ragkit.yaml` points at.

## Run / Dev
```
docker compose up -d                         # qdrant + tika only
docker compose --profile rerank up -d        # + reranker sidecar
docker compose --profile ollama up -d        # + a local Ollama (GPU reservation in compose)

ragkit ingest ./examples/docs                # end-to-end ingest (connector=files by default)
ragkit ingest /path --connector sigma|literature|files --collection NAME [--gate]
ragkit status [--collection NAME]            # per-state counts + recent errors
ragkit search "query" [--collection NAME] [--rerank] [--top-k N]
ragkit serve-mcp                             # runs mcp/server.py over stdio for Claude Code / MCP clients
```
`ragkit.yaml` (copied from `ragkit.example.yaml`) is the single source of config: Ollama URL +
embed/gate model, Qdrant URL, Tika URL + `allowed_root` (confines filesystem reads — never point
at a root), reranker URL (null = reranking disabled), state backend (sqlite default / postgres),
chunk sizes, `gate_enabled`. Every field is also overridable via `RAGKIT_*` env vars
(`ragkit/config.py:_apply_env`) — see `.env.example` for the docker-compose-facing ones.

## Test / Lint / Format
```
pytest                # tests/ — pure unit tests, no network (Ollama/Qdrant/Tika) allowed, see tests/conftest.py
```
No lint/format tool is configured in this repo (no ruff/black/flake8 config in `pyproject.toml`) —
match surrounding style by hand; don't add tooling config unasked.

## Architecture
- Pipeline (`ragkit/pipeline.py`): stage-batched — every doc clears a stage before any advances,
  so a resident model (gate, then bge-m3) loads once per batch, not once per doc. State is
  persisted after each transition (`ragkit/state.py`, `ragkit/models.py:State` enum) so a crash
  resumes; per-doc errors are recorded and isolated, never abort the whole run.
- Data flow: connector yields `SourceDoc` -> `TikaExtractor.extract` (`ragkit/extract.py`, skipped
  for .txt/.md) -> optional `RelevanceGate.keep` (`ragkit/gate.py`, cheap Ollama model, opt-in) ->
  `TokenChunker.chunk` (`ragkit/chunk.py`) -> `Embedder.embed_batch` (`ragkit/embed.py`, Ollama
  `/api/embeddings`, default `bge-m3` 1024-dim symmetric) -> `Store.upsert` (`ragkit/store.py`, Qdrant).
- Idempotency: point id is a stable hash of `(source_uri, ordinal)` (`ragkit/models.py:stable_id`),
  so re-ingest upserts in place — never duplicates. `Store.ensure_collection` asserts embedder dim
  against the existing collection before any write (prevents silent dim cross-wire, e.g. 768 vs 1024).
- CLI entry point: `ragkit/cli.py` (`main()` -> `ragkit.cli:main` in `pyproject.toml`) — thin wiring
  only; pipeline/search logic lives in `pipeline.py` / `search.py`, not here.
- Connectors (`ragkit/connectors/`): `files.py` is the headline (folder of PDFs/docx/md/txt),
  `literature.py` and `sigma.py` are worked custom-connector examples. Contract: `base.py`.
- MCP server (`mcp/server.py`): exposes `rag_search`/list-collections (+ ingest control) over
  stdio for Claude Code; reads the same `ragkit.yaml`/`RAGKIT_CONFIG`, never hardcodes an endpoint.
- Reranker (`reranker/app.py`): optional `bge-reranker-v2-m3` FastAPI sidecar, only used when
  `ragkit.yaml`'s `reranker.url` is set and `ragkit search --rerank` is passed.

## Conventions & gotchas
- `.gitattributes` forces LF for `*.py/.yml/.yaml/.toml/.json/.md/.sh` — don't let Windows tooling
  reintroduce CRLF.
- Never read/echo `ragkit.yaml`, `.env`, or `.ragkit/state.db` contents containing real
  endpoints/secrets into outputs; both are gitignored. Only `ragkit.example.yaml`/`.env.example`
  are committed.
- `ragkit.example.yaml`'s "homelab tower example" block (bottom, commented out) shows a real LAN
  deployment shape (192.168.1.x) — illustrative only, not a hardcoded default anywhere in code.
- Tika's `allowed_root` (and the ingest path itself) bounds filesystem access — respect it when
  extending connectors; don't widen it to a filesystem root.
- Ollama embeddings must stay dimension-consistent per collection; if you change `embed_model`,
  either use a fresh collection or expect `DimensionMismatch` (`ragkit/store.py`) by design.
- Tests must stay network-free (`tests/conftest.py` docstring) — mock/stub Ollama/Qdrant/Tika
  rather than hitting real services.
