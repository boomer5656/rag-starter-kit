# ragkit

**The self-hosted RAG ingestion pipeline that isn't a black box.**

Chatting with your documents is a solved, commoditized problem — Dify, Onyx, Open WebUI and a
dozen others do it for free. The part they hide from you is the part that actually decides whether
your answers are any good: **how your messy real-world documents get into the vector store.**

ragkit is that part, done right, on hardware you own:

- 🔍 **Observable** — every document moves through explicit states (`extracted → gated → chunked → embedded → stored`). Nothing is ever silently dropped. `ragkit status` tells you exactly what happened to every file.
- ⚡ **Resource-aware** — stage-batched execution keeps each model resident instead of thrashing model loads on a single-GPU box. Built for a homelab, not a datacenter.
- 🚦 **A relevance gate** — filter junk with a cheap model *before* you spend GPU embedding it (opt-in).
- ♻️ **Idempotent** — re-run any ingest safely; it updates in place. No duplicates, ever.
- ✅ **Correct by assertion** — embedding dimension is checked against the collection before write. The infamous 768-vs-1024 cross-wire bug can't happen.
- 🔌 **Pluggable** — point it at a folder of PDFs (the headline), or write a 20-line connector for anything else.

Your data never leaves your network. Bring your own Ollama; ragkit brings the pipeline.

---

## Quick start

Prereqs: Docker, and an [Ollama](https://ollama.com) with `bge-m3` pulled (`ollama pull bge-m3`).

```bash
git clone <this-repo> && cd rag-starter-kit
cp ragkit.example.yaml ragkit.yaml     # edit URLs if your Ollama/Qdrant aren't the defaults
docker compose up -d                   # Qdrant + Tika (+ optional reranker)
pip install -e .

ragkit ingest ./examples/docs          # ingest the sample folder, end to end
ragkit status                          # see the state table: what got in, what errored
ragkit search "how does the pipeline avoid duplicates?"
```

Point it at your own documents:

```bash
ragkit ingest /path/to/your/pdfs --collection mydocs
ragkit search "..." --collection mydocs --rerank
```

Wire it into Claude Code (or any MCP client) for retrieval:

```bash
ragkit serve-mcp        # exposes rag_search / list_collections over stdio
```

---

## What's in the box

| Piece | What it does |
|---|---|
| `ragkit` (Python pkg) | the pipeline: extract → gate → chunk → embed → store, with a visible state machine |
| `connectors/files` | point at a directory of PDFs/docx/md/txt — the headline connector |
| `connectors/literature`, `connectors/sigma` | two worked examples of custom connectors |
| `mcp/server.py` | a hardened MCP server so Claude Code can retrieve from your store |
| `reranker/` | optional `bge-reranker-v2-m3` sidecar for better ranking |
| `docker-compose.yml` | Qdrant + Tika (+ reranker); Ollama is external or an opt-in profile |

See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) for the full design.

## Why self-hosted

Because for a lot of documents — leases, medical records, client files, anything under NDA or
regulation — "just use the OpenAI API" isn't an option. ragkit runs entirely on your LAN: your
Ollama, your Qdrant, your disk. Nothing is sent anywhere.

## License

MIT — see [LICENSE](LICENSE).
