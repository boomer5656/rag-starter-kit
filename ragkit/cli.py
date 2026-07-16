"""`ragkit` command-line entry point: ingest, search, status, serve-mcp.

Thin wiring layer — every subcommand builds the same handful of components
(TikaExtractor, TokenChunker, Embedder, Store, StateStore, optionally
RelevanceGate) straight from Config and hands them to Pipeline or Searcher.
No business logic lives here; a bug in ingest behavior belongs in pipeline.py,
not this file.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys

from .config import Config
from .embed import Embedder
from .extract import TikaExtractor
from .gate import RelevanceGate
from .models import PipelineResult
from .pipeline import Pipeline
from .search import Searcher
from .state import StateStore
from .store import Store

_DEFAULT_GATE_CRITERIA = "documents relevant to the user's corpus"


def _build_connector(name: str, path: str):
    """Import + construct the chosen connector. Each connector takes the
    ingest path/source as its only required argument (see connectors/base.py)."""
    if name == "files":
        from .connectors.files import FilesConnector
        return FilesConnector(path)
    if name == "sigma":
        from .connectors.sigma import SigmaConnector
        return SigmaConnector(path)
    if name == "literature":
        from .connectors.literature import LiteratureConnector
        return LiteratureConnector(path)
    raise ValueError(f"unknown connector: {name!r}")


def _gate_criteria(cfg: Config) -> str:
    # Config has no dedicated field for this yet; read it if a project has
    # stashed one in ragkit.yaml anyway, otherwise fall back to a generic gate.
    return getattr(cfg, "gate_criteria", None) or _DEFAULT_GATE_CRITERIA


def cmd_ingest(args: argparse.Namespace) -> int:
    cfg = Config.load(args.config)
    collection = args.collection or cfg.collection

    connector = _build_connector(args.connector, args.path)
    source_type = getattr(connector, "source_type", args.connector)
    index_fields = getattr(connector, "index_fields", {})

    allowed_root = args.path if os.path.isdir(args.path) else (os.path.dirname(args.path) or ".")
    extractor = TikaExtractor(cfg.tika, allowed_root)
    from .chunk import TokenChunker  # imported lazily so `--help` works before chunk.py lands
    chunker = TokenChunker(cfg.chunk)
    embedder = Embedder(cfg.ollama)
    store = Store(cfg.qdrant, collection)
    state = StateStore(cfg.state, collection)
    gate = RelevanceGate(cfg.ollama, _gate_criteria(cfg)) if args.gate else None

    pipeline = Pipeline(
        cfg, extractor=extractor, chunker=chunker, embedder=embedder,
        store=store, state=state, gate=gate, index_fields=index_fields,
    )
    try:
        result: PipelineResult = pipeline.run(
            connector.iter_docs(), source_type=source_type,
            gate_enabled=True if args.gate else None,
        )
        print(result.summary())
        return 1 if result.errors else 0
    finally:
        extractor.close()
        embedder.close()
        store.close()
        state.close()
        if gate is not None:
            gate.close()


def cmd_search(args: argparse.Namespace) -> int:
    cfg = Config.load(args.config)
    if args.collection:
        cfg.collection = args.collection
    searcher = Searcher(cfg)
    try:
        hits = searcher.search(args.query, top_k=args.top_k, rerank=args.rerank)
        if not hits:
            print("no results")
            return 0
        for i, h in enumerate(hits, start=1):
            score = h["score"]
            score_str = f"{score:.4f}" if isinstance(score, (int, float)) else str(score)
            heading = h["title"] or h["source_uri"]
            print(f"{i}. [{score_str}] {heading}")
            print(f"   {h['source_uri']}")
            print(f"   {h['text']}")
            print()
        return 0
    finally:
        searcher.close()


def cmd_status(args: argparse.Namespace) -> int:
    cfg = Config.load(args.config)
    collection = args.collection or cfg.collection
    state = StateStore(cfg.state, collection)
    try:
        counts = state.counts()
        print(f"collection: {collection}")
        if not counts:
            print("  (no ingest history)")
        else:
            width = max(len(s) for s in counts)
            for s, n in sorted(counts.items()):
                print(f"  {s.ljust(width)}  {n}")

        errors = state.errors(limit=20)
        if errors:
            print(f"\nrecent errors ({len(errors)}):")
            for e in errors:
                print(f"  {e.uri}")
                print(f"    {e.error}")
        return 0
    finally:
        state.close()


def cmd_serve_mcp(args: argparse.Namespace) -> int:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    server_path = os.path.join(root, "mcp", "server.py")
    if not os.path.exists(server_path):
        print(f"MCP server not found at {server_path}", file=sys.stderr)
        return 1
    return subprocess.call([sys.executable, server_path])


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ragkit", description="Self-hosted RAG ingestion pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    p_ingest = sub.add_parser("ingest", help="ingest documents into the vector store")
    p_ingest.add_argument("path", help="path (or connector-specific source) to ingest")
    p_ingest.add_argument("--collection", default=None)
    p_ingest.add_argument("--connector", choices=["files", "sigma", "literature"], default="files")
    p_ingest.add_argument("--gate", action="store_true", help="run the relevance gate for this ingest")
    p_ingest.add_argument("--config", default="ragkit.yaml")
    p_ingest.set_defaults(func=cmd_ingest)

    p_search = sub.add_parser("search", help="search the vector store")
    p_search.add_argument("query")
    p_search.add_argument("--collection", default=None)
    p_search.add_argument("--top-k", type=int, default=5)
    p_search.add_argument("--rerank", action="store_true")
    p_search.add_argument("--config", default="ragkit.yaml")
    p_search.set_defaults(func=cmd_search)

    p_status = sub.add_parser("status", help="show ingest state counts and recent errors")
    p_status.add_argument("--collection", default=None)
    p_status.add_argument("--config", default="ragkit.yaml")
    p_status.set_defaults(func=cmd_status)

    p_serve = sub.add_parser("serve-mcp", help="run the MCP retrieval server")
    p_serve.add_argument("--config", default="ragkit.yaml")
    p_serve.set_defaults(func=cmd_serve_mcp)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
