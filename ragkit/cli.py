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

import httpx

from .config import Config
from .embed import Embedder
from .eval import (
    EvalReport, compare, evaluate, generate_golden, load_golden, load_report,
    save_golden, save_report,
)
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
            # think=False is load-bearing: Qwen3 models default to a thinking pass that,
            # under format="json", consumes the whole budget and returns "" or "{}".
            # num_predict caps output per the usage-limits doctrine (no uncapped generations).
            "stream": False, "format": "json", "think": False,
            "options": {"temperature": 0.2, "num_predict": 1024},
        })
        r.raise_for_status()
        raw = r.json().get("response")
        if not raw:
            raise RuntimeError(f"gen: empty response from {model}")
        import json as _json
        data = _json.loads(raw)
        qs = data.get("questions") if isinstance(data, dict) else data
        if not isinstance(qs, list):      # model returned a non-list shape -> no questions
            return []
        return [str(q) for q in qs][:n]

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
    if args.k:
        try:
            k_values = [int(x) for x in args.k.split(",") if x.strip()]
        except ValueError:
            print("invalid --k: expected comma-separated integers, e.g. 1,3,5,10")
            return 1
    else:
        k_values = cfg.eval.k_values
    if not k_values:
        print("no k values to evaluate (check --k or eval.k_values)")
        return 1
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
