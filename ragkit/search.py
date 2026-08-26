"""Query path: embed the query, search Qdrant, optionally rerank.

Deliberately thin — Searcher owns just the three clients it needs (Embedder,
Store, Reranker) and turns raw Qdrant hits into plain dicts so callers (the
CLI, the MCP server) never need to know Qdrant's response shape. Reranking is
a no-op whenever no reranker URL is configured (see rerank.py), so `rerank=True`
is always safe to pass even with the sidecar absent.
"""
from __future__ import annotations

from .config import Config
from .embed import Embedder
from .rerank import Reranker
from .sparse import CorpusStats, query_sparse, stats_path, to_qdrant
from .store import Store

# How much extra depth to pull from Qdrant before handing hits to the
# reranker — reranking only helps if it has more than top_k candidates to sort.
_RERANK_FANOUT = 4
_SNIPPET_CHARS = 500


class Searcher:
    """Owns the embed/search/rerank clients for one Config; safe to reuse across queries."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.embedder = Embedder(cfg.ollama)
        self.store = Store(cfg.qdrant, cfg.collection)
        self.reranker = Reranker(cfg.reranker)
        self._stats = CorpusStats.load(stats_path(cfg.collection))

    def search(self, query: str, *, top_k: int = 5, rerank: bool = False, hybrid: bool = False,
               query_filter: dict | None = None, collection: str | None = None) -> list[dict]:
        store = self.store
        opened = False
        coll = collection if collection is not None else self.cfg.collection
        if collection is not None and collection != self.store.collection:
            store = Store(self.cfg.qdrant, collection)
            opened = True
        try:
            vector = self.embedder.embed(query)
            want_rerank = rerank and bool(self.cfg.reranker.url)
            fetch_k = top_k * _RERANK_FANOUT if want_rerank else top_k
            stats = self._stats if coll == self.cfg.collection else CorpusStats.load(stats_path(coll))
            if hybrid and stats.n_docs > 0:
                sparse = to_qdrant(query_sparse(query, stats))
                hits = store.query_hybrid(vector, sparse, top_k=fetch_k, query_filter=query_filter)
            else:
                hits = store.search(vector, top_k=fetch_k, query_filter=query_filter)
            if want_rerank:
                hits = self.reranker.rerank(query, hits, top_k=top_k)
            else:
                hits = hits[:top_k]
            return [self._to_result(h) for h in hits]
        finally:
            if opened:
                store.close()

    @staticmethod
    def _to_result(hit: dict) -> dict:
        payload = hit.get("payload") or {}
        text = payload.get("text", "")
        snippet = text if len(text) <= _SNIPPET_CHARS else text[:_SNIPPET_CHARS] + "..."
        return {
            "score": hit.get("rerank_score", hit.get("score")),
            "source_uri": payload.get("source_uri", ""),
            "title": payload.get("title", ""),
            "text": snippet,
        }

    def close(self) -> None:
        self.embedder.close()
        self.store.close()
        self.reranker.close()
