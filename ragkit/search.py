"""Query path: embed the query, search Qdrant, optionally rerank.

Deliberately thin — Searcher owns just the three clients it needs (Embedder,
Store, Reranker) and turns raw Qdrant hits into plain dicts so callers (the
CLI, the MCP server) never need to know Qdrant's response shape. Reranking is
a no-op whenever no reranker URL is configured (see rerank.py), so `rerank=True`
is always safe to pass even with the sidecar absent.
"""
from __future__ import annotations

import httpx

from .config import Config
from .embed import Embedder
from .expand import parse_expansions, rrf_fuse
from .ollama import generate_json
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
        self._expand_client = httpx.Client(timeout=120.0)

    _EXPAND_SYSTEM = (
        "You rewrite a search query into alternative phrasings for a retrieval system — "
        "vary vocabulary, specificity, and synonyms while preserving intent. Output ONLY JSON: "
        "{\"queries\": [\"...\"]}."
    )

    def _retrieve_hits(self, store: Store, query: str, fetch_k: int, hybrid: bool,
                       coll: str, query_filter: dict | None) -> list[dict]:
        """One query's raw hits (dense, or hybrid when stats exist) — no rerank."""
        vector = self.embedder.embed(query)
        stats = self._stats if coll == self.cfg.collection else CorpusStats.load(stats_path(coll))
        if hybrid and stats.n_docs > 0:
            sparse = to_qdrant(query_sparse(query, stats))
            return store.query_hybrid(vector, sparse, top_k=fetch_k, query_filter=query_filter)
        return store.search(vector, top_k=fetch_k, query_filter=query_filter)

    def _expand(self, query: str, n: int) -> list[str]:
        """n LLM paraphrases (best-effort → [] on any failure)."""
        model = self.cfg.multi_query.model or self.cfg.ollama.gate_model
        prompt = f"Query: {query}\n\nReturn JSON: {{\"queries\": [...]}} with {n} distinct rephrasings."
        try:
            raw = generate_json(self._expand_client, self.cfg.ollama.url, model,
                                system=self._EXPAND_SYSTEM, prompt=prompt)
            return parse_expansions(raw, query, n)
        except Exception:
            return []

    def search(self, query: str, *, top_k: int = 5, rerank: bool = False, hybrid: bool = False,
               multi: int = 0, query_filter: dict | None = None,
               collection: str | None = None) -> list[dict]:
        store = self.store
        opened = False
        coll = collection if collection is not None else self.cfg.collection
        if collection is not None and collection != self.store.collection:
            store = Store(self.cfg.qdrant, collection)
            opened = True
        try:
            want_rerank = rerank and bool(self.cfg.reranker.url)
            fetch_k = top_k * _RERANK_FANOUT if want_rerank else top_k
            if multi > 0:
                queries = [query] + self._expand(query, multi)
                hits = rrf_fuse([self._retrieve_hits(store, q, fetch_k, hybrid, coll, query_filter)
                                 for q in queries])
            else:
                hits = self._retrieve_hits(store, query, fetch_k, hybrid, coll, query_filter)
            if want_rerank:
                hits = self.reranker.rerank(query, hits, top_k=top_k)   # rerank vs ORIGINAL query
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
        self._expand_client.close()
