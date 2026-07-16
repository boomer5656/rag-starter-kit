"""Optional cross-encoder reranking sidecar client.

The sidecar is a small, separately-run service (e.g. bge-reranker-v2-m3 behind
FastAPI/uvicorn) that scores (query, document) pairs more precisely than the bi-encoder
vector search alone. It is entirely optional: when reranker_cfg.url is unset, `.rerank()`
is a no-op so `ragkit search` works with nothing but Qdrant + Ollama running.

Expected sidecar contract (adjust to match your actual sidecar):
  POST {url}/rerank  { "query": str, "documents": [str, ...] }
  ->   200 { "scores": [float, ...] }              # one score per input document, same order
  or   200 { "results": [{"index": int, "score": float}, ...] }  # cross-encoder-style reorder

Either shape is accepted; whichever fields are present are used, and hits are always
returned as the original Qdrant hit dicts (with a `rerank_score` key added) sorted by
descending score.
"""
from __future__ import annotations

import httpx

from .config import RerankerConfig


class Reranker:
    def __init__(self, reranker_cfg: RerankerConfig, timeout: float = 60.0):
        self.cfg = reranker_cfg
        self._client = httpx.Client(timeout=timeout) if reranker_cfg.url else None

    def rerank(self, query: str, hits: list[dict], top_k: int | None = None) -> list[dict]:
        if not self.cfg.url or not hits:
            return hits[:top_k] if top_k is not None else hits

        documents = [self._text(h) for h in hits]
        r = self._client.post(f"{self.cfg.url}/rerank", json={"query": query, "documents": documents})
        r.raise_for_status()
        body = r.json()

        scores = self._extract_scores(body, len(hits))
        for hit, score in zip(hits, scores):
            hit["rerank_score"] = score

        ranked = sorted(hits, key=lambda h: h.get("rerank_score", float("-inf")), reverse=True)
        return ranked[:top_k] if top_k is not None else ranked

    @staticmethod
    def _text(hit: dict) -> str:
        return (hit.get("payload") or {}).get("text", "")

    @staticmethod
    def _extract_scores(body: dict, n: int) -> list[float]:
        # Shape A: {"scores": [float, ...]} aligned to input order.
        if isinstance(body.get("scores"), list) and len(body["scores"]) == n:
            return [float(s) for s in body["scores"]]
        # Shape B: {"results": [{"index": int, "score": float}, ...]}, possibly reordered.
        if isinstance(body.get("results"), list):
            scores = [0.0] * n
            for item in body["results"]:
                idx = item.get("index")
                if idx is not None and 0 <= idx < n:
                    scores[idx] = float(item.get("score", 0.0))
            return scores
        raise RuntimeError(f"reranker: unrecognized response shape: {list(body.keys())}")

    def close(self) -> None:
        if self._client is not None:
            self._client.close()
