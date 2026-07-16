"""Cross-encoder reranker sidecar for ragkit — pairs with the bge-m3 embedder.

Wraps BAAI/bge-reranker-v2-m3 (sentence-transformers CrossEncoder) behind a tiny
FastAPI service so ragkit/rerank.py can call a plain HTTP endpoint instead of
loading the model in-process. CPU-friendly by default: reranking a few dozen
candidates on CPU is a few hundred ms, which is fine for a search-time reorder
step and keeps the sidecar off the GPU used by embedding/generation.

The model is loaded once at import time (process startup), not per-request.
"""
from __future__ import annotations

import os

from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import CrossEncoder

MODEL_NAME = os.environ.get("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3")
DEVICE = os.environ.get("RERANKER_DEVICE", "cpu")
MAX_LENGTH = int(os.environ.get("RERANKER_MAX_LEN", "512"))

print(f"[reranker] loading {MODEL_NAME} on {DEVICE}", flush=True)
_model = CrossEncoder(MODEL_NAME, device=DEVICE, max_length=MAX_LENGTH)
print("[reranker] ready", flush=True)

app = FastAPI(title="ragkit-reranker")


class RerankRequest(BaseModel):
    query: str
    documents: list[str]
    top_k: int | None = None


class RerankResult(BaseModel):
    index: int
    score: float


class RerankResponse(BaseModel):
    results: list[RerankResult]
    model: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model": MODEL_NAME, "device": DEVICE}


@app.post("/rerank", response_model=RerankResponse)
def rerank(req: RerankRequest) -> RerankResponse:
    """Score each document against the query and return indices sorted best-first.

    Response shape is {results: [{index, score}], model}, sorted descending by
    score — this is the exact shape ragkit/rerank.py expects to consume.
    """
    if not req.documents:
        return RerankResponse(results=[], model=MODEL_NAME)

    pairs = [[req.query, doc] for doc in req.documents]
    scores = _model.predict(pairs, batch_size=32)

    ranked = sorted(enumerate(scores), key=lambda kv: kv[1], reverse=True)
    if req.top_k is not None:
        ranked = ranked[: req.top_k]

    results = [RerankResult(index=int(i), score=float(s)) for i, s in ranked]
    return RerankResponse(results=results, model=MODEL_NAME)
