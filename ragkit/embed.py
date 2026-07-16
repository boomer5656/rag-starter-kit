"""Embeddings via Ollama (bge-m3 by default).

bge-m3 is symmetric (no query/document prefix), which is why it's the default:
it removes the `search_query:`/`search_document:` prefix footgun that asymmetric
models like nomic-embed-text impose. The dimension is *probed* once and asserted
against the collection before any write, so a model/collection mismatch fails
loudly instead of silently corrupting the store.
"""
from __future__ import annotations

import httpx

from .config import OllamaConfig


class Embedder:
    def __init__(self, cfg: OllamaConfig, timeout: float = 120.0):
        self.cfg = cfg
        self._client = httpx.Client(timeout=timeout)
        self._dim: int | None = None

    def embed(self, text: str) -> list[float]:
        r = self._client.post(
            f"{self.cfg.url}/api/embeddings",
            json={"model": self.cfg.embed_model, "prompt": text},
        )
        r.raise_for_status()
        vec = r.json().get("embedding", [])
        if not vec:
            raise RuntimeError(
                f"empty embedding from {self.cfg.embed_model} "
                f"(is it pulled on the Ollama at {self.cfg.url}?)"
            )
        return vec

    def embed_batch(self, texts: list[str]) -> list[list[float]]:
        # Ollama's /api/embeddings is one-at-a-time; keep the model resident by
        # looping here rather than reconnecting per call. (Stage batching upstream
        # ensures this loop runs with bge-m3 already loaded.)
        return [self.embed(t) for t in texts]

    @property
    def dim(self) -> int:
        if self._dim is None:
            self._dim = len(self.embed("dimension probe"))
        return self._dim

    def close(self) -> None:
        self._client.close()
