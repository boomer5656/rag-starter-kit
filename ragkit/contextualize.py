"""Contextual retrieval: prepend an LLM-written situating sentence to each chunk
before embedding (Anthropic's contextual-embeddings technique), on the local
Ollama. Batch-per-doc: one call per <=max_chunks_per_call chunks — Ollama has no
cross-request prompt cache, so re-sending the doc per chunk would be wasteful.

Fail-loud: a generation failure propagates to the pipeline, which marks the doc
state='error' (visible in `ragkit status`) rather than silently embedding raw.
"""
from __future__ import annotations

import httpx

from .config import OllamaConfig
from .models import Chunk
from .ollama import generate_json

_DOC_CHARS = 6000
_SYSTEM = (
    "You situate document chunks for a retrieval system. For each numbered chunk, "
    "write ONE short sentence (<=25 words) giving the context needed to understand it "
    "in isolation — what document/section/entity/period it concerns. Output ONLY JSON: "
    "{\"contexts\": [\"...\"]} with exactly one entry per chunk, in the given order."
)


class Contextualizer:
    def __init__(self, ollama_cfg: OllamaConfig, model: str,
                 max_chunks_per_call: int = 10, timeout: float = 120.0):
        self.cfg = ollama_cfg
        self.model = model
        self.max_per_call = max(1, max_chunks_per_call)
        self._client = httpx.Client(timeout=timeout)

    def contextualize(self, doc_text: str, chunks: list[Chunk]) -> list[str]:
        """One context string per chunk, aligned by index. Empty string = no context
        for that chunk (it embeds raw). Raises if the model call fails."""
        doc = (doc_text or "")[:_DOC_CHARS]
        out: list[str] = []
        for start in range(0, len(chunks), self.max_per_call):
            out.extend(self._one_call(doc, chunks[start:start + self.max_per_call]))
        if len(out) < len(chunks):
            out.extend([""] * (len(chunks) - len(out)))
        return out[:len(chunks)]

    def _one_call(self, doc: str, batch: list[Chunk]) -> list[str]:
        numbered = "\n\n".join(f"[chunk {i}]\n{c.text}" for i, c in enumerate(batch))
        prompt = (f"Document:\n{doc}\n\nChunks to situate:\n{numbered}\n\n"
                  f"Return JSON: {{\"contexts\": [...]}} with exactly {len(batch)} entries, in order.")
        data = generate_json(self._client, self.cfg.url, self.model, system=_SYSTEM, prompt=prompt)
        ctxs = data.get("contexts") if isinstance(data, dict) else data
        if not isinstance(ctxs, list):
            ctxs = []
        ctxs = [str(c) for c in ctxs][:len(batch)]
        if len(ctxs) < len(batch):
            ctxs.extend([""] * (len(batch) - len(ctxs)))
        return ctxs

    def close(self) -> None:
        self._client.close()
