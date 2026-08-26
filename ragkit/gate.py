"""Relevance gate: an LLM call that decides whether a document belongs in the index.

Mirrors the gate phase of the knowledge-ingest-pipeline's wf2_process.sh (batched
qwen3:1.7b call against Ollama's /api/generate with format="json") but adapted to a
single-document Gate.keep() call — the pipeline itself does the batching by stage,
keeping the gate model resident for the whole batch before moving on.

Failure mode is deliberate: a bad HTTP call, an empty response, or unparseable JSON
all raise. The pipeline (see pipeline.py) catches that and records the source as
state='error' with the message — never silently keeps or drops a document because the
gate had a bad day.
"""
from __future__ import annotations

import httpx

from .config import OllamaConfig
from .models import SourceDoc
from .ollama import generate_json

_TEXT_CHARS = 2000

_SYSTEM_TEMPLATE = (
    "You are a relevance gate for a document collection. Decide whether to KEEP or "
    "REJECT a document using this criteria:\n{criteria}\n"
    "Output JSON only: {{\"keep\": true|false, \"reason\": \"<=10 words\"}}"
)


class RelevanceGate:
    def __init__(self, ollama_cfg: OllamaConfig, criteria: str, timeout: float = 90.0):
        self.cfg = ollama_cfg
        self.criteria = criteria
        self._client = httpx.Client(timeout=timeout)

    def keep(self, doc: SourceDoc) -> tuple[bool, str]:
        text = (doc.text or "")[:_TEXT_CHARS]
        system = _SYSTEM_TEMPLATE.format(criteria=self.criteria)
        prompt = f"Title: {doc.title}\n\n{text}\n\nReturn JSON: {{\"keep\":true|false,\"reason\":\"<=10 words\"}}"
        inner = generate_json(self._client, self.cfg.url, self.cfg.gate_model,
                              system=system, prompt=prompt)
        if not isinstance(inner, dict) or "keep" not in inner or not isinstance(inner["keep"], bool):
            raise RuntimeError(f"gate: missing/invalid 'keep' field: {inner!r}")
        return bool(inner["keep"]), str(inner.get("reason", ""))

    def close(self) -> None:
        self._client.close()
