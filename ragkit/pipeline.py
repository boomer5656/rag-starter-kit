"""The orchestrator — the part that makes ragkit ragkit.

Stage-batched execution: every document clears one stage before any moves to the
next, so a resident model (the gate model, then bge-m3) is loaded once per stage
instead of thrashed once per document. On a single-GPU homelab that's the
difference between ~40s/doc and near-continuous throughput.

State is written after every stage transition, so a crash resumes from the last
completed stage. An error at any stage is per-document and *visible* — the source
is marked state='error' with its message and dropped from later stages, never
silently discarded, never aborting the whole run.
"""
from __future__ import annotations

from typing import Iterable, Protocol

from .config import Config
from .embed import Embedder
from .models import Chunk, PipelineResult, SourceDoc, State
from .state import StateStore
from .store import Store


class Extractor(Protocol):
    def extract(self, doc: SourceDoc) -> str: ...


class Chunker(Protocol):
    def chunk(self, source_uri: str, text: str, meta: dict) -> list[Chunk]: ...


class Gate(Protocol):
    def keep(self, doc: SourceDoc) -> tuple[bool, str]: ...


class ContextualizerLike(Protocol):
    def contextualize(self, doc_text: str, chunks: list[Chunk]) -> list[str]: ...


class Pipeline:
    def __init__(self, cfg: Config, *, extractor: Extractor, chunker: Chunker,
                 embedder: Embedder, store: Store, state: StateStore,
                 gate: Gate | None = None, contextualizer: ContextualizerLike | None = None,
                 index_fields: dict[str, str] | None = None):
        self.cfg = cfg
        self.extractor = extractor
        self.chunker = chunker
        self.embedder = embedder
        self.store = store
        self.state = state
        self.gate = gate
        self.contextualizer = contextualizer
        self.index_fields = index_fields or {}

    def run(self, docs: Iterable[SourceDoc], *, source_type: str = "document",
            gate_enabled: bool | None = None) -> PipelineResult:
        res = PipelineResult()
        use_gate = self.cfg.gate_enabled if gate_enabled is None else gate_enabled

        # --- discover ---
        batch = list(docs)
        res.discovered = len(batch)
        for d in batch:
            self.state.set(d.uri, State.DISCOVERED)

        # --- extract (fill text; Tika stays warm across the batch) ---
        extracted: list[SourceDoc] = []
        for d in batch:
            try:
                if not d.text:
                    d.text = self.extractor.extract(d)
                if not d.text or not d.text.strip():
                    raise ValueError("no extractable text")
                self.state.set(d.uri, State.EXTRACTED)
                extracted.append(d)
            except Exception as e:  # noqa: BLE001 — per-doc isolation is the point
                self._err(res, d.uri, f"extract: {e}")
        res.extracted = len(extracted)

        # --- gate (optional; gate model resident for the whole batch) ---
        if use_gate and self.gate is not None:
            gated: list[SourceDoc] = []
            for d in extracted:
                try:
                    keep, reason = self.gate.keep(d)
                    if keep:
                        self.state.set(d.uri, State.GATED)
                        gated.append(d)
                    else:
                        self.state.set(d.uri, State.SKIPPED, error=f"gate: {reason}")
                        res.skipped += 1
                except Exception as e:  # noqa: BLE001
                    self._err(res, d.uri, f"gate: {e}")
            res.gated = len(gated)
        else:
            gated = extracted
            for d in gated:
                self.state.set(d.uri, State.GATED)
            res.gated = len(gated)

        # --- chunk (CPU; cheap) ---
        chunked: list[tuple[SourceDoc, list[Chunk]]] = []
        for d in gated:
            try:
                chunks = self.chunker.chunk(d.uri, d.text or "", d.meta)
                if not chunks:
                    raise ValueError("chunker produced 0 chunks")
                self.state.set(d.uri, State.CHUNKED, n_chunks=len(chunks))
                chunked.append((d, chunks))
                res.chunked += len(chunks)
            except Exception as e:  # noqa: BLE001
                self._err(res, d.uri, f"chunk: {e}")

        if not chunked:
            return res

        # --- ensure collection once (dim probe asserts against any existing collection) ---
        dim = self.embedder.dim
        self.store.ensure_collection(dim, self.index_fields)

        # --- embed + store (bge-m3 resident for the whole batch) ---
        for d, chunks in chunked:
            try:
                texts = [c.text for c in chunks]
                if self.contextualizer is not None:
                    ctxs = self.contextualizer.contextualize(d.text or "", chunks)
                    texts = [f"{ctx}\n{c.text}" if ctx else c.text
                             for ctx, c in zip(ctxs, chunks)]
                vectors = self.embedder.embed_batch(texts)
                bad = next((v for v in vectors if len(v) != dim), None)
                if bad is not None:
                    raise ValueError(f"embedding dim {len(bad)} != expected {dim}")
                self.state.set(d.uri, State.EMBEDDED, dim=dim, n_chunks=len(chunks))
                res.embedded += len(chunks)
                # re-chunking may change ordinals; clear the source then upsert fresh
                self.store.delete_source(d.uri)
                self.store.upsert(chunks, vectors, source_type=source_type)
                self.state.set(d.uri, State.STORED, dim=dim, n_chunks=len(chunks))
                res.stored += len(chunks)
            except Exception as e:  # noqa: BLE001
                self._err(res, d.uri, f"embed/store: {e}")

        return res

    def _err(self, res: PipelineResult, uri: str, msg: str) -> None:
        self.state.set(uri, State.ERROR, error=msg)
        res.errors += 1
        res.error_uris.append(uri)
