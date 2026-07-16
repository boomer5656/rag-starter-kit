"""Core data types passed between pipeline stages.

Deliberately plain dataclasses — no ORM, no pydantic. A connector yields SourceDocs;
the pipeline turns each into Chunks, embeds them, and stores them, tracking IngestState
per source along the way.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


def stable_id(*parts: Any) -> int:
    """Deterministic unsigned 63-bit id from the given parts.

    Used for both point ids (source_uri + ordinal) and any place we need a
    re-run to land on the same id so upserts replace rather than duplicate.
    Qdrant accepts unsigned ints; we stay within signed-64 range to be safe.
    """
    h = hashlib.sha256("\x1f".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return int(h[:16], 16) & 0x7FFFFFFFFFFFFFFF


class State(str, Enum):
    """Where a source document is in the pipeline. `error`/`skipped` are terminal.

    Nothing is ever removed from the state store on failure — a document that
    errors stays visible as `error` with its message, so `ragkit status` can
    always answer "what happened to this file?".
    """
    DISCOVERED = "discovered"
    EXTRACTED = "extracted"
    GATED = "gated"          # passed the relevance gate (or gate disabled)
    CHUNKED = "chunked"
    EMBEDDED = "embedded"
    STORED = "stored"
    SKIPPED = "skipped"      # gate rejected it — intentional, not a failure
    ERROR = "error"


@dataclass
class SourceDoc:
    """What a connector yields. Either `text` or `raw_bytes` must be set;
    if only raw_bytes, the extract stage fills in text (via Tika)."""
    uri: str                              # stable, unique per document (a file path, a DOI, ...)
    title: str = ""
    mime: str = ""
    text: Optional[str] = None
    raw_bytes: Optional[bytes] = None
    meta: dict[str, Any] = field(default_factory=dict)   # connector-specific payload fields


@dataclass
class Chunk:
    source_uri: str
    ordinal: int
    text: str
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> int:
        return stable_id(self.source_uri, self.ordinal)


@dataclass
class IngestState:
    uri: str
    state: State
    dim: Optional[int] = None
    n_chunks: Optional[int] = None
    error: Optional[str] = None
    updated_at: Optional[str] = None


@dataclass
class PipelineResult:
    """Returned by pipeline.run() — the per-stage tallies `ragkit ingest` prints."""
    discovered: int = 0
    extracted: int = 0
    gated: int = 0
    skipped: int = 0
    chunked: int = 0
    embedded: int = 0
    stored: int = 0
    errors: int = 0
    error_uris: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"discovered={self.discovered} extracted={self.extracted} "
            f"gated={self.gated} skipped={self.skipped} chunked={self.chunked} "
            f"embedded={self.embedded} stored={self.stored} errors={self.errors}"
        )
