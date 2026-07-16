"""The Connector contract every source adapter implements.

A connector's only job is to yield SourceDocs — it knows nothing about
extraction, chunking, embedding, or storage. `source_type` tags every chunk
this connector produces (for filtering at search time) and `index_fields`
tells the pipeline which payload fields on this connector's docs should get
a Qdrant keyword/integer index, so `Store.ensure_collection` can create them
up front instead of guessing.
"""
from __future__ import annotations

from typing import Iterable, Protocol, runtime_checkable

from ..models import SourceDoc


@runtime_checkable
class Connector(Protocol):
    source_type: str
    index_fields: dict[str, str]

    def iter_docs(self) -> Iterable[SourceDoc]: ...
