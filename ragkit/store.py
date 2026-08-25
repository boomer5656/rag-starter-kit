"""Qdrant vector store: collection lifecycle, payload indexes, idempotent upsert.

Idempotency is structural: a chunk's point id is a stable hash of
(source_uri, ordinal), so re-ingesting the same document upserts the same points
in place — never duplicates. The collection's vector size is checked against the
embedder's dimension before any write (see Store.ensure_collection), making the
768-vs-1024 cross-wire bug impossible.
"""
from __future__ import annotations

import httpx

from .config import QdrantConfig
from .models import Chunk


class DimensionMismatch(RuntimeError):
    pass


class Store:
    def __init__(self, cfg: QdrantConfig, collection: str, timeout: float = 120.0):
        self.cfg = cfg
        self.collection = collection
        self._client = httpx.Client(timeout=timeout)

    def _url(self, suffix: str = "") -> str:
        return f"{self.cfg.url}/collections/{self.collection}{suffix}"

    def exists(self) -> bool:
        return self._client.get(self._url()).status_code == 200

    def current_dim(self) -> int | None:
        r = self._client.get(self._url())
        if r.status_code != 200:
            return None
        cfg = r.json().get("result", {}).get("config", {}).get("params", {}).get("vectors", {})
        # unnamed-vector collections put size at vectors.size
        return cfg.get("size") if isinstance(cfg, dict) else None

    def ensure_collection(self, dim: int, index_fields: dict[str, str] | None = None) -> None:
        """Create the collection at `dim` if absent; if present, assert its dim matches.

        `index_fields` maps payload field -> schema ("keyword"|"integer"|"bool"|"float")
        so connectors can declare what they want to filter on.
        """
        existing = self.current_dim()
        if existing is not None:
            if existing != dim:
                raise DimensionMismatch(
                    f"collection '{self.collection}' is dim={existing} but the embedder "
                    f"produces dim={dim}. Refusing to write (this is the cross-wire bug). "
                    f"Use a fresh collection or re-embed."
                )
            return
        self._client.put(
            self._url(),
            json={"vectors": {"size": dim, "distance": self.cfg.distance}, "on_disk_payload": True},
        ).raise_for_status()
        for field_name, schema in (index_fields or {}).items():
            # index creation is best-effort; a duplicate/late index shouldn't abort ingest
            self._client.put(self._url("/index"),
                             json={"field_name": field_name, "field_schema": schema})

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]],
               source_type: str) -> None:
        assert len(chunks) == len(vectors)
        points = []
        for ch, vec in zip(chunks, vectors):
            payload = {
                "text": ch.text,
                "source_uri": ch.source_uri,
                "ordinal": ch.ordinal,
                "source_type": source_type,
                **ch.meta,
            }
            points.append({"id": ch.id, "vector": vec, "payload": payload})
        if points:
            self._client.put(self._url("/points?wait=true"),
                             json={"points": points}).raise_for_status()

    def delete_source(self, source_uri: str) -> None:
        """Remove all points for a source_uri (used when re-chunking changes ordinals)."""
        self._client.post(
            self._url("/points/delete?wait=true"),
            json={"filter": {"must": [{"key": "source_uri", "match": {"value": source_uri}}]}},
        )

    def search(self, vector: list[float], top_k: int = 5,
               query_filter: dict | None = None) -> list[dict]:
        body: dict = {"vector": vector, "limit": top_k, "with_payload": True}
        if query_filter:
            body["filter"] = query_filter
        r = self._client.post(self._url("/points/search"), json=body)
        r.raise_for_status()
        return r.json().get("result", [])

    def scroll_points(self, page: int = 256):
        """Yield every stored point (payload only) via Qdrant's scroll API.
        Used by the eval harness to reconstruct source docs from the collection."""
        offset = None
        while True:
            body: dict = {"limit": page, "with_payload": True, "with_vector": False}
            if offset is not None:
                body["offset"] = offset
            r = self._client.post(self._url("/points/scroll"), json=body)
            r.raise_for_status()
            result = r.json().get("result", {})
            points = result.get("points", [])
            for p in points:
                yield p
            offset = result.get("next_page_offset")
            if offset is None or not points:
                break

    def close(self) -> None:
        self._client.close()
