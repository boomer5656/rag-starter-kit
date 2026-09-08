"""Qdrant vector store: collection lifecycle, payload indexes, idempotent upsert.

Idempotency is structural: a chunk's point id is a stable hash of
(source_uri, ordinal), so re-ingesting the same document upserts the same points
in place — never duplicates. The collection's vector size is checked against the
embedder's dimension before any write (see Store.ensure_collection), making the
768-vs-1024 cross-wire bug impossible.
"""
from __future__ import annotations

import os

import httpx

from .config import QdrantConfig
from .models import Chunk


class DimensionMismatch(RuntimeError):
    pass


class Store:
    def __init__(self, cfg: QdrantConfig, collection: str, timeout: float = 120.0):
        self.cfg = cfg
        self.collection = collection
        # Tower Qdrant is API-keyed since 2026-09-08; unset = no header (keyless local Qdrant unchanged).
        key = os.environ.get("QDRANT_API_KEY", "")
        self._client = httpx.Client(timeout=timeout, headers=({"api-key": key} if key else {}))

    def _url(self, suffix: str = "") -> str:
        return f"{self.cfg.url}/collections/{self.collection}{suffix}"

    def exists(self) -> bool:
        return self._client.get(self._url()).status_code == 200

    def _vectors_cfg(self):
        r = self._client.get(self._url())
        if r.status_code != 200:
            return None
        return r.json().get("result", {}).get("config", {}).get("params", {}).get("vectors", {})

    def current_dim(self) -> int | None:
        v = self._vectors_cfg()
        if not isinstance(v, dict):
            return None
        dense = v.get("dense")
        if isinstance(dense, dict):
            return dense.get("size")
        return v.get("size")  # legacy unnamed collection (pre-hybrid)

    def ensure_collection(self, dim: int, index_fields: dict[str, str] | None = None) -> None:
        """Create a named-vector (dense) + sparse collection if absent; else assert its
        dense dim matches. A pre-hybrid unnamed collection is rejected — recreate it."""
        v = self._vectors_cfg()
        if v is not None:
            dense = v.get("dense") if isinstance(v, dict) else None
            if not isinstance(dense, dict):
                raise DimensionMismatch(
                    f"collection '{self.collection}' predates the hybrid (named-vector) schema. "
                    f"Delete it or use a fresh collection, then re-ingest."
                )
            if dense.get("size") != dim:
                raise DimensionMismatch(
                    f"collection '{self.collection}' is dim={dense.get('size')} but the embedder "
                    f"produces dim={dim}. Use a fresh collection or re-embed."
                )
            return
        self._client.put(
            self._url(),
            json={
                "vectors": {"dense": {"size": dim, "distance": self.cfg.distance}},
                "sparse_vectors": {"sparse": {}},
                "on_disk_payload": True,
            },
        ).raise_for_status()
        for field_name, schema in (index_fields or {}).items():
            self._client.put(self._url("/index"),
                             json={"field_name": field_name, "field_schema": schema})

    def upsert(self, chunks: list[Chunk], dense_vectors: list[list[float]],
               sparse_vectors: list[dict], source_type: str) -> None:
        assert len(chunks) == len(dense_vectors) == len(sparse_vectors)
        points = []
        for ch, dv, sv in zip(chunks, dense_vectors, sparse_vectors):
            payload = {
                "text": ch.text,
                "source_uri": ch.source_uri,
                "ordinal": ch.ordinal,
                "source_type": source_type,
                **ch.meta,
            }
            points.append({"id": ch.id, "vector": {"dense": dv, "sparse": sv}, "payload": payload})
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
        body: dict = {"vector": {"name": "dense", "vector": vector},
                      "limit": top_k, "with_payload": True}
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

    def query_hybrid(self, dense_vector: list[float], sparse_vector: dict,
                     top_k: int = 5, query_filter: dict | None = None,
                     prefetch: int = 50) -> list[dict]:
        """Dense + sparse legs fused server-side with RRF (Qdrant Query API)."""
        legs = [
            {"query": dense_vector, "using": "dense", "limit": prefetch},
            {"query": {"indices": sparse_vector.get("indices", []),
                       "values": sparse_vector.get("values", [])},
             "using": "sparse", "limit": prefetch},
        ]
        if query_filter:
            for leg in legs:
                leg["filter"] = query_filter
        body = {"prefetch": legs, "query": {"fusion": "rrf"},
                "limit": top_k, "with_payload": True}
        r = self._client.post(self._url("/points/query"), json=body)
        r.raise_for_status()
        return r.json().get("result", {}).get("points", [])

    def close(self) -> None:
        self._client.close()
