from __future__ import annotations

import pytest

from ragkit.config import QdrantConfig
from ragkit.models import Chunk
from ragkit.store import Store


class _Resp:
    def __init__(self, status=200, payload=None):
        self.status_code = status
        self._p = payload or {}
    def raise_for_status(self): pass
    def json(self): return self._p


class _FakeClient:
    def __init__(self, get_status=404, get_payload=None):
        self.calls = []
        self._gs, self._gp = get_status, get_payload
    def get(self, url):
        self.calls.append(("GET", url, None)); return _Resp(self._gs, self._gp)
    def put(self, url, json=None):
        self.calls.append(("PUT", url, json)); return _Resp(200)
    def post(self, url, json=None):
        self.calls.append(("POST", url, json)); return _Resp(200, {"result": []})
    def close(self): pass


def _store(client):
    s = Store(QdrantConfig(), "c")
    s._client = client
    return s


def test_ensure_collection_creates_named_dense_and_sparse():
    s = _store(_FakeClient(get_status=404))
    s.ensure_collection(1024)
    body = next(c[2] for c in s._client.calls if c[0] == "PUT" and c[1].endswith("/collections/c"))
    assert body["vectors"]["dense"]["size"] == 1024
    assert body["vectors"]["dense"]["distance"] == "Cosine"
    assert "sparse" in body["sparse_vectors"]


def test_ensure_collection_rejects_legacy_unnamed_schema():
    legacy = {"result": {"config": {"params": {"vectors": {"size": 768, "distance": "Cosine"}}}}}
    s = _store(_FakeClient(get_status=200, get_payload=legacy))
    with pytest.raises(Exception):
        s.ensure_collection(1024)


def test_ensure_collection_noops_when_dense_dim_matches():
    named = {"result": {"config": {"params": {"vectors": {"dense": {"size": 1024, "distance": "Cosine"}}}}}}
    s = _store(_FakeClient(get_status=200, get_payload=named))
    s.ensure_collection(1024)
    assert not any(c[0] == "PUT" for c in s._client.calls)   # no create


def test_upsert_writes_named_dense_and_sparse():
    s = _store(_FakeClient())
    s.upsert([Chunk("u", 0, "t")], [[0.1, 0.2, 0.3, 0.4]],
             [{"indices": [7], "values": [0.5]}], source_type="files")
    pt = next(c[2] for c in s._client.calls if c[0] == "PUT" and "/points" in c[1])["points"][0]
    assert pt["vector"]["dense"] == [0.1, 0.2, 0.3, 0.4]
    assert pt["vector"]["sparse"] == {"indices": [7], "values": [0.5]}


def test_search_uses_named_dense_vector():
    s = _store(_FakeClient())
    s.search([0.1, 0.2], top_k=3)
    body = next(c[2] for c in s._client.calls if c[0] == "POST" and c[1].endswith("/points/search"))
    assert body["vector"] == {"name": "dense", "vector": [0.1, 0.2]}
    assert body["limit"] == 3
