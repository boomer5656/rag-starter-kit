"""Unit tests for ragkit.state.StateStore -- sqlite only, no network.

Every test gets its own sqlite file under pytest's tmp_path (either built
directly, or via the tmp_config fixture in conftest.py) so nothing leaks
between tests or touches a real `.ragkit/state.db`.
"""
from __future__ import annotations

from ragkit.config import StateConfig
from ragkit.models import State
from ragkit.state import StateStore


def _make_store(tmp_path, collection: str = "test-collection") -> StateStore:
    cfg = StateConfig(backend="sqlite", sqlite_path=str(tmp_path / "state.db"))
    return StateStore(cfg, collection)


def test_set_then_get_round_trips(tmp_path):
    store = _make_store(tmp_path)
    store.set("file:///a.txt", State.EMBEDDED, dim=1024, n_chunks=7)
    got = store.get("file:///a.txt")
    assert got is not None
    assert got.uri == "file:///a.txt"
    assert got.state == State.EMBEDDED
    assert got.dim == 1024
    assert got.n_chunks == 7
    assert got.error is None
    assert got.updated_at is not None
    store.close()


def test_get_missing_uri_returns_none(tmp_path):
    store = _make_store(tmp_path)
    assert store.get("file:///missing.txt") is None
    store.close()


def test_counts_aggregates_by_state(tmp_path):
    store = _make_store(tmp_path)
    store.set("a", State.DISCOVERED)
    store.set("b", State.DISCOVERED)
    store.set("c", State.STORED, dim=1024, n_chunks=3)
    store.set("d", State.ERROR, error="boom")
    counts = store.counts()
    assert counts[State.DISCOVERED.value] == 2
    assert counts[State.STORED.value] == 1
    assert counts[State.ERROR.value] == 1
    assert sum(counts.values()) == 4
    store.close()


def test_counts_only_reflects_current_collection(tmp_path):
    cfg = StateConfig(backend="sqlite", sqlite_path=str(tmp_path / "state.db"))
    store_a = StateStore(cfg, "collection-a")
    store_b = StateStore(cfg, "collection-b")
    store_a.set("file:///x.txt", State.STORED, dim=1024, n_chunks=1)
    assert store_a.counts() == {State.STORED.value: 1}
    assert store_b.counts() == {}
    store_a.close()
    store_b.close()


def test_error_row_is_retained_and_surfaced_by_errors(tmp_path):
    store = _make_store(tmp_path)
    store.set("file:///bad.txt", State.ERROR, error="extract: boom")
    store.set("file:///good.txt", State.STORED, dim=1024, n_chunks=1)
    errs = store.errors()
    assert len(errs) == 1
    assert errs[0].uri == "file:///bad.txt"
    assert errs[0].state == State.ERROR
    assert errs[0].error == "extract: boom"
    store.close()


def test_errors_respects_limit(tmp_path):
    store = _make_store(tmp_path)
    for i in range(5):
        store.set(f"file:///bad{i}.txt", State.ERROR, error=f"boom {i}")
    errs = store.errors(limit=2)
    assert len(errs) == 2
    store.close()


def test_resetting_same_uri_updates_in_place_no_duplicate(tmp_path):
    store = _make_store(tmp_path)
    store.set("file:///a.txt", State.DISCOVERED)
    store.set("file:///a.txt", State.EXTRACTED)
    store.set("file:///a.txt", State.CHUNKED, n_chunks=4)
    got = store.get("file:///a.txt")
    assert got.state == State.CHUNKED
    assert got.n_chunks == 4
    counts = store.counts()
    assert sum(counts.values()) == 1
    assert counts[State.CHUNKED.value] == 1
    store.close()


def test_resetting_to_error_after_success_replaces_previous_row(tmp_path):
    store = _make_store(tmp_path)
    store.set("file:///a.txt", State.STORED, dim=1024, n_chunks=2)
    store.set("file:///a.txt", State.ERROR, error="embed/store: timeout")
    got = store.get("file:///a.txt")
    assert got.state == State.ERROR
    assert got.error == "embed/store: timeout"
    errs = store.errors()
    assert len(errs) == 1
    assert errs[0].uri == "file:///a.txt"
    assert sum(store.counts().values()) == 1  # still one row, not two
    store.close()


def test_recovering_from_error_clears_it_from_errors_list(tmp_path):
    store = _make_store(tmp_path)
    store.set("file:///a.txt", State.ERROR, error="extract: boom")
    assert len(store.errors()) == 1
    store.set("file:///a.txt", State.STORED, dim=1024, n_chunks=1)
    assert store.errors() == []
    got = store.get("file:///a.txt")
    assert got.state == State.STORED
    assert got.error is None
    store.close()


def test_state_is_scoped_per_collection(tmp_path):
    cfg = StateConfig(backend="sqlite", sqlite_path=str(tmp_path / "state.db"))
    store_a = StateStore(cfg, "collection-a")
    store_b = StateStore(cfg, "collection-b")
    store_a.set("file:///shared.txt", State.STORED, dim=1024, n_chunks=1)
    assert store_a.get("file:///shared.txt") is not None
    assert store_b.get("file:///shared.txt") is None
    store_a.close()
    store_b.close()


def test_state_store_from_tmp_config_fixture(tmp_config):
    store = StateStore(tmp_config.state, tmp_config.collection)
    store.set("file:///a.txt", State.DISCOVERED)
    got = store.get("file:///a.txt")
    assert got is not None
    assert got.state == State.DISCOVERED
    store.close()


def test_unsupported_backend_raises_not_implemented(tmp_path):
    cfg = StateConfig(backend="postgres", postgres_dsn="postgresql://example/db")
    try:
        StateStore(cfg, "test-collection")
        assert False, "expected NotImplementedError for the unwired postgres backend"
    except NotImplementedError:
        pass
