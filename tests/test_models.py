"""Unit tests for ragkit.models: stable_id, Chunk.id, and dataclass defaults.

Pure-unit: hashing and dataclass construction only, no network.
"""
from __future__ import annotations

from ragkit.models import Chunk, IngestState, PipelineResult, SourceDoc, State, stable_id

_SIGNED_63_BIT_MAX = 0x7FFFFFFFFFFFFFFF  # 2**63 - 1


def test_stable_id_is_deterministic():
    assert stable_id("a", 1) == stable_id("a", 1)
    assert stable_id("doc://foo", 0) == stable_id("doc://foo", 0)


def test_stable_id_differs_for_different_inputs():
    assert stable_id("a", 1) != stable_id("a", 2)
    assert stable_id("a", 1) != stable_id("b", 1)


def test_stable_id_stays_in_signed_63_bit_range():
    samples = [stable_id(i, "x", i * 7, "tail") for i in range(300)]
    for sid in samples:
        assert isinstance(sid, int)
        assert 0 <= sid <= _SIGNED_63_BIT_MAX
        assert sid.bit_length() <= 63


def test_stable_id_handles_varied_arg_shapes():
    assert isinstance(stable_id(), int)
    assert isinstance(stable_id(1, 2.5, None, "x"), int)
    # order matters -- it's not a set/bag hash
    assert stable_id("a", "b") != stable_id("b", "a")


def test_chunk_id_is_stable_for_same_source_uri_and_ordinal():
    c1 = Chunk(source_uri="file:///a.txt", ordinal=3, text="hello")
    c2 = Chunk(source_uri="file:///a.txt", ordinal=3, text="a completely different body")
    assert c1.id == c2.id  # id depends on (source_uri, ordinal), not text


def test_chunk_id_differs_across_ordinals():
    base = Chunk(source_uri="file:///a.txt", ordinal=0, text="hello")
    other = Chunk(source_uri="file:///a.txt", ordinal=1, text="hello")
    assert base.id != other.id


def test_chunk_id_differs_across_source_uri():
    c1 = Chunk(source_uri="file:///a.txt", ordinal=0, text="hello")
    c2 = Chunk(source_uri="file:///b.txt", ordinal=0, text="hello")
    assert c1.id != c2.id


def test_chunk_id_matches_stable_id_helper_directly():
    c = Chunk(source_uri="file:///a.txt", ordinal=5, text="hello")
    assert c.id == stable_id(c.source_uri, c.ordinal)


def test_chunk_id_is_in_signed_63_bit_range():
    c = Chunk(source_uri="file:///a.txt", ordinal=999999, text="hello")
    assert 0 <= c.id <= _SIGNED_63_BIT_MAX


def test_source_doc_constructs_with_defaults():
    d = SourceDoc(uri="file:///a.txt")
    assert d.uri == "file:///a.txt"
    assert d.title == ""
    assert d.mime == ""
    assert d.text is None
    assert d.raw_bytes is None
    assert d.meta == {}


def test_source_doc_meta_default_is_independent_per_instance():
    d1 = SourceDoc(uri="a")
    d2 = SourceDoc(uri="b")
    d1.meta["x"] = 1
    assert d2.meta == {}  # default_factory=dict must not share a mutable default


def test_chunk_constructs_with_defaults():
    c = Chunk(source_uri="file:///a.txt", ordinal=0, text="hi")
    assert c.meta == {}


def test_chunk_meta_default_is_independent_per_instance():
    c1 = Chunk(source_uri="a", ordinal=0, text="x")
    c2 = Chunk(source_uri="a", ordinal=1, text="y")
    c1.meta["k"] = "v"
    assert c2.meta == {}


def test_ingest_state_constructs_with_defaults():
    st = IngestState(uri="file:///a.txt", state=State.DISCOVERED)
    assert st.dim is None
    assert st.n_chunks is None
    assert st.error is None
    assert st.updated_at is None


def test_pipeline_result_defaults_and_summary():
    res = PipelineResult()
    assert res.discovered == 0
    assert res.extracted == 0
    assert res.gated == 0
    assert res.skipped == 0
    assert res.chunked == 0
    assert res.embedded == 0
    assert res.stored == 0
    assert res.errors == 0
    assert res.error_uris == []
    summary = res.summary()
    assert "discovered=0" in summary
    assert "errors=0" in summary


def test_pipeline_result_error_uris_default_is_independent_per_instance():
    r1 = PipelineResult()
    r2 = PipelineResult()
    r1.error_uris.append("file:///bad.txt")
    assert r2.error_uris == []
