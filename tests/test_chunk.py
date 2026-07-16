"""Unit tests for ragkit.chunk.TokenChunker -- pure text processing, no network.

TokenChunker takes a ChunkConfig and exposes .chunk(source_uri, text, meta) ->
list[Chunk], matching the Chunker protocol in ragkit.pipeline. Token counts are
an approximation (words-per-token heuristic, per the module docstring), so
assertions here check the *intent* -- contiguous ordinals, an approx token
budget, overlap, no empty chunks, meta carried through -- rather than pinning
the exact heuristic constant.
"""
from __future__ import annotations

from ragkit.chunk import TokenChunker
from ragkit.config import ChunkConfig
from ragkit.models import Chunk


def _chunker(max_tokens: int, overlap_tokens: int) -> TokenChunker:
    return TokenChunker(ChunkConfig(max_tokens=max_tokens, overlap_tokens=overlap_tokens))


def _words(n: int) -> str:
    return " ".join(f"word{i}" for i in range(n))


def test_ordinals_are_contiguous_and_zero_based():
    chunker = _chunker(max_tokens=20, overlap_tokens=5)
    chunks = chunker.chunk("file:///a.txt", _words(200), {})
    assert len(chunks) > 1
    assert [c.ordinal for c in chunks] == list(range(len(chunks)))


def test_chunks_are_chunk_instances_with_source_uri_set():
    chunker = _chunker(max_tokens=20, overlap_tokens=0)
    chunks = chunker.chunk("file:///a.txt", _words(50), {})
    assert len(chunks) >= 1
    for c in chunks:
        assert isinstance(c, Chunk)
        assert c.source_uri == "file:///a.txt"


def test_respects_approx_max_tokens():
    max_tokens = 20
    chunker = _chunker(max_tokens=max_tokens, overlap_tokens=0)
    chunks = chunker.chunk("file:///a.txt", _words(300), {})
    assert len(chunks) > 1
    for c in chunks:
        n_words = len(c.text.split())
        # "approx" token budget -- generous slack for whatever words-per-token
        # heuristic is used internally, but a chunk must never balloon to
        # several multiples of the configured budget.
        assert n_words <= max_tokens * 2


def test_more_chunks_for_smaller_max_tokens():
    text = _words(300)
    small = _chunker(max_tokens=10, overlap_tokens=0).chunk("file:///a.txt", text, {})
    large = _chunker(max_tokens=100, overlap_tokens=0).chunk("file:///a.txt", text, {})
    assert len(small) > len(large)


def test_overlap_is_applied_between_consecutive_chunks():
    chunker = _chunker(max_tokens=20, overlap_tokens=8)
    chunks = chunker.chunk("file:///a.txt", _words(150), {})
    assert len(chunks) > 1
    for prev, nxt in zip(chunks, chunks[1:]):
        shared = set(prev.text.split()) & set(nxt.text.split())
        assert shared, "expected overlapping words between consecutive chunks"


def test_larger_overlap_config_yields_more_chunks_for_same_text():
    text = _words(150)
    low_overlap = _chunker(max_tokens=20, overlap_tokens=0).chunk("file:///a.txt", text, {})
    high_overlap = _chunker(max_tokens=20, overlap_tokens=15).chunk("file:///a.txt", text, {})
    assert len(high_overlap) > len(low_overlap)


def test_never_emits_empty_chunks():
    chunker = _chunker(max_tokens=20, overlap_tokens=5)
    chunks = chunker.chunk("file:///a.txt", _words(100), {})
    assert chunks  # sanity: this text does produce chunks
    for c in chunks:
        assert c.text.strip() != ""


def test_empty_text_produces_no_chunks():
    chunker = _chunker(max_tokens=20, overlap_tokens=5)
    assert chunker.chunk("file:///a.txt", "", {}) == []


def test_whitespace_only_text_produces_no_chunks():
    chunker = _chunker(max_tokens=20, overlap_tokens=5)
    assert chunker.chunk("file:///a.txt", "   \n\t  \n  ", {}) == []


def test_short_text_yields_a_single_chunk():
    chunker = _chunker(max_tokens=512, overlap_tokens=64)
    chunks = chunker.chunk("file:///a.txt", "just a few words here", {})
    assert len(chunks) == 1
    assert chunks[0].ordinal == 0
    assert chunks[0].text.split() == "just a few words here".split()


def test_meta_is_carried_through_to_every_chunk():
    chunker = _chunker(max_tokens=20, overlap_tokens=5)
    meta = {"title": "Doc Title", "source_type": "document"}
    chunks = chunker.chunk("file:///a.txt", _words(100), meta)
    assert len(chunks) > 1
    for c in chunks:
        assert c.meta.get("title") == "Doc Title"
        assert c.meta.get("source_type") == "document"


def test_meta_is_copied_not_shared_across_chunks():
    chunker = _chunker(max_tokens=20, overlap_tokens=5)
    meta = {"title": "Doc Title"}
    chunks = chunker.chunk("file:///a.txt", _words(100), meta)
    assert len(chunks) > 1
    chunks[0].meta["mutated"] = True
    assert "mutated" not in chunks[1].meta
    assert "mutated" not in meta  # caller's dict must not be mutated either
