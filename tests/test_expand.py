from __future__ import annotations

from ragkit.expand import parse_expansions, rrf_fuse


def test_parse_expansions_dedups_and_caps():
    raw = {"queries": ["how much rent", "HOW MUCH RENT", "monthly rent due", "rent amount", "lease cost"]}
    out = parse_expansions(raw, original="how much rent", n=2)
    assert out == ["monthly rent due", "rent amount"]   # original + dup dropped, capped to 2


def test_parse_expansions_accepts_bare_list_and_handles_junk():
    assert parse_expansions(["a query here"], "orig", 3) == ["a query here"]
    assert parse_expansions({"nope": 1}, "orig", 3) == []
    assert parse_expansions("garbage", "orig", 3) == []


def test_rrf_fuse_ranks_by_reciprocal_rank_across_lists():
    a = [{"id": 1, "payload": {"t": "a1"}}, {"id": 2}, {"id": 3}]
    b = [{"id": 2, "payload": {"t": "b2"}}, {"id": 4}, {"id": 1}]
    fused = rrf_fuse([a, b], k=60)
    ids = [h["id"] for h in fused]
    # 2 appears at ranks 2 and 1; 1 at ranks 1 and 3 -> both high; unique; 3 and 4 lower
    assert set(ids) == {1, 2, 3, 4}
    assert ids[0] in (1, 2) and ids[1] in (1, 2)
    # first-seen payload kept
    assert next(h for h in fused if h["id"] == 1)["payload"] == {"t": "a1"}


def test_rrf_fuse_skips_hits_without_id():
    fused = rrf_fuse([[{"score": 1}, {"id": 5}]])
    assert [h["id"] for h in fused] == [5]
