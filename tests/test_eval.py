from __future__ import annotations

import pytest

from ragkit.eval import (
    EvalReport, GoldenItem, compare, distinct_in_order, evaluate,
    first_relevant_rank, generate_golden, load_golden, load_report,
    recall_at_k, save_golden, save_report,
)


def test_distinct_in_order_collapses_and_preserves_first():
    assert distinct_in_order(["a", "b", "a", "c", ""]) == ["a", "b", "c"]


def test_recall_at_k_hit_and_miss():
    assert recall_at_k(["a", "b", "c"], ["b"], 2) == 1.0
    assert recall_at_k(["a", "b", "c"], ["z"], 3) == 0.0
    assert recall_at_k(["a", "b", "c"], ["a", "z"], 3) == 0.5   # multi-relevant partial


def test_first_relevant_rank():
    assert first_relevant_rank(["a", "b", "c"], ["b"]) == 2
    assert first_relevant_rank(["a", "b"], ["z"]) is None


def test_generate_golden_respects_n_and_filters():
    def gen(text, n):
        return ["What is the cap rate for this property?", "hi", "What is the cap rate for this property?"]
    items, skipped = generate_golden([("file:///a.pdf", "body")], gen, per_doc_n=3)
    assert skipped == 0
    # "hi" dropped (too short), dup dropped -> one item
    assert [i.query for i in items] == ["What is the cap rate for this property?"]
    assert items[0].relevant_uris == ["file:///a.pdf"]


def test_generate_golden_drops_filename_leak():
    def gen(text, n):
        return ["What does a1234.pdf say about the roof replacement?"]
    items, _ = generate_golden([("file:///a1234.pdf", "body")], gen, per_doc_n=1)
    assert items == []


def test_generate_golden_isolates_failing_doc():
    def gen(text, n):
        if text == "boom":
            raise RuntimeError("model down")
        return ["What is the total monthly rent collected across units?"]
    docs = [("file:///a.pdf", "boom"), ("file:///b.pdf", "ok")]
    items, skipped = generate_golden(docs, gen, per_doc_n=1)
    assert skipped == 1
    assert [i.relevant_uris for i in items] == [["file:///b.pdf"]]


def test_evaluate_metrics_and_error_isolation():
    golden = [
        GoldenItem("q1", ["a"]),   # a at rank 1
        GoldenItem("q2", ["b"]),   # b at rank 2 (after dup collapse)
        GoldenItem("q3", ["z"]),   # boom -> errored, miss
    ]

    def search_fn(query, top_k):
        if query == "q1":
            return ["a", "a", "x"]
        if query == "q2":
            return ["x", "b", "y"]
        raise RuntimeError("search down")

    recall, mrr, per_query, errored = evaluate(golden, search_fn, [1, 3])
    assert errored == 1
    assert recall[1] == pytest.approx(1 / 3)     # only q1 hits at k=1
    assert recall[3] == pytest.approx(2 / 3)     # q1,q2 hit within 3; q3 miss
    assert mrr == pytest.approx((1.0 + 0.5 + 0.0) / 3)
    assert per_query[2]["first_rank"] is None and per_query[2]["retrieved_uris"] == []


def test_compare_deltas_and_warnings():
    base = EvalReport("c", 3, [1, 3], False, "m", {1: 0.3, 3: 0.6}, 0.4, [])
    cur = EvalReport("c", 3, [1, 3], True, "m", {1: 0.5, 3: 0.7}, 0.55, [])
    out = compare(base, cur)
    assert out["recall_at_k"][1] == (0.3, 0.5, pytest.approx(0.2))
    assert out["mrr"][2] == pytest.approx(0.15)
    assert out["warnings"] == []
    mism = compare(base, EvalReport("d", 4, [1], True, "m", {1: 0.5}, 0.5, []))
    assert any("collection differs" in w for w in mism["warnings"])


def test_golden_and_report_roundtrip(tmp_path):
    items = [GoldenItem("q?", ["file:///a.pdf"])]
    gp = str(tmp_path / "g.jsonl")
    save_golden(items, gp)
    assert load_golden(gp) == items
    rep = EvalReport("c", 1, [1, 5], False, None, {1: 0.0, 5: 1.0}, 0.5, [{"query": "q?"}])
    rp = str(tmp_path / "r.json")
    save_report(rep, rp)
    assert load_report(rp) == rep
