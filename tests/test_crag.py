from __future__ import annotations

from ragkit.crag import DROP_THRESHOLD, apply_grades, is_weak, parse_grades


def test_parse_grades_clamps_and_orders():
    assert parse_grades({"grades": [0.9, -1, 2, "x"]}, 4) == [0.9, 0.0, 1.0, 0.5]


def test_parse_grades_failsafe_neutral_on_junk():
    assert parse_grades({"nope": 1}, 3) == [0.5, 0.5, 0.5]
    assert parse_grades("garbage", 2) == [0.5, 0.5]
    assert parse_grades({"grades": [0.8]}, 3) == [0.8, 0.5, 0.5]   # short -> padded neutral


def test_apply_grades_drops_below_threshold_keeps_order_and_best():
    hits = [{"id": 1}, {"id": 2}, {"id": 3}]
    kept, best = apply_grades(hits, [0.9, 0.1, 0.6], drop=0.25)
    assert [h["id"] for h in kept] == [1, 3]      # id 2 (0.1) dropped, order preserved
    assert kept[0]["crag_grade"] == 0.9           # grade attached
    assert best == 0.9


def test_apply_grades_empty_is_zero_best():
    kept, best = apply_grades([], [])
    assert kept == [] and best == 0.0


def test_is_weak_trigger():
    assert is_weak(0.3, floor=0.5) is True
    assert is_weak(0.7, floor=0.5) is False
    assert is_weak(0.0) is True
    assert DROP_THRESHOLD < 1.0
