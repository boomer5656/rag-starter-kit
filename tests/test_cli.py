"""CLI wiring tests — network-free via a fake httpx client.

The eval logic lives in ragkit.eval (see test_eval.py); here we pin the one
contract the unit tests there can't reach: the Ollama request `_make_gen_fn`
builds. `think: False` is load-bearing — Qwen3 models otherwise spend their
whole budget in the thinking channel under format="json" and return "" or "{}",
which silently yields an empty golden set (caught only by a live smoke).
"""
from __future__ import annotations

from ragkit.cli import _make_gen_fn


class _FakeResp:
    def __init__(self, payload: dict):
        self._payload = payload

    def raise_for_status(self) -> None:
        pass

    def json(self) -> dict:
        return self._payload


class _FakeClient:
    """Records the last request body; returns a canned questions response."""

    def __init__(self):
        self.last_body: dict | None = None

    def post(self, url: str, json: dict) -> _FakeResp:
        self.last_body = json
        return _FakeResp({"response": '{"questions": ["What is X?", "How does Y work?"]}'})


def test_make_gen_fn_disables_thinking_and_parses():
    client = _FakeClient()
    gen = _make_gen_fn(client, "http://tower:11434", "qwen3.5:9b")

    out = gen("some document text", 2)

    assert out == ["What is X?", "How does Y work?"]
    # Regression guard for the empty-golden-set bug: thinking MUST be disabled.
    assert client.last_body is not None
    assert client.last_body["think"] is False
    assert client.last_body["format"] == "json"
    assert client.last_body["model"] == "qwen3.5:9b"


def test_make_gen_fn_caps_to_n():
    client = _FakeClient()
    gen = _make_gen_fn(client, "http://tower:11434", "m")
    # canned response has 2 questions; asking for 1 must cap.
    assert gen("doc", 1) == ["What is X?"]


def test_make_gen_fn_sets_output_cap():
    # usage-limits doctrine: no uncapped generations.
    client = _FakeClient()
    _make_gen_fn(client, "http://tower:11434", "m")("doc", 2)
    assert 0 < client.last_body["options"]["num_predict"] <= 1024


class _FakeClientBadShape:
    def post(self, url: str, json: dict) -> _FakeResp:
        return _FakeResp({"response": '{"questions": "not a list"}'})


def test_make_gen_fn_non_list_questions_returns_empty():
    # model returns {"questions": "<string>"} -> must NOT iterate into characters.
    gen = _make_gen_fn(_FakeClientBadShape(), "http://x", "m")
    assert gen("doc", 2) == []
