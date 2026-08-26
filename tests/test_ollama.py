from __future__ import annotations

import pytest

from ragkit.ollama import generate_json


class _Resp:
    def __init__(self, payload): self._p = payload
    def raise_for_status(self): pass
    def json(self): return self._p


class _Client:
    def __init__(self, payload): self._p = payload; self.last = None
    def post(self, url, json): self.last = json; return _Resp(self._p)


def test_generate_json_contract_and_parse():
    c = _Client({"response": '{"ok": true}'})
    out = generate_json(c, "http://x", "m", system="s", prompt="p")
    assert out == {"ok": True}
    assert c.last["think"] is False
    assert c.last["format"] == "json"
    assert c.last["model"] == "m"
    assert 0 < c.last["options"]["num_predict"] <= 1024


def test_generate_json_raises_on_empty():
    with pytest.raises(RuntimeError):
        generate_json(_Client({"response": ""}), "http://x", "m", system="s", prompt="p")
