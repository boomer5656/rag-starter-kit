from __future__ import annotations

import ragkit.contextualize as C
from ragkit.config import OllamaConfig
from ragkit.models import Chunk


def test_contextualize_batches_and_aligns(monkeypatch):
    calls = []

    def fake_gen(client, url, model, *, system, prompt, **kw):
        n = prompt.count("[chunk ")
        calls.append(n)
        return {"contexts": [f"ctx{i}" for i in range(n)]}

    monkeypatch.setattr(C, "generate_json", fake_gen)
    ctx = C.Contextualizer(OllamaConfig(), "m", max_chunks_per_call=2)
    chunks = [Chunk("u", i, f"chunk{i}") for i in range(5)]
    out = ctx.contextualize("doc text", chunks)
    ctx.close()
    assert len(out) == 5
    assert calls == [2, 2, 1]            # batched by max_chunks_per_call


def test_contextualize_pads_short_response(monkeypatch):
    monkeypatch.setattr(C, "generate_json", lambda *a, **k: {"contexts": ["only-one"]})
    ctx = C.Contextualizer(OllamaConfig(), "m", max_chunks_per_call=10)
    chunks = [Chunk("u", i, f"c{i}") for i in range(3)]
    out = ctx.contextualize("doc", chunks)
    ctx.close()
    assert out == ["only-one", "", ""]   # defensive: one context per chunk


def test_contextualize_non_list_yields_blanks(monkeypatch):
    monkeypatch.setattr(C, "generate_json", lambda *a, **k: {"contexts": "oops"})
    ctx = C.Contextualizer(OllamaConfig(), "m")
    chunks = [Chunk("u", 0, "c0")]
    out = ctx.contextualize("doc", chunks)
    ctx.close()
    assert out == [""]
