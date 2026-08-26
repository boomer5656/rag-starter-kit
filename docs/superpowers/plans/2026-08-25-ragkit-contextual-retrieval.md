# ragkit contextual retrieval — Implementation Plan

> **For agentic workers:** Implement task-by-task, TDD. Stay on branch `ragkit-contextual-retrieval`. Do NOT merge/push/deploy. Steps use `- [ ]`.

**Goal:** Add opt-in contextual retrieval — an LLM-written situating sentence prepended to each chunk before embedding — on the local Ollama, plus a shared Ollama helper that all three JSON-generate callers use.

**Architecture:** New `ragkit/ollama.py` (shared `generate_json`) and `ragkit/contextualize.py` (batch-per-doc Contextualizer). `gate.py` and `cli._make_gen_fn` migrate onto the helper. `pipeline.py` embeds `context+chunk` when a contextualizer is injected but still stores the raw chunk. All gated by `contextual.enabled` (default off).

**Tech Stack:** Python 3, stdlib + httpx. pytest. No new deps.

## Global Constraints (verbatim from spec)
- stdlib + httpx only; plain dataclasses; no ORM/pydantic in core.
- Config single-source; fail-loud (per-item error → state='error', run never aborts).
- Embed `context+chunk`; **store the raw chunk** (no payload-schema change).
- `contextual.enabled: false` (default) ⇒ ingest behavior byte-for-byte unchanged.
- Tests network-free (mock/inject the Ollama call); LF endings.
- `think:false` + `num_predict` cap live in the shared helper, one place.

---

### Task 1: shared `ragkit/ollama.py` helper

**Files:** Create `ragkit/ollama.py`; Test `tests/test_ollama.py`.
**Interfaces — Produces:** `generate_json(client, url, model, *, system, prompt, num_predict=1024, temperature=0.0) -> Any`.

- [ ] **Step 1: test** `tests/test_ollama.py`

```python
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
```

- [ ] **Step 2: run → fails** (`ModuleNotFoundError: ragkit.ollama`).
- [ ] **Step 3: implement** `ragkit/ollama.py`

```python
"""Single home for ragkit's Ollama /api/generate + format=json contract.

Three callers share this shape (the gate, the eval golden-set generator, the
contextualizer). think=False is load-bearing: Qwen3 models otherwise spend
their whole budget in the thinking channel under format="json" and return ""
or "{}". num_predict caps output per the usage-limits doctrine.
"""
from __future__ import annotations

import json
from typing import Any


def generate_json(client: Any, url: str, model: str, *, system: str, prompt: str,
                  num_predict: int = 1024, temperature: float = 0.0) -> Any:
    """POST /api/generate (format=json, think=False, capped) and return parsed JSON.
    Raises on HTTP error or an empty/unparseable response — callers isolate per item."""
    r = client.post(f"{url}/api/generate", json={
        "model": model, "system": system, "prompt": prompt,
        "stream": False, "format": "json", "think": False,
        "options": {"temperature": temperature, "num_predict": num_predict},
    })
    r.raise_for_status()
    raw = r.json().get("response")
    if not raw:
        raise RuntimeError(f"ollama: empty response from {model}")
    return json.loads(raw)
```

- [ ] **Step 4: run → pass.**
- [ ] **Step 5: commit** — `git add ragkit/ollama.py tests/test_ollama.py && git commit -m "feat(ollama): shared generate_json helper (think:false + cap)"`

---

### Task 2: migrate `gate.py` and `cli._make_gen_fn` onto the helper

**Files:** Modify `ragkit/gate.py`, `ragkit/cli.py`. Tests: existing `tests/test_cli.py` must still pass (its FakeClient asserts the same payload the helper builds).

**Interfaces — Consumes:** `ragkit.ollama.generate_json`.

- [ ] **Step 1: edit `gate.py`** — replace the inline post/parse in `keep()`. Add `from .ollama import generate_json` at the top (keep the `import json`? it's no longer used → remove it). New `keep`:

```python
    def keep(self, doc: SourceDoc) -> tuple[bool, str]:
        text = (doc.text or "")[:_TEXT_CHARS]
        system = _SYSTEM_TEMPLATE.format(criteria=self.criteria)
        prompt = f"Title: {doc.title}\n\n{text}\n\nReturn JSON: {{\"keep\":true|false,\"reason\":\"<=10 words\"}}"
        inner = generate_json(self._client, self.cfg.url, self.cfg.gate_model,
                              system=system, prompt=prompt)
        if not isinstance(inner, dict) or "keep" not in inner or not isinstance(inner["keep"], bool):
            raise RuntimeError(f"gate: missing/invalid 'keep' field: {inner!r}")
        return bool(inner["keep"]), str(inner.get("reason", ""))
```

Remove the now-unused `import json` line from gate.py (the helper owns json). Leave `import httpx` (still used for the client in `__init__`).

- [ ] **Step 2: edit `cli.py _make_gen_fn`** — route through the helper. Replace the body of the inner `gen`:

```python
    def gen(doc_text: str, n: int) -> list[str]:
        prompt = (f"Write {n} distinct questions a user would ask that this document answers.\n\n"
                  f"Document:\n{doc_text}\n\nReturn JSON: {{\"questions\": [...]}}")
        data = generate_json(client, ollama_url, model, system=system, prompt=prompt, temperature=0.2)
        qs = data.get("questions") if isinstance(data, dict) else data
        if not isinstance(qs, list):
            return []
        return [str(q) for q in qs][:n]
```

Add `from .ollama import generate_json` to cli.py imports. The old inline `import json as _json`, `r = client.post(...)`, `raise_for_status`, empty-check are all removed (the helper does them).

- [ ] **Step 3: run tests** — `pytest tests/test_cli.py tests/test_ollama.py -q`. The existing `test_make_gen_fn_*` still pass: the helper builds the same body (`think:False`, `format:json`, `num_predict`, `model`) the FakeClient records. Expected: PASS.
- [ ] **Step 4: full suite** — `pytest -q` green.
- [ ] **Step 5: commit** — `git add ragkit/gate.py ragkit/cli.py && git commit -m "refactor: gate + eval-gen use shared generate_json (carries think:false to gate)"`

---

### Task 3: `ragkit/contextualize.py`

**Files:** Create `ragkit/contextualize.py`; Test `tests/test_contextualize.py`.
**Interfaces — Produces:** `Contextualizer(ollama_cfg, model, max_chunks_per_call=10)`, `.contextualize(doc_text, chunks) -> list[str]` (one per chunk, aligned), `.close()`.

- [ ] **Step 1: test** `tests/test_contextualize.py` (monkeypatch the helper — network-free)

```python
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
```

- [ ] **Step 2: run → fails.**
- [ ] **Step 3: implement** `ragkit/contextualize.py`

```python
"""Contextual retrieval: prepend an LLM-written situating sentence to each chunk
before embedding (Anthropic's contextual-embeddings technique), on the local
Ollama. Batch-per-doc: one call per <=max_chunks_per_call chunks — Ollama has no
cross-request prompt cache, so re-sending the doc per chunk would be wasteful.

Fail-loud: a generation failure propagates to the pipeline, which marks the doc
state='error' (visible in `ragkit status`) rather than silently embedding raw.
"""
from __future__ import annotations

import httpx

from .config import OllamaConfig
from .models import Chunk
from .ollama import generate_json

_DOC_CHARS = 6000
_SYSTEM = (
    "You situate document chunks for a retrieval system. For each numbered chunk, "
    "write ONE short sentence (<=25 words) giving the context needed to understand it "
    "in isolation — what document/section/entity/period it concerns. Output ONLY JSON: "
    "{\"contexts\": [\"...\"]} with exactly one entry per chunk, in the given order."
)


class Contextualizer:
    def __init__(self, ollama_cfg: OllamaConfig, model: str,
                 max_chunks_per_call: int = 10, timeout: float = 120.0):
        self.cfg = ollama_cfg
        self.model = model
        self.max_per_call = max(1, max_chunks_per_call)
        self._client = httpx.Client(timeout=timeout)

    def contextualize(self, doc_text: str, chunks: list[Chunk]) -> list[str]:
        """One context string per chunk, aligned by index. Empty string = no context
        for that chunk (it embeds raw). Raises if the model call fails."""
        doc = (doc_text or "")[:_DOC_CHARS]
        out: list[str] = []
        for start in range(0, len(chunks), self.max_per_call):
            out.extend(self._one_call(doc, chunks[start:start + self.max_per_call]))
        if len(out) < len(chunks):
            out.extend([""] * (len(chunks) - len(out)))
        return out[:len(chunks)]

    def _one_call(self, doc: str, batch: list[Chunk]) -> list[str]:
        numbered = "\n\n".join(f"[chunk {i}]\n{c.text}" for i, c in enumerate(batch))
        prompt = (f"Document:\n{doc}\n\nChunks to situate:\n{numbered}\n\n"
                  f"Return JSON: {{\"contexts\": [...]}} with exactly {len(batch)} entries, in order.")
        data = generate_json(self._client, self.cfg.url, self.model, system=_SYSTEM, prompt=prompt)
        ctxs = data.get("contexts") if isinstance(data, dict) else data
        if not isinstance(ctxs, list):
            ctxs = []
        ctxs = [str(c) for c in ctxs][:len(batch)]
        if len(ctxs) < len(batch):
            ctxs.extend([""] * (len(batch) - len(ctxs)))
        return ctxs

    def close(self) -> None:
        self._client.close()
```

- [ ] **Step 4: run → pass.**
- [ ] **Step 5: commit** — `git add ragkit/contextualize.py tests/test_contextualize.py && git commit -m "feat(contextualize): batch-per-doc local contextual retrieval"`

---

### Task 4: `ContextualConfig` in `config.py` + example.yaml

**Files:** Modify `ragkit/config.py`, `ragkit.example.yaml`.
**Interfaces — Produces:** `ContextualConfig(enabled, model, max_chunks_per_call)`; `Config.contextual`.

- [ ] **Step 1: add dataclass** after `EvalConfig` (before `class Config`):

```python
@dataclass
class ContextualConfig:
    enabled: bool = False
    model: Optional[str] = None          # None -> ollama.gate_model (use a CAPABLE model)
    max_chunks_per_call: int = 10
```

- [ ] **Step 2: field on `Config`** — after `eval: EvalConfig = ...`, before `gate_enabled`:

```python
    contextual: ContextualConfig = field(default_factory=ContextualConfig)
```

- [ ] **Step 3: wire `Config.load`** — after the `eval=` line in the constructor:

```python
            contextual=ContextualConfig(**(data.get("contextual") or {})),
```

- [ ] **Step 4: document in `ragkit.example.yaml`** (append):

```yaml

# Contextual retrieval (opt-in). Prepends an LLM-written situating sentence to
# each chunk before embedding. Adds one Ollama call per document at ingest.
contextual:
  enabled: false
  # model: null                # null -> ollama.gate_model. Use a CAPABLE instruct
  #                            # model (>=~9B); a tiny model writes weak context.
  max_chunks_per_call: 10
```

- [ ] **Step 5: verify** — `python -c "from ragkit.config import Config; print(Config().contextual.enabled)"` → `False`; `pytest -q` green.
- [ ] **Step 6: commit** — `git add ragkit/config.py ragkit.example.yaml && git commit -m "feat(config): ContextualConfig block"`

---

### Task 5: wire into `pipeline.py` + `cli.cmd_ingest`

**Files:** Modify `ragkit/pipeline.py`, `ragkit/cli.py`.
**Interfaces — Consumes:** `Contextualizer` (Task 3), `Config.contextual` (Task 4).

- [ ] **Step 1: pipeline Protocol + param.** In `pipeline.py`, add after the `Gate` Protocol:

```python
class ContextualizerLike(Protocol):
    def contextualize(self, doc_text: str, chunks: list[Chunk]) -> list[str]: ...
```

Add `contextualizer: ContextualizerLike | None = None` to `Pipeline.__init__` signature (after `gate`), and `self.contextualizer = contextualizer` in the body.

- [ ] **Step 2: apply in the embed loop.** In `run()`, replace the first line of the embed+store `try` (currently `vectors = self.embedder.embed_batch([c.text for c in chunks])`) with:

```python
                texts = [c.text for c in chunks]
                if self.contextualizer is not None:
                    ctxs = self.contextualizer.contextualize(d.text or "", chunks)
                    texts = [f"{ctx}\n{c.text}" if ctx else c.text
                             for ctx, c in zip(ctxs, chunks)]
                vectors = self.embedder.embed_batch(texts)
```

(This stays inside the existing `try`, so a contextualize failure hits `_err(res, d.uri, ...)` — fail-loud. `upsert(chunks, ...)` below is unchanged: raw chunk stored.)

- [ ] **Step 3: build it in `cmd_ingest`.** In `cli.py cmd_ingest`, after the `gate = ...` line add:

```python
    contextualizer = None
    if cfg.contextual.enabled:
        from .contextualize import Contextualizer
        ctx_model = cfg.contextual.model or cfg.ollama.gate_model
        contextualizer = Contextualizer(cfg.ollama, ctx_model, cfg.contextual.max_chunks_per_call)
```

Pass it to the `Pipeline(...)` constructor (add `contextualizer=contextualizer,`), and in the `finally` add:

```python
        if contextualizer is not None:
            contextualizer.close()
```

- [ ] **Step 4: verify** — `python -m ragkit.cli ingest --help` exits 0; `python -c "import ragkit.pipeline, ragkit.cli"` OK; `pytest -q` green (default off ⇒ existing behavior unchanged).
- [ ] **Step 5: commit** — `git add ragkit/pipeline.py ragkit/cli.py && git commit -m "feat(pipeline): apply contextual retrieval when enabled"`

---

## Acceptance (verify at end)
1. `pytest -q` green (existing + test_ollama + test_contextualize), network-free.
2. `python -c "import ragkit.ollama, ragkit.contextualize, ragkit.pipeline, ragkit.cli"` OK.
3. `contextual.enabled` default False; help commands exit 0.
4. gate.py, _make_gen_fn, contextualize.py all import from ragkit.ollama.
5. All work committed on branch `ragkit-contextual-retrieval`; master untouched; nothing pushed.

## Out of scope
- Hybrid retrieval (sub-project 3). The `literature`/`serve-mcp --config` bugs (tracked separately).
