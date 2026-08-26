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
