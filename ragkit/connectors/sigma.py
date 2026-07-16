"""Sigma detection-rule connector — adapted from homelab-ai's ingest_sigma.py.

Walks a Sigma `rules/` tree and yields one SourceDoc per rule, with the same
semantic blob (title, description, logsource, level, MITRE techniques, tags,
detection fields) that the standalone script embedded directly. Since the
blob is already text, it's set as `SourceDoc.text` so the pipeline's extract
stage is a no-op for these docs (Tika is skipped entirely).
"""
from __future__ import annotations

import glob
import os
import re
from typing import Iterable, Iterator

import yaml

from ..models import SourceDoc, stable_id

_TECH_RE = re.compile(r"^attack\.t(\d+(?:\.\d+)?)$", re.I)


def _logsource_str(ls) -> str:
    if not isinstance(ls, dict):
        return ""
    return " ".join(f"{k}={v}" for k, v in ls.items() if v)


def _rule_to_blob(rule: dict) -> tuple[str, list[str], str]:
    """Return (embeddable_text, mitre_techniques, logsource_string)."""
    title = str(rule.get("title", "")).strip()
    desc = str(rule.get("description", "")).strip()
    ls = _logsource_str(rule.get("logsource", {}))
    level = str(rule.get("level", "")).strip()
    tags = [str(t) for t in (rule.get("tags") or [])]
    mitre = sorted({("T" + m.group(1).upper()) for t in tags if (m := _TECH_RE.match(t))})
    det = rule.get("detection", {})
    det_fields = ", ".join(k for k in det.keys() if k != "condition") if isinstance(det, dict) else ""

    blob = (
        f"{title}\n\n{desc}\n\n"
        f"Log source: {ls}\n"
        f"Level: {level}\n"
        f"MITRE ATT&CK: {', '.join(mitre)}\n"
        f"Tags: {', '.join(tags)}\n"
        f"Detection fields: {det_fields}"
    )
    return blob, mitre, ls


class SigmaConnector:
    source_type = "detection"
    index_fields = {"mitre": "keyword", "level": "keyword", "logsource": "keyword", "status": "keyword"}

    def __init__(self, rules_dir: str):
        self.rules_dir = rules_dir

    def iter_docs(self) -> Iterable[SourceDoc]:
        return self._walk()

    def _walk(self) -> Iterator[SourceDoc]:
        files = sorted(glob.glob(os.path.join(self.rules_dir, "**", "*.yml"), recursive=True))
        for path in files:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    docs = list(yaml.safe_load_all(f))
            except Exception:
                continue  # unparseable file — surfaced as a discovery gap, not a pipeline error
            for rule in docs:
                if not isinstance(rule, dict) or "title" not in rule:
                    continue
                blob, mitre, ls = _rule_to_blob(rule)
                sid = rule.get("id")
                repo_path = os.path.relpath(path, self.rules_dir).replace("\\", "/")
                uri = str(sid) if isinstance(sid, str) and sid else str(stable_id(repo_path))
                yield SourceDoc(
                    uri=uri,
                    title=rule.get("title", "") or "",
                    mime="text/plain",
                    text=blob,
                    meta={
                        "mitre": mitre,
                        "level": rule.get("level", "") or "",
                        "logsource": ls,
                        "status": rule.get("status", "") or "",
                        "tags": [str(t) for t in (rule.get("tags") or [])],
                        "repo_path": repo_path,
                    },
                )
