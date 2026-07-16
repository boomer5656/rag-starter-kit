"""Text extraction via Tika, with local fast-paths for plain text.

.txt/.md never touch Tika — they're decoded directly, which keeps ingest of a
docs folder fast and Tika-outage-proof. Everything else goes to the tower's
Tika sidecar as raw bytes. When a SourceDoc points at a local file path (no
raw_bytes in hand yet), reads are confined to `allowed_root` — the same
path-traversal defense as homelab-ai's tika_extract (H-HA-1): reject '..',
resolve with os.path.realpath, and verify containment with os.path.commonpath.
A SourceDoc that already carries raw_bytes (e.g. from a connector that fetched
a remote resource) skips confinement entirely — the bytes are already in hand,
there's no filesystem read to confine.
"""
from __future__ import annotations

import logging
import os

import httpx

from .config import TikaConfig
from .models import SourceDoc

log = logging.getLogger(__name__)

# Extensions read directly, no Tika round-trip.
_PLAIN_TEXT_EXTS = {".txt", ".md"}
_PLAIN_TEXT_MIMES = {"text/plain", "text/markdown"}

# Extensions Tika is expected to handle; anything else is rejected outright
# when we're about to read a local file path.
_TIKA_ALLOWED_EXTS = {
    ".pdf", ".docx", ".doc", ".pptx", ".ppt", ".xlsx", ".xls",
    ".txt", ".md", ".html", ".htm", ".csv", ".rtf", ".odt", ".epub",
}

# Hard cap on extracted text length; documents beyond this are truncated
# rather than allowed to blow up downstream chunking/embedding.
_MAX_CHARS = 200_000


class TikaExtractor:
    """Extractor: turns a SourceDoc's bytes (or on-disk file) into plain text."""

    def __init__(self, tika_cfg: TikaConfig, allowed_root: str, timeout: float = 120.0):
        self.cfg = tika_cfg
        self.allowed_root = os.path.realpath(allowed_root)
        self._client = httpx.Client(timeout=timeout)

    def extract(self, doc: SourceDoc) -> str:
        if doc.text:
            return doc.text

        ext = os.path.splitext(doc.uri)[1].lower()
        is_plain_text = ext in _PLAIN_TEXT_EXTS or doc.mime in _PLAIN_TEXT_MIMES

        if doc.raw_bytes is not None:
            raw = doc.raw_bytes
        else:
            # No bytes in hand -> doc.uri must be a local file path. Confine the read.
            path = self._resolve_confined(doc.uri)
            with open(path, "rb") as f:
                raw = f.read()

        if is_plain_text:
            text = raw.decode("utf-8", errors="replace")
        else:
            text = self._tika_extract(raw)

        return self._truncate(text, doc.uri)

    def _resolve_confined(self, uri: str) -> str:
        """Resolve `uri` as a local file path confined to self.allowed_root.

        Mirrors homelab-ai's tika_extract confinement: reject '..' segments,
        canonicalize with realpath (follows symlinks), then verify the result
        is inside allowed_root via commonpath. Also rejects extensions Tika
        doesn't handle.
        """
        if ".." in uri.replace("\\", "/").split("/"):
            raise PermissionError(f"path traversal ('..') is not allowed: {uri!r}")

        candidate = uri if os.path.isabs(uri) else os.path.join(self.allowed_root, uri)
        real_path = os.path.realpath(candidate)
        try:
            inside = os.path.commonpath([real_path, self.allowed_root]) == self.allowed_root
        except ValueError:
            # Different drives on Windows, etc. -> definitely not inside.
            inside = False
        if not inside:
            raise PermissionError(
                f"path is outside the allowed root and will not be read: "
                f"requested={real_path!r} allowed_root={self.allowed_root!r}"
            )

        ext = os.path.splitext(real_path)[1].lower()
        if ext not in _TIKA_ALLOWED_EXTS:
            raise ValueError(
                f"extension {ext!r} is not an allowed document type "
                f"(allowed: {', '.join(sorted(_TIKA_ALLOWED_EXTS))})"
            )

        if not os.path.isfile(real_path):
            raise FileNotFoundError(f"not a readable file: {real_path}")

        return real_path

    def _tika_extract(self, raw: bytes) -> str:
        r = self._client.put(
            f"{self.cfg.url}/tika",
            content=raw,
            headers={"Accept": "text/plain"},
        )
        r.raise_for_status()
        return r.text

    def _truncate(self, text: str, uri: str) -> str:
        if len(text) > _MAX_CHARS:
            log.warning(
                "extraction for %s truncated: %d chars -> %d cap", uri, len(text), _MAX_CHARS
            )
            return text[:_MAX_CHARS]
        return text

    def close(self) -> None:
        self._client.close()
