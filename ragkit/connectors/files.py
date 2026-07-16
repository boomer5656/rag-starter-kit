"""The headline connector — a local (or mounted) directory of mixed documents.

Walks `root` recursively and yields one SourceDoc per file whose extension is
in the allowed set. Most formats are left with `text=None`: Tika does the
extraction later, in the pipeline's extract stage, where it can stay warm
across the whole batch. Plain-text formats (.txt/.md) are cheap enough to
read inline here, which also lets them skip Tika entirely.
"""
from __future__ import annotations

import mimetypes
import os
from typing import Iterable, Iterator

from ..models import SourceDoc

DEFAULT_EXTS = {
    "pdf", "docx", "doc", "pptx", "xlsx", "txt", "md", "html", "htm", "csv", "rtf", "odt",
}

# Extensions read inline (skip Tika) — kept in sync with DEFAULT_EXTS' text formats.
_INLINE_TEXT_EXTS = {"txt", "md"}


class FilesConnector:
    source_type = "document"
    index_fields = {"source_uri": "keyword", "ext": "keyword"}

    def __init__(self, root: str, exts: set[str] | None = None):
        self.root = root
        self.exts = {e.lower().lstrip(".") for e in (exts or DEFAULT_EXTS)}

    def iter_docs(self) -> Iterable[SourceDoc]:
        return self._walk()

    def _walk(self) -> Iterator[SourceDoc]:
        root = os.path.abspath(self.root)
        for dirpath, _dirnames, filenames in os.walk(root):
            for name in filenames:
                ext = os.path.splitext(name)[1].lower().lstrip(".")
                if ext not in self.exts:
                    continue
                path = os.path.join(dirpath, name)
                uri = os.path.abspath(path)
                rel_path = os.path.relpath(uri, root).replace("\\", "/")
                mime = mimetypes.guess_type(name)[0] or "application/octet-stream"

                text = None
                if ext in _INLINE_TEXT_EXTS:
                    try:
                        with open(uri, "r", encoding="utf-8", errors="replace") as f:
                            text = f.read()
                    except OSError:
                        text = None  # let it fall through to Tika/extract-stage error handling

                yield SourceDoc(
                    uri=uri,
                    title=name,
                    mime=mime,
                    text=text,
                    meta={"path": rel_path, "ext": ext},
                )
