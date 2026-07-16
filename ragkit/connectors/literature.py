"""Literature (paper metadata) connector — a self-contained sibling of the
knowledge-ingest-pipeline's wf2_process.sh.

Unlike that script, this connector does not call OpenAlex, run a relevance
gate, or summarize with a live model — it is purely a data source. It turns
paper records (already fetched elsewhere) into SourceDocs; gating, chunking,
and embedding are the pipeline's job.
"""
from __future__ import annotations

import json
from typing import Iterable, Iterator, Optional

from ..models import SourceDoc


class LiteratureConnector:
    source_type = "literature"
    index_fields = {"doi": "keyword", "year": "integer"}

    def __init__(self, papers: Optional[list[dict]] = None, jsonl_path: Optional[str] = None):
        if papers is None and jsonl_path is None:
            raise ValueError("LiteratureConnector needs either `papers` or `jsonl_path`")
        self.papers = papers
        self.jsonl_path = jsonl_path

    def iter_docs(self) -> Iterable[SourceDoc]:
        return self._iter()

    def _load(self) -> Iterator[dict]:
        if self.papers is not None:
            yield from self.papers
            return
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)

    def _iter(self) -> Iterator[SourceDoc]:
        for paper in self._load():
            doi = paper.get("doi")
            if not doi:
                continue  # doi is the uri; skip records that can't be identified
            title = paper.get("title", "") or ""
            abstract = paper.get("abstract", "") or ""
            year = paper.get("year")
            yield SourceDoc(
                uri=str(doi),
                title=title,
                mime="text/plain",
                text=f"Title: {title}. Abstract: {abstract}",
                meta={"doi": doi, "year": year, "source_type": "literature"},
            )
