"""Pure-Python BM25 sparse encoding for hybrid retrieval.

No model, no service. BM25 factorizes so IDF lives on the QUERY vector and the
TF-saturation (with doc-length norm) lives on the DOC vector; their dot product
reconstructs BM25. Stored doc vectors therefore carry no corpus-wide IDF and
don't go stale as the corpus grows (only avgdl drifts — a re-ingest refreshes
it). CorpusStats is the small ingest-time state, persisted as JSON per collection.
Terms hash to u32 indices (Qdrant sparse indices); collisions are rare/tolerable.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass, field

_TOKEN = re.compile(r"[a-z0-9]+")
K1 = 1.5
B = 0.75


def tokenize(text: str) -> list[str]:
    return [t for t in _TOKEN.findall((text or "").lower()) if len(t) >= 2]


def term_index(term: str) -> int:
    """Stable u32 index for a term (blake2b hash; collisions rare and tolerable)."""
    return int(hashlib.blake2b(term.encode("utf-8"), digest_size=4).hexdigest(), 16)


@dataclass
class CorpusStats:
    n_docs: int = 0
    total_len: int = 0
    df: dict[int, int] = field(default_factory=dict)   # term_index -> document frequency

    @property
    def avgdl(self) -> float:
        return self.total_len / self.n_docs if self.n_docs else 0.0

    def add_doc(self, text: str) -> None:
        toks = tokenize(text)
        self.n_docs += 1
        self.total_len += len(toks)
        for ti in {term_index(t) for t in toks}:      # unique per doc -> document frequency
            self.df[ti] = self.df.get(ti, 0) + 1

    def idf(self, ti: int) -> float:
        # BM25 idf, +1 inside log to stay non-negative; unseen term (df 0) -> max idf.
        df = self.df.get(ti, 0)
        return math.log(1 + (self.n_docs - df + 0.5) / (df + 0.5))

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            json.dump({"n_docs": self.n_docs, "total_len": self.total_len,
                       "df": {str(k): v for k, v in self.df.items()}}, f)

    @staticmethod
    def load(path: str) -> "CorpusStats":
        if not os.path.exists(path):
            return CorpusStats()
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        return CorpusStats(n_docs=d["n_docs"], total_len=d["total_len"],
                           df={int(k): v for k, v in d["df"].items()})


def doc_sparse(text: str, stats: CorpusStats, *, k1: float = K1, b: float = B) -> dict[int, float]:
    """Doc-side BM25 vector: TF-saturation + length norm, NO idf (applied at query)."""
    toks = tokenize(text)
    dl = len(toks)
    avgdl = stats.avgdl or dl or 1.0
    tf: dict[int, int] = {}
    for t in toks:
        ti = term_index(t)
        tf[ti] = tf.get(ti, 0) + 1
    denom_norm = k1 * (1 - b + b * dl / avgdl)
    return {ti: f * (k1 + 1) / (f + denom_norm) for ti, f in tf.items()}


def query_sparse(text: str, stats: CorpusStats) -> dict[int, float]:
    """Query-side vector: idf per unique query term."""
    return {ti: stats.idf(ti) for ti in {term_index(t) for t in tokenize(text)}}


def to_qdrant(sparse: dict[int, float]) -> dict:
    """{index: value} -> Qdrant sparse vector {indices, values}."""
    if not sparse:
        return {"indices": [], "values": []}
    idx, val = zip(*sparse.items())
    return {"indices": list(idx), "values": list(val)}


def stats_path(collection: str) -> str:
    """Where a collection's BM25 corpus stats are persisted."""
    return os.path.join(".ragkit", "sparse", f"{collection}.json")
