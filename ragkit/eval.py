"""Retrieval eval harness: build a local-synthetic golden set, score a
collection with recall@k + MRR at document granularity, and compare runs.

Pure logic here. `generate_golden` and `evaluate` take injected callables
(`gen_fn` / `search_fn`) so the metric math is unit-tested without touching
Ollama or Qdrant — the CLI wires the real Ollama client and Searcher.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Callable, Iterable, Optional


@dataclass
class GoldenItem:
    query: str
    relevant_uris: list[str]        # doc-level; usually length 1


@dataclass
class EvalReport:
    collection: str
    n_queries: int
    k_values: list[int]
    rerank: bool
    gen_model: Optional[str]
    recall_at_k: dict[int, float]   # k -> mean recall@k
    mrr: float
    per_query: list[dict] = field(default_factory=list)


GenFn = Callable[[str, int], list[str]]     # (doc_text, n) -> questions
SearchFn = Callable[[str, int], list[str]]  # (query, top_k) -> ranked source_uris (dups ok)


def distinct_in_order(uris: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for u in uris:
        if u and u not in seen:
            seen.add(u)
            out.append(u)
    return out


def recall_at_k(retrieved_distinct: list[str], relevant: list[str], k: int) -> float:
    if not relevant:
        return 0.0
    topk = set(retrieved_distinct[:k])
    return len(topk & set(relevant)) / len(relevant)


def first_relevant_rank(retrieved_distinct: list[str], relevant: list[str]) -> Optional[int]:
    rel = set(relevant)
    for i, u in enumerate(retrieved_distinct):
        if u in rel:
            return i + 1
    return None


def generate_golden(docs: Iterable[tuple[str, str]], gen_fn: GenFn, per_doc_n: int,
                    *, min_query_len: int = 15) -> tuple[list[GoldenItem], int]:
    """Return (items, skipped_docs). One GoldenItem per accepted question, relevance =
    the doc it came from. A gen_fn that raises for a doc skips only that doc."""
    items: list[GoldenItem] = []
    skipped = 0
    for uri, text in docs:
        try:
            questions = gen_fn(text, per_doc_n)
        except Exception:
            skipped += 1
            continue
        tail = uri.rsplit("/", 1)[-1].lower()
        seen: set[str] = set()
        for q in questions:
            q = (q or "").strip()
            low = q.lower()
            if len(q) < min_query_len or low in seen:
                continue
            if tail and tail in low:      # drop questions that leak the filename
                continue
            seen.add(low)
            items.append(GoldenItem(query=q, relevant_uris=[uri]))
    return items, skipped


def evaluate(golden: list[GoldenItem], search_fn: SearchFn,
             k_values: list[int]) -> tuple[dict[int, float], float, list[dict], int]:
    """Return (recall_at_k, mrr, per_query, errored). An errored query counts as a 0
    and is surfaced via `errored`, never dropped."""
    maxk = max(k_values)
    recall_sums = {k: 0.0 for k in k_values}
    rr_sum = 0.0
    errored = 0
    per_query: list[dict] = []
    for item in golden:
        try:
            retrieved = distinct_in_order(search_fn(item.query, maxk))
        except Exception:
            retrieved = []
            errored += 1
        rank = first_relevant_rank(retrieved, item.relevant_uris)
        rr_sum += (1.0 / rank) if rank else 0.0
        for k in k_values:
            recall_sums[k] += recall_at_k(retrieved, item.relevant_uris, k)
        per_query.append({
            "query": item.query,
            "relevant_uris": item.relevant_uris,
            "retrieved_uris": retrieved[:maxk],
            "first_rank": rank,
        })
    n = len(golden) or 1
    return {k: recall_sums[k] / n for k in k_values}, rr_sum / n, per_query, errored


def compare(baseline: EvalReport, current: EvalReport) -> dict:
    warnings: list[str] = []
    if baseline.collection != current.collection:
        warnings.append(f"collection differs: {baseline.collection!r} vs {current.collection!r}")
    if baseline.k_values != current.k_values:
        warnings.append(f"k_values differ: {baseline.k_values} vs {current.k_values}")
    if baseline.n_queries != current.n_queries:
        warnings.append(f"n_queries differ: {baseline.n_queries} vs {current.n_queries}")
    ks = [k for k in current.k_values if k in baseline.recall_at_k and k in current.recall_at_k]
    recall = {k: (baseline.recall_at_k[k], current.recall_at_k[k],
                  current.recall_at_k[k] - baseline.recall_at_k[k]) for k in ks}
    mrr = (baseline.mrr, current.mrr, current.mrr - baseline.mrr)
    return {"recall_at_k": recall, "mrr": mrr, "warnings": warnings}


def save_golden(items: list[GoldenItem], path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        for it in items:
            f.write(json.dumps({"query": it.query, "relevant_uris": it.relevant_uris}) + "\n")


def load_golden(path: str) -> list[GoldenItem]:
    items: list[GoldenItem] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            items.append(GoldenItem(query=d["query"], relevant_uris=list(d["relevant_uris"])))
    return items


def save_report(report: EvalReport, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    d = asdict(report)
    d["recall_at_k"] = {str(k): v for k, v in report.recall_at_k.items()}  # JSON keys are strings
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        json.dump(d, f, indent=2)


def load_report(path: str) -> EvalReport:
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    return EvalReport(
        collection=d["collection"], n_queries=d["n_queries"],
        k_values=[int(k) for k in d["k_values"]], rerank=d["rerank"],
        gen_model=d.get("gen_model"),
        recall_at_k={int(k): v for k, v in d["recall_at_k"].items()},
        mrr=d["mrr"], per_query=d.get("per_query", []),
    )
