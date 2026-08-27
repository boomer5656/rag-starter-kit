"""Token-aware chunking with overlap, dependency-free.

No tokenizer is bundled (keeps the dependency list at stdlib + httpx + pyyaml),
so token counts are *approximated* from whitespace-split word counts. The common
rule of thumb for English prose + BPE tokenizers is ~1 token ≈ 0.75 words
(equivalently ~1.3 tokens per word). So a budget of `max_tokens` tokens maps to a
word budget of `max_tokens * 0.75`; `overlap_tokens` converts the same way. It's
an approximation, not exact BPE — but it keeps chunks at (not over) the target
token size so they stay within the embedder's context window and retrieval
granularity, without pulling in tiktoken/transformers.
"""
from __future__ import annotations

from .config import ChunkConfig
from .models import Chunk

# Heuristic: 1 token ~= 0.75 whitespace-split words (English prose + BPE average).
_WORDS_PER_TOKEN = 0.75

# Character safety cap per chunk. The words-per-token heuristic UNDER-counts tokens for
# long tokens (dense code, IDs, base64 / minified runs, table cells with no spaces), so a
# purely word-bounded chunk can balloon in characters and blow past the embedder's context
# window — Ollama then 500s and the pipeline drops the WHOLE document. This bound is generous
# for normal prose (a ~512-token prose chunk is ~2.7k chars, well under it, so word-budgeting
# still governs those) and only caps pathological, char-dense chunks down to an embeddable
# size. Verified safe against bge-m3 (5k chars embeds; ~20k 500s). Fixes the large-doc/code
# ingest failures observed 2026-08-26.
_MAX_CHUNK_CHARS = 5000


def _split_long_words(words: list[str], max_chars: int) -> list[str]:
    """Break any single 'word' longer than max_chars into max_chars pieces, so one giant
    whitespace-free token (a base64 blob, a minified line) can't form an oversized chunk."""
    out: list[str] = []
    for w in words:
        if len(w) <= max_chars:
            out.append(w)
        else:
            out.extend(w[k:k + max_chars] for k in range(0, len(w), max_chars))
    return out


class TokenChunker:
    """Chunker: splits text into overlapping, token-budgeted windows (char-capped)."""

    def __init__(self, chunk_cfg: ChunkConfig):
        self.cfg = chunk_cfg

    def chunk(self, source_uri: str, text: str, meta: dict) -> list[Chunk]:
        # str.split() with no args collapses all whitespace runs (including
        # newlines/tabs) and drops leading/trailing whitespace for free.
        words = text.split()
        if not words:
            return []

        window = max(1, round(self.cfg.max_tokens * _WORDS_PER_TOKEN))
        overlap = max(0, round(self.cfg.overlap_tokens * _WORDS_PER_TOKEN))
        overlap = min(overlap, window - 1)  # never let overlap swallow the whole window
        words = _split_long_words(words, _MAX_CHUNK_CHARS)

        chunks: list[Chunk] = []
        ordinal = 0
        i = 0
        n = len(words)
        while i < n:
            # accumulate words up to BOTH the word window and the char cap (the first word
            # is always taken, so a single max-chars word still forms one valid chunk)
            acc: list[str] = []
            clen = 0
            j = i
            while j < n and len(acc) < window:
                add = len(words[j]) + (1 if acc else 0)
                if acc and clen + add > _MAX_CHUNK_CHARS:
                    break
                acc.append(words[j])
                clen += add
                j += 1
            chunk_text = " ".join(acc).strip()
            if chunk_text:
                chunks.append(
                    Chunk(source_uri=source_uri, ordinal=ordinal, text=chunk_text, meta=dict(meta))
                )
                ordinal += 1
            if j >= n:
                break
            # advance by the words ACTUALLY consumed (minus overlap) so char-capped windows
            # don't skip content — a word-fixed step would leave gaps in dense text
            i += max(1, (j - i) - overlap)

        return chunks
