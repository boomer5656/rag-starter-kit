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


class TokenChunker:
    """Chunker: splits text into overlapping, token-budgeted windows."""

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
        step = max(1, window - overlap)

        chunks: list[Chunk] = []
        ordinal = 0
        for start in range(0, len(words), step):
            piece = words[start:start + window]
            if not piece:
                continue
            chunk_text = " ".join(piece).strip()
            if not chunk_text:
                continue
            chunks.append(
                Chunk(source_uri=source_uri, ordinal=ordinal, text=chunk_text, meta=dict(meta))
            )
            ordinal += 1
            if start + window >= len(words):
                break

        return chunks
