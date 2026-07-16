"""Shared pytest fixtures for ragkit's unit test suite.

Pure-unit only: nothing here (or in any test using these fixtures) is allowed
to touch the network — no Ollama, no Qdrant, no Tika. State is always backed
by a throwaway SQLite file under pytest's tmp_path.
"""
from __future__ import annotations

import pytest

from ragkit.config import (
    ChunkConfig,
    Config,
    OllamaConfig,
    QdrantConfig,
    RerankerConfig,
    StateConfig,
    TikaConfig,
)


@pytest.fixture
def tmp_config(tmp_path) -> Config:
    """A fully-formed Config whose sqlite state file lives under tmp_path.

    Every field is a plain default except `state.sqlite_path`, which is
    redirected into pytest's isolated tmp_path so tests never touch a real
    `.ragkit/state.db` or leak state between test runs.
    """
    return Config(
        collection="test-collection",
        ollama=OllamaConfig(),
        qdrant=QdrantConfig(),
        tika=TikaConfig(),
        reranker=RerankerConfig(),
        state=StateConfig(backend="sqlite", sqlite_path=str(tmp_path / "state.db")),
        chunk=ChunkConfig(),
        gate_enabled=False,
    )
