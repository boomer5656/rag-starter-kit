"""ragkit — self-hosted RAG ingestion pipeline.

Re-exports the classes most callers need (config, orchestrator, and the
lower-level embed/store/state/search primitives) so `import ragkit` covers the
common cases without reaching into submodules.
"""
from __future__ import annotations

from .config import Config
from .embed import Embedder
from .pipeline import Pipeline
from .search import Searcher
from .state import StateStore
from .store import Store

__version__ = "0.0.1"

__all__ = [
    "Config",
    "Pipeline",
    "Embedder",
    "Store",
    "StateStore",
    "Searcher",
    "__version__",
]
