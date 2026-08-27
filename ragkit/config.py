"""Load and validate ragkit.yaml, with environment-variable overrides.

The shipped `ragkit.example.yaml` uses the homelab tower defaults as a concrete
example (192.168.1.100 + the standard ports). Nothing here is hardcoded — every
value comes from the yaml or an env override, so a buyer points it at their own
Ollama/Qdrant by editing one file.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Optional

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None


@dataclass
class OllamaConfig:
    url: str = "http://localhost:11434"
    # Tags track ~/dev/homelab-plugin/model-registry.json (roles `embed_text` / `triage`).
    # gate_model was `llama3.2:3b` until 2026-07-30; that model was deleted from the fleet
    # on 2026-07-29 as redundant with qwen3:1.7b, so the old default no longer resolves.
    embed_model: str = "bge-m3"
    gate_model: str = "qwen3:1.7b"


@dataclass
class QdrantConfig:
    url: str = "http://localhost:6333"
    distance: str = "Cosine"


@dataclass
class TikaConfig:
    url: str = "http://localhost:9998"
    allowed_root: str = "."          # file reads confined here; overridden per ingest


@dataclass
class RerankerConfig:
    url: Optional[str] = None        # None => search skips reranking


@dataclass
class StateConfig:
    backend: str = "sqlite"          # "sqlite" | "postgres"
    sqlite_path: str = ".ragkit/state.db"
    postgres_dsn: Optional[str] = None


@dataclass
class ChunkConfig:
    max_tokens: int = 512
    overlap_tokens: int = 64


@dataclass
class EvalConfig:
    golden_path: str = ".ragkit/eval/{collection}.golden.jsonl"   # {collection} templated at use
    gen_model: Optional[str] = None    # None -> falls back to ollama.gate_model
    per_doc_n: int = 3
    k_values: list[int] = field(default_factory=lambda: [1, 3, 5, 10])


@dataclass
class ContextualConfig:
    enabled: bool = False
    model: Optional[str] = None          # None -> ollama.gate_model (use a CAPABLE model)
    max_chunks_per_call: int = 10


@dataclass
class MultiQueryConfig:
    model: Optional[str] = None    # None -> ollama.gate_model; use a CAPABLE instruct model
    n: int = 3                     # default paraphrase count for `--multi` with no number


@dataclass
class Config:
    collection: str = "ragkit"
    ollama: OllamaConfig = field(default_factory=OllamaConfig)
    qdrant: QdrantConfig = field(default_factory=QdrantConfig)
    tika: TikaConfig = field(default_factory=TikaConfig)
    reranker: RerankerConfig = field(default_factory=RerankerConfig)
    state: StateConfig = field(default_factory=StateConfig)
    chunk: ChunkConfig = field(default_factory=ChunkConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    contextual: ContextualConfig = field(default_factory=ContextualConfig)
    multi_query: MultiQueryConfig = field(default_factory=MultiQueryConfig)
    gate_enabled: bool = False       # opt-in per run

    @staticmethod
    def load(path: str = "ragkit.yaml") -> "Config":
        data: dict[str, Any] = {}
        if os.path.exists(path):
            if yaml is None:
                raise RuntimeError("pyyaml is required to read ragkit.yaml (pip install pyyaml)")
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
        cfg = Config(
            collection=data.get("collection", "ragkit"),
            ollama=OllamaConfig(**(data.get("ollama") or {})),
            qdrant=QdrantConfig(**(data.get("qdrant") or {})),
            tika=TikaConfig(**(data.get("tika") or {})),
            reranker=RerankerConfig(**(data.get("reranker") or {})),
            state=StateConfig(**(data.get("state") or {})),
            chunk=ChunkConfig(**(data.get("chunk") or {})),
            eval=EvalConfig(**(data.get("eval") or {})),
            contextual=ContextualConfig(**(data.get("contextual") or {})),
            multi_query=MultiQueryConfig(**(data.get("multi_query") or {})),
            gate_enabled=bool(data.get("gate_enabled", False)),
        )
        cfg._apply_env()
        return cfg

    def _apply_env(self) -> None:
        """RAGKIT_* env vars override yaml — handy for Docker/CI without editing files."""
        self.ollama.url = os.environ.get("RAGKIT_OLLAMA_URL", self.ollama.url)
        self.ollama.embed_model = os.environ.get("RAGKIT_EMBED_MODEL", self.ollama.embed_model)
        self.qdrant.url = os.environ.get("RAGKIT_QDRANT_URL", self.qdrant.url)
        self.tika.url = os.environ.get("RAGKIT_TIKA_URL", self.tika.url)
        if os.environ.get("RAGKIT_RERANKER_URL"):
            self.reranker.url = os.environ["RAGKIT_RERANKER_URL"]
        if os.environ.get("RAGKIT_COLLECTION"):
            self.collection = os.environ["RAGKIT_COLLECTION"]
