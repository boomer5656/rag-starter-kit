"""The ingest state machine, SQLite-backed by default.

SQLite is the default so ragkit works from a clean clone with zero infra beyond
Qdrant + Ollama — no Postgres to stand up. Set `state.backend: postgres` in
ragkit.yaml for a shared/multi-host setup (same schema, same API).

The store is append-in-place: every stage transition upserts the row. A document
that errors is *kept* as state='error' with its message, so it stays visible to
`ragkit status` — the pipeline never deletes a source's state on failure.
"""
from __future__ import annotations

import os
import sqlite3
from datetime import datetime, timezone

from .config import StateConfig
from .models import IngestState, State


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class StateStore:
    def __init__(self, cfg: StateConfig, collection: str):
        self.cfg = cfg
        self.collection = collection
        if cfg.backend != "sqlite":
            raise NotImplementedError(
                f"state backend '{cfg.backend}' not wired yet; use 'sqlite' "
                f"(postgres backend is a drop-in with the same schema — see ARCHITECTURE.md)"
            )
        os.makedirs(os.path.dirname(cfg.sqlite_path) or ".", exist_ok=True)
        self._db = sqlite3.connect(cfg.sqlite_path)
        self._db.execute(
            """CREATE TABLE IF NOT EXISTS ingest_state (
                   collection TEXT NOT NULL,
                   uri        TEXT NOT NULL,
                   state      TEXT NOT NULL,
                   dim        INTEGER,
                   n_chunks   INTEGER,
                   error      TEXT,
                   updated_at TEXT NOT NULL,
                   PRIMARY KEY (collection, uri)
               )"""
        )
        self._db.commit()

    def set(self, uri: str, state: State, *, dim: int | None = None,
            n_chunks: int | None = None, error: str | None = None) -> None:
        self._db.execute(
            """INSERT INTO ingest_state (collection, uri, state, dim, n_chunks, error, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT (collection, uri) DO UPDATE SET
                   state=excluded.state, dim=excluded.dim, n_chunks=excluded.n_chunks,
                   error=excluded.error, updated_at=excluded.updated_at""",
            (self.collection, uri, state.value, dim, n_chunks, error, _now()),
        )
        self._db.commit()

    def get(self, uri: str) -> IngestState | None:
        row = self._db.execute(
            "SELECT uri, state, dim, n_chunks, error, updated_at FROM ingest_state "
            "WHERE collection=? AND uri=?", (self.collection, uri)).fetchone()
        if not row:
            return None
        return IngestState(uri=row[0], state=State(row[1]), dim=row[2],
                           n_chunks=row[3], error=row[4], updated_at=row[5])

    def counts(self) -> dict[str, int]:
        rows = self._db.execute(
            "SELECT state, COUNT(*) FROM ingest_state WHERE collection=? GROUP BY state",
            (self.collection,)).fetchall()
        return {state: n for state, n in rows}

    def errors(self, limit: int = 50) -> list[IngestState]:
        rows = self._db.execute(
            "SELECT uri, state, dim, n_chunks, error, updated_at FROM ingest_state "
            "WHERE collection=? AND state=? ORDER BY updated_at DESC LIMIT ?",
            (self.collection, State.ERROR.value, limit)).fetchall()
        return [IngestState(uri=r[0], state=State(r[1]), dim=r[2], n_chunks=r[3],
                            error=r[4], updated_at=r[5]) for r in rows]

    def close(self) -> None:
        self._db.close()
