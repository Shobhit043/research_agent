"""Persistence for sessions, conversation history, turn results and indexed chunks.

Two backends share every query: SQLite (default; zero setup, one file) and PostgreSQL
(`DATABASE_URL=postgresql://...`; pooled connections, for multi-instance deployments).
Queries are written once with `?` placeholders and `{json}` markers for JSON values;
each backend translates them to its own dialect.
"""

import json
import logging
import sqlite3
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage, messages_from_dict, messages_to_dict

logger = logging.getLogger(__name__)

_SQLITE_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id          TEXT PRIMARY KEY,
    owner       TEXT NOT NULL,
    created_at  REAL NOT NULL,
    updated_at  REAL NOT NULL,
    messages    TEXT NOT NULL DEFAULT '[]',
    tokens_used INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS turns (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    question    TEXT NOT NULL,
    result      TEXT NOT NULL,
    attachments TEXT NOT NULL DEFAULT '[]',
    created_at  REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS chunks (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    source     TEXT NOT NULL,
    metadata   TEXT NOT NULL,
    content    TEXT NOT NULL,
    embedding  BLOB
);
CREATE INDEX IF NOT EXISTS turns_by_session ON turns(session_id, id);
CREATE INDEX IF NOT EXISTS chunks_by_source ON chunks(session_id, source);
"""

_POSTGRES_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id          TEXT PRIMARY KEY,
    owner       TEXT NOT NULL,
    created_at  DOUBLE PRECISION NOT NULL,
    updated_at  DOUBLE PRECISION NOT NULL,
    messages    JSONB NOT NULL DEFAULT '[]',
    tokens_used BIGINT NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS turns (
    id          BIGSERIAL PRIMARY KEY,
    session_id  TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    question    TEXT NOT NULL,
    result      JSONB NOT NULL,
    created_at  DOUBLE PRECISION NOT NULL
);
ALTER TABLE turns ADD COLUMN IF NOT EXISTS attachments JSONB NOT NULL DEFAULT '[]';
CREATE TABLE IF NOT EXISTS chunks (
    id         BIGSERIAL PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES sessions(id) ON DELETE CASCADE,
    source     TEXT NOT NULL,
    metadata   JSONB NOT NULL,
    content    TEXT NOT NULL,
    embedding  BYTEA
);
CREATE INDEX IF NOT EXISTS turns_by_session ON turns(session_id, id);
CREATE INDEX IF NOT EXISTS chunks_by_source ON chunks(session_id, source);
CREATE INDEX IF NOT EXISTS sessions_by_updated ON sessions(updated_at);
"""


def _json(value: Any) -> Any:
    # SQLite returns JSON columns as text; Postgres JSONB comes back already decoded.
    return json.loads(value) if isinstance(value, str) else value


class Storage:
    """Backend-independent operations. Subclasses provide `_db()`, `_sql()` and `_executemany()`."""

    backend = "base"
    # True when dense search runs in the database (pgvector) instead of in process memory.
    vector_search_enabled = False

    def _sql(self, sql: str) -> str:
        raise NotImplementedError

    @contextmanager
    def _db(self) -> Iterator[Any]:
        raise NotImplementedError
        yield

    def _executemany(self, db: Any, sql: str, rows: list[tuple]) -> None:
        raise NotImplementedError

    def _run(self, db: Any, sql: str, params: tuple = ()):
        return db.execute(self._sql(sql), params)

    def ping(self) -> bool:
        try:
            with self._db() as db:
                self._run(db, "SELECT 1").fetchone()
            return True
        except Exception as exc:
            logger.warning("Database health check failed: %s", exc)
            return False

    def close(self) -> None:
        pass

    def vector_search(self, session_id: str, query: np.ndarray, k: int) -> list[tuple[str, float]]:
        """(chunk_id, cosine similarity) of the k nearest chunks; only with pgvector."""
        raise NotImplementedError

    # ----- sessions -----

    def create_session(self, session_id: str, owner: str) -> None:
        now = time.time()
        with self._db() as db:
            self._run(db, "INSERT INTO sessions (id, owner, created_at, updated_at) VALUES (?, ?, ?, ?)",
                      (session_id, owner, now, now))

    def session_owner(self, session_id: str) -> str | None:
        with self._db() as db:
            row = self._run(db, "SELECT owner FROM sessions WHERE id = ?", (session_id,)).fetchone()
        return row["owner"] if row else None

    def tokens_used(self, session_id: str) -> int:
        with self._db() as db:
            row = self._run(db, "SELECT tokens_used FROM sessions WHERE id = ?", (session_id,)).fetchone()
        return row["tokens_used"] if row else 0

    def load_messages(self, session_id: str) -> list[BaseMessage]:
        with self._db() as db:
            row = self._run(db, "SELECT messages FROM sessions WHERE id = ?", (session_id,)).fetchone()
        return messages_from_dict(_json(row["messages"])) if row else []

    def save_turn(
        self, session_id: str, question: str, result: dict, messages: list[BaseMessage], tokens: int,
        attachments: list[str] | None = None,
    ) -> None:
        now = time.time()
        with self._db() as db:
            self._run(
                db,
                "UPDATE sessions SET messages = {json}, tokens_used = tokens_used + ?, updated_at = ? WHERE id = ?",
                (json.dumps(messages_to_dict(messages)), tokens, now, session_id),
            )
            self._run(
                db,
                "INSERT INTO turns (session_id, question, result, attachments, created_at)"
                " VALUES (?, ?, {json}, {json}, ?)",
                (session_id, question, json.dumps(result), json.dumps(attachments or []), now),
            )

    def list_turns(self, session_id: str) -> list[dict]:
        with self._db() as db:
            rows = self._run(
                db, "SELECT question, result, attachments FROM turns WHERE session_id = ? ORDER BY id", (session_id,)
            ).fetchall()
        return [
            {"question": r["question"], "result": _json(r["result"]), "attachments": _json(r["attachments"])}
            for r in rows
        ]

    def reset_session(self, session_id: str) -> None:
        with self._db() as db:
            self._run(db, "UPDATE sessions SET messages = {json}, updated_at = ? WHERE id = ?",
                      ("[]", time.time(), session_id))
            self._run(db, "DELETE FROM turns WHERE session_id = ?", (session_id,))

    def purge_idle(self, older_than_days: int) -> int:
        cutoff = time.time() - older_than_days * 86400
        with self._db() as db:
            return self._run(db, "DELETE FROM sessions WHERE updated_at < ?", (cutoff,)).rowcount

    # ----- documents -----

    def replace_document(
        self, session_id: str, source: str, chunks: list[Document], vectors: np.ndarray | None
    ) -> None:
        self._replace_chunks(session_id, source, chunks, vectors)

    def _replace_chunks(
        self, session_id: str, source: str, chunks: list[Document], vectors: np.ndarray | None
    ) -> None:
        rows = [
            (
                session_id,
                source,
                json.dumps(chunk.metadata),
                chunk.page_content,
                vectors[i].astype(np.float32).tobytes() if vectors is not None else None,
            )
            for i, chunk in enumerate(chunks)
        ]
        with self._db() as db:
            self._run(db, "DELETE FROM chunks WHERE session_id = ? AND source = ?", (session_id, source))
            self._executemany(
                db,
                "INSERT INTO chunks (session_id, source, metadata, content, embedding) VALUES (?, ?, {json}, ?, ?)",
                rows,
            )

    def delete_document(self, session_id: str, source: str) -> None:
        with self._db() as db:
            self._run(db, "DELETE FROM chunks WHERE session_id = ? AND source = ?", (session_id, source))

    def load_chunks(self, session_id: str) -> tuple[list[Document], np.ndarray | None]:
        with self._db() as db:
            rows = self._run(
                db, "SELECT metadata, content, embedding FROM chunks WHERE session_id = ? ORDER BY id", (session_id,)
            ).fetchall()
        chunks = [Document(page_content=r["content"], metadata=_json(r["metadata"])) for r in rows]
        blobs = [r["embedding"] for r in rows]
        # Only reuse vectors if every chunk has one; otherwise the store re-embeds.
        if not rows or any(b is None for b in blobs):
            return chunks, None
        return chunks, np.stack([np.frombuffer(bytes(b), dtype=np.float32) for b in blobs])


class SQLiteStorage(Storage):
    """One short-lived connection per operation keeps it safe to call from worker threads."""

    backend = "sqlite"

    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self._db() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript(_SQLITE_SCHEMA)
            self._migrate(db)

    @staticmethod
    def _migrate(db: sqlite3.Connection) -> None:
        # Databases created before attachments existed lack the column.
        columns = {row["name"] for row in db.execute("PRAGMA table_info(turns)")}
        if "attachments" not in columns:
            db.execute("ALTER TABLE turns ADD COLUMN attachments TEXT NOT NULL DEFAULT '[]'")

    def _sql(self, sql: str) -> str:
        return sql.replace("{json}", "?")

    @contextmanager
    def _db(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            with conn:
                yield conn
        finally:
            conn.close()

    def _executemany(self, db: sqlite3.Connection, sql: str, rows: list[tuple]) -> None:
        db.executemany(self._sql(sql), rows)


class PostgresStorage(Storage):
    """Pooled psycopg 3 connections; each `_db()` block is one transaction."""

    backend = "postgres"

    def __init__(
        self, url: str, embedding_dim: int = 384, use_pgvector: bool = True, min_size: int = 1, max_size: int = 10
    ):
        from psycopg.rows import dict_row
        from psycopg_pool import ConnectionPool

        self.embedding_dim = int(embedding_dim)
        self._iterative_scan = False
        self._pool = ConnectionPool(
            url, min_size=min_size, max_size=max_size, open=True, timeout=10,
            kwargs={"row_factory": dict_row, "autocommit": False},
        )
        with self._db() as db:
            # Serialise schema setup across instances starting at the same time.
            db.execute("SELECT pg_advisory_xact_lock(4242)")
            db.execute(_POSTGRES_SCHEMA)
        if use_pgvector:
            self.vector_search_enabled = self._setup_pgvector()

    def _setup_pgvector(self) -> bool:
        """Enable server-side vector search if the pgvector extension is available.

        Any failure (extension not installed, no privilege, a column created for another
        embedding size) leaves in-memory vector search in place rather than failing startup.
        """
        import psycopg

        dim = self.embedding_dim
        try:
            with self._db() as db:
                db.execute("SELECT pg_advisory_xact_lock(4242)")
                db.execute("CREATE EXTENSION IF NOT EXISTS vector")
                row = db.execute(
                    "SELECT atttypmod FROM pg_attribute WHERE attrelid = 'chunks'::regclass AND attname = 'vec'"
                ).fetchone()
                if row and row["atttypmod"] != dim:
                    logger.warning(
                        "pgvector disabled: chunks.vec holds %d-dim vectors but EMBEDDING_DIM is %d",
                        row["atttypmod"], dim,
                    )
                    return False
                db.execute(f"ALTER TABLE chunks ADD COLUMN IF NOT EXISTS vec vector({dim})")
                # HNSW: approximate nearest neighbours, good recall without tuning or training.
                db.execute("CREATE INDEX IF NOT EXISTS chunks_vec_hnsw ON chunks USING hnsw (vec vector_cosine_ops)")
                version = db.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'").fetchone()
        except psycopg.Error as exc:
            logger.warning("pgvector unavailable (%s); vector search stays in memory", str(exc).strip())
            return False
        major, minor = (int(part) for part in version["extversion"].split(".")[:2])
        self._iterative_scan = (major, minor) >= (0, 8)
        backfilled = self._backfill_vectors()
        logger.info("pgvector %s enabled (%d dimensions, HNSW index%s)", version["extversion"], dim,
                    f", {backfilled} vectors migrated" if backfilled else "")
        return True

    def _backfill_vectors(self, batch: int = 500) -> int:
        """Copy vectors stored as raw bytes (before pgvector was enabled) into the vector column."""
        moved = 0
        while True:
            with self._db() as db:
                rows = db.execute(
                    "SELECT id, embedding FROM chunks WHERE vec IS NULL AND embedding IS NOT NULL"
                    " AND length(embedding) = %s LIMIT %s",
                    (self.embedding_dim * 4, batch),
                ).fetchall()
                if not rows:
                    return moved
                with db.cursor() as cursor:
                    cursor.executemany(
                        "UPDATE chunks SET vec = %s::vector, embedding = NULL WHERE id = %s",
                        [(_vector_literal(np.frombuffer(bytes(r["embedding"]), dtype=np.float32)), r["id"])
                         for r in rows],
                    )
            moved += len(rows)

    def replace_document(
        self, session_id: str, source: str, chunks: list[Document], vectors: np.ndarray | None
    ) -> None:
        if not self.vector_search_enabled or vectors is None:
            self._replace_chunks(session_id, source, chunks, vectors)
            return
        if vectors.shape[1] != self.embedding_dim:
            logger.warning("Vectors for %s have %d dims, expected %d; storing without pgvector",
                           source, vectors.shape[1], self.embedding_dim)
            self._replace_chunks(session_id, source, chunks, vectors)
            return
        rows = [
            (session_id, source, json.dumps(chunk.metadata), chunk.page_content, _vector_literal(vectors[i]))
            for i, chunk in enumerate(chunks)
        ]
        with self._db() as db:
            db.execute("DELETE FROM chunks WHERE session_id = %s AND source = %s", (session_id, source))
            with db.cursor() as cursor:
                cursor.executemany(
                    "INSERT INTO chunks (session_id, source, metadata, content, vec)"
                    " VALUES (%s, %s, %s::jsonb, %s, %s::vector)",
                    rows,
                )

    def load_chunks(self, session_id: str) -> tuple[list[Document], np.ndarray | None]:
        chunks, vectors = super().load_chunks(session_id)
        # With pgvector the vectors stay in the database; the store queries them there.
        return (chunks, None) if self.vector_search_enabled else (chunks, vectors)

    def vector_search(self, session_id: str, query: np.ndarray, k: int) -> list[tuple[str, float]]:
        literal = _vector_literal(query)
        with self._db() as db:
            if self._iterative_scan:
                # Without this, HNSW finds the global nearest rows first and the session filter
                # can leave fewer than k; iterative scans keep searching until k rows match.
                db.execute("SET LOCAL hnsw.iterative_scan = 'relaxed_order'")
            rows = db.execute(
                "SELECT metadata->>'chunk_id' AS chunk_id, 1 - (vec <=> %s::vector) AS similarity"
                " FROM chunks WHERE session_id = %s AND vec IS NOT NULL"
                " ORDER BY vec <=> %s::vector LIMIT %s",
                (literal, session_id, literal, k),
            ).fetchall()
        return [(row["chunk_id"], float(row["similarity"])) for row in rows]

    def _sql(self, sql: str) -> str:
        return sql.replace("{json}", "%s::jsonb").replace("?", "%s")

    @contextmanager
    def _db(self) -> Iterator[Any]:
        with self._pool.connection() as conn, conn.transaction():
            yield conn

    def _executemany(self, db: Any, sql: str, rows: list[tuple]) -> None:
        with db.cursor() as cursor:
            cursor.executemany(self._sql(sql), rows)

    def close(self) -> None:
        self._pool.close()


def _vector_literal(vector: np.ndarray) -> str:
    return "[" + ",".join(f"{float(x):.7g}" for x in vector) + "]"


def create_storage(
    database_url: str | None, sqlite_path: Path, embedding_dim: int = 384, use_pgvector: bool = True
) -> Storage:
    if database_url and database_url.startswith(("postgres://", "postgresql://")):
        storage = PostgresStorage(
            database_url.replace("postgres://", "postgresql://", 1), embedding_dim, use_pgvector
        )
    elif database_url:
        raise ValueError("DATABASE_URL must start with postgresql:// (leave it unset to use SQLite)")
    else:
        storage = SQLiteStorage(sqlite_path)
    logger.info("Using %s storage (vector search: %s)", storage.backend,
                "pgvector" if storage.vector_search_enabled else "in memory")
    return storage
