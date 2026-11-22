from __future__ import annotations

import json
from array import array
import sqlite3
import threading
from pathlib import Path
from typing import List, Optional, Union

from ..types import Layer, MemoryItem, Session
from .base import StorageBackend

_SCHEMA = """
CREATE TABLE IF NOT EXISTS memories (
    id TEXT PRIMARY KEY,
    namespace TEXT NOT NULL,
    layer TEXT NOT NULL,
    content TEXT NOT NULL,
    importance REAL NOT NULL,
    session_id TEXT,
    tags TEXT NOT NULL DEFAULT '[]',
    metadata TEXT NOT NULL DEFAULT '{}',
    embedding BLOB,
    pinned INTEGER NOT NULL DEFAULT 0,
    access_count INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL,
    last_accessed REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_mem_ns_layer ON memories(namespace, layer, created_at);
CREATE INDEX IF NOT EXISTS ix_mem_session ON memories(session_id);
CREATE TABLE IF NOT EXISTS sessions (
    id TEXT PRIMARY KEY,
    namespace TEXT NOT NULL,
    started_at REAL NOT NULL,
    ended_at REAL,
    summary TEXT,
    metadata TEXT NOT NULL DEFAULT '{}'
);
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);
"""
SCHEMA_VERSION = "1"


def _decode_vec(v):
    if not v:
        return None
    if isinstance(v, str):  # legacy JSON text
        return json.loads(v)
    a = array("f")
    a.frombytes(v)
    return a.tolist()


class SQLiteStorage(StorageBackend):
    """Durable, zero-config persistence - memories survive across sessions and restarts."""

    def __init__(self, path: Union[str, Path] = "agentmem.db") -> None:
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False, timeout=30)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.RLock()
        self._version = 0  # writes through this connection
        with self._lock, self._conn:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")  # durable with WAL, much faster
            self._conn.execute("PRAGMA busy_timeout=30000")   # wait for other writers
            self._conn.executescript(_SCHEMA)
            self._conn.execute("INSERT OR IGNORE INTO meta VALUES ('schema_version', ?)", (SCHEMA_VERSION,))

    @staticmethod
    def _row(r: sqlite3.Row) -> MemoryItem:
        return MemoryItem(
            id=r["id"], namespace=r["namespace"], layer=Layer(r["layer"]), content=r["content"],
            importance=r["importance"], session_id=r["session_id"], tags=json.loads(r["tags"]),
            metadata=json.loads(r["metadata"]),
            embedding=_decode_vec(r["embedding"]),
            pinned=bool(r["pinned"]), access_count=r["access_count"],
            created_at=r["created_at"], last_accessed=r["last_accessed"],
        )

    def put(self, item: MemoryItem) -> None:
        with self._lock, self._conn:
            self._conn.execute(
                "INSERT OR REPLACE INTO memories VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (item.id, item.namespace, item.layer.value, item.content, item.importance,
                 item.session_id, json.dumps(item.tags), json.dumps(item.metadata),
                 array("f", item.embedding).tobytes() if item.embedding else None,
                 int(item.pinned), item.access_count, item.created_at, item.last_accessed))
            self._version += 1

    def put_many(self, items) -> None:
        rows = [(i.id, i.namespace, i.layer.value, i.content, i.importance, i.session_id, json.dumps(i.tags),
                 json.dumps(i.metadata), array("f", i.embedding).tobytes() if i.embedding else None,
                 int(i.pinned), i.access_count, i.created_at, i.last_accessed) for i in items]
        with self._lock, self._conn:
            self._conn.executemany("INSERT OR REPLACE INTO memories VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
            self._version += 1

    def get(self, item_id):
        with self._lock:
            r = self._conn.execute("SELECT * FROM memories WHERE id=?", (item_id,)).fetchone()
        return self._row(r) if r else None

    def delete(self, item_id) -> bool:
        with self._lock, self._conn:
            self._version += 1
            return self._conn.execute("DELETE FROM memories WHERE id=?", (item_id,)).rowcount > 0

    def list(self, namespace, layer=None, session_id=None, limit=None) -> List[MemoryItem]:
        q, args = "SELECT * FROM memories WHERE namespace=?", [namespace]
        if layer is not None:
            q += " AND layer=?"; args.append(layer.value)
        if session_id is not None:
            q += " AND session_id=?"; args.append(session_id)
        q += " ORDER BY created_at DESC"
        if limit:
            q += " LIMIT ?"; args.append(int(limit))
        with self._lock:
            return [self._row(r) for r in self._conn.execute(q, args).fetchall()]

    def change_token(self):
        # data_version changes when *other* connections/processes commit
        with self._lock:
            return (self._version, self._conn.execute("PRAGMA data_version").fetchone()[0])

    def get_many(self, item_ids):
        ids = list(item_ids)
        out = []
        with self._lock:
            for i in range(0, len(ids), 500):
                chunk = ids[i:i + 500]
                q = f"SELECT * FROM memories WHERE id IN ({','.join('?' * len(chunk))})"
                out += [self._row(r) for r in self._conn.execute(q, chunk).fetchall()]
        return out

    def count(self, namespace, layer=None) -> int:
        q, args = "SELECT COUNT(*) FROM memories WHERE namespace=?", [namespace]
        if layer is not None:
            q += " AND layer=?"; args.append(layer.value)
        with self._lock:
            return self._conn.execute(q, args).fetchone()[0]

    def clear(self, namespace) -> None:
        with self._lock, self._conn:
            self._version += 1
            self._conn.execute("DELETE FROM memories WHERE namespace=?", (namespace,))
            self._conn.execute("DELETE FROM sessions WHERE namespace=?", (namespace,))

    def put_session(self, s: Session) -> None:
        with self._lock, self._conn:
            self._conn.execute("INSERT OR REPLACE INTO sessions VALUES (?,?,?,?,?,?)",
                               (s.id, s.namespace, s.started_at, s.ended_at, s.summary, json.dumps(s.metadata)))

    @staticmethod
    def _srow(r) -> Session:
        return Session(id=r["id"], namespace=r["namespace"], started_at=r["started_at"],
                       ended_at=r["ended_at"], summary=r["summary"], metadata=json.loads(r["metadata"]))

    def get_session(self, session_id):
        with self._lock:
            r = self._conn.execute("SELECT * FROM sessions WHERE id=?", (session_id,)).fetchone()
        return self._srow(r) if r else None

    def list_sessions(self, namespace, limit=None):
        q, args = "SELECT * FROM sessions WHERE namespace=? ORDER BY started_at DESC", [namespace]
        if limit:
            q += " LIMIT ?"; args.append(int(limit))
        with self._lock:
            return [self._srow(r) for r in self._conn.execute(q, args).fetchall()]

    def close(self) -> None:
        with self._lock:
            self._conn.close()
