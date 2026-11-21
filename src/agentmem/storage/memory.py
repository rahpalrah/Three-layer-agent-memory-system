from __future__ import annotations

import copy
import threading
from typing import Dict, List, Optional

from ..types import Layer, MemoryItem, Session
from .base import StorageBackend


class InMemoryStorage(StorageBackend):
    """Process-local storage. Fast; nothing survives a restart."""

    def __init__(self) -> None:
        self._items: Dict[str, MemoryItem] = {}
        self._sessions: Dict[str, Session] = {}
        self._lock = threading.RLock()

    def put(self, item: MemoryItem) -> None:
        with self._lock:
            self._items[item.id] = copy.deepcopy(item)

    def get(self, item_id: str) -> Optional[MemoryItem]:
        with self._lock:
            it = self._items.get(item_id)
            return copy.deepcopy(it) if it else None

    def delete(self, item_id: str) -> bool:
        with self._lock:
            return self._items.pop(item_id, None) is not None

    def list(self, namespace, layer=None, session_id=None, limit=None) -> List[MemoryItem]:
        with self._lock:
            out = [copy.deepcopy(i) for i in self._items.values()
                   if i.namespace == namespace
                   and (layer is None or i.layer == layer)
                   and (session_id is None or i.session_id == session_id)]
        out.sort(key=lambda i: i.created_at, reverse=True)
        return out[:limit] if limit else out

    def count(self, namespace, layer=None) -> int:
        return len(self.list(namespace, layer))

    def put_session(self, session: Session) -> None:
        with self._lock:
            self._sessions[session.id] = copy.deepcopy(session)

    def get_session(self, session_id):
        with self._lock:
            s = self._sessions.get(session_id)
            return copy.deepcopy(s) if s else None

    def list_sessions(self, namespace, limit=None):
        with self._lock:
            out = [copy.deepcopy(s) for s in self._sessions.values() if s.namespace == namespace]
        out.sort(key=lambda s: s.started_at, reverse=True)
        return out[:limit] if limit else out
