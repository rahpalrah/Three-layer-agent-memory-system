"""Storage backend protocol. Implement this to persist memory anywhere
(Postgres + pgvector, Redis, DynamoDB, a vector DB...)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List, Optional

from ..types import Layer, MemoryItem, Session


class StorageBackend(ABC):
    @abstractmethod
    def put(self, item: MemoryItem) -> None:
        """Insert or replace an item (upsert by id)."""

    @abstractmethod
    def get(self, item_id: str) -> Optional[MemoryItem]: ...

    @abstractmethod
    def delete(self, item_id: str) -> bool: ...

    @abstractmethod
    def list(self, namespace: str, layer: Optional[Layer] = None,
             session_id: Optional[str] = None, limit: Optional[int] = None) -> List[MemoryItem]:
        """Items newest-first, filtered."""

    @abstractmethod
    def count(self, namespace: str, layer: Optional[Layer] = None) -> int: ...

    # sessions
    @abstractmethod
    def put_session(self, session: Session) -> None: ...

    @abstractmethod
    def get_session(self, session_id: str) -> Optional[Session]: ...

    @abstractmethod
    def list_sessions(self, namespace: str, limit: Optional[int] = None) -> List[Session]: ...

    def put_many(self, items) -> None:
        """Bulk upsert. Override for efficiency (e.g. one transaction)."""
        for it in items:
            self.put(it)

    def get_many(self, item_ids) -> List[MemoryItem]:
        """Fetch several items (missing ids are skipped). Override for efficiency."""
        return [i for i in (self.get(x) for x in item_ids) if i is not None]

    def change_token(self):
        """Opaque value that changes whenever the stored data may have changed
        (including writes by other processes). Indexes use it to stay fresh."""
        return None

    def clear(self, namespace: str) -> None:
        for it in self.list(namespace):
            self.delete(it.id)

    def close(self) -> None:
        pass
