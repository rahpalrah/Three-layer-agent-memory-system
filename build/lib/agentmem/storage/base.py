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

    def clear(self, namespace: str) -> None:
        for it in self.list(namespace):
            self.delete(it.id)

    def close(self) -> None:
        pass
