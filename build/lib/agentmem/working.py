"""Layer 1 - Working memory: the agent's short, bounded scratchpad.

Holds the most recent turns of the current session. When it overflows (by
item count or approximate token budget) the oldest items are *evicted*: they
are promoted into episodic memory, or dropped if they are trivial.
Working items are mirrored to storage so a crashed session can be resumed.
"""
from __future__ import annotations

from typing import Callable, List, Optional

from .storage.base import StorageBackend
from .types import Layer, MemoryItem


def approx_tokens(text: str) -> int:
    return max(1, len(text) // 4)


class WorkingMemory:
    def __init__(self, storage: StorageBackend, namespace: str, capacity: int = 20,
                 token_budget: Optional[int] = 4000,
                 on_evict: Optional[Callable[[MemoryItem], None]] = None) -> None:
        self.storage = storage
        self.namespace = namespace
        self.capacity = capacity
        self.token_budget = token_budget
        self.on_evict = on_evict
        self._items: List[MemoryItem] = []

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self):
        return iter(list(self._items))

    @property
    def tokens(self) -> int:
        return sum(approx_tokens(i.content) for i in self._items)

    def add(self, item: MemoryItem) -> List[MemoryItem]:
        """Add an item; returns any items evicted to make room."""
        item.layer = Layer.WORKING
        item.namespace = self.namespace
        self._items.append(item)
        self.storage.put(item)
        evicted = []
        while len(self._items) > 1 and (
            len(self._items) > self.capacity
            or (self.token_budget is not None and self.tokens > self.token_budget)
        ):
            evicted.append(self._evict(0))
        return evicted

    def _evict(self, idx: int) -> MemoryItem:
        item = self._items.pop(idx)
        self.storage.delete(item.id)
        if self.on_evict:
            self.on_evict(item)
        return item

    def items(self) -> List[MemoryItem]:
        return list(self._items)

    def flush(self) -> List[MemoryItem]:
        """Evict everything (called at session end)."""
        out = []
        while self._items:
            out.append(self._evict(0))
        return out

    def restore(self, session_id: str) -> int:
        """Reload working items persisted for an interrupted session."""
        items = self.storage.list(self.namespace, Layer.WORKING, session_id=session_id)
        self._items = sorted(items, key=lambda i: i.created_at)
        return len(self._items)

    def as_messages(self) -> List[dict]:
        """Render as chat messages ``[{"role", "content"}]`` for an LLM call."""
        return [{"role": i.metadata.get("role", "user"), "content": i.content} for i in self._items]
