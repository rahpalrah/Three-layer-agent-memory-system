"""Layer 2 - Episodic memory: time-stamped experiences ("what happened").

Every episode keeps its session id, timestamp, importance and access stats,
so the agent can recall *when* and *in which conversation* something occurred.
"""
from __future__ import annotations

import time
from collections import deque
from typing import Callable, Deque, List, Optional

from .embeddings import Embedder
from .scoring import RetrievalWeights, rank, retention_score
from .storage.base import StorageBackend
from .types import Layer, MemoryItem, ScoredMemory


class EpisodicMemory:
    def __init__(self, storage: StorageBackend, embedder: Embedder, namespace: str,
                 weights: Optional[RetrievalWeights] = None,
                 clock: Callable[[], float] = time.time) -> None:
        self.clock = clock
        self._recent: Deque[MemoryItem] = deque(maxlen=64)
        self.storage = storage
        self.embedder = embedder
        self.namespace = namespace
        self.weights = weights or RetrievalWeights()

    def record(self, item: MemoryItem) -> MemoryItem:
        item.layer = Layer.EPISODIC
        item.namespace = self.namespace
        if item.embedding is None:
            item.embedding = self.embedder.embed(item.content)
        self.storage.put(item)
        self._recent.append(item)
        return item

    def recent_context(self, n: int = 20) -> List[MemoryItem]:
        """The last few episodes recorded in this process, newest first. Used for the novelty
        check on every write; cheap regardless of how large the store is. (Entries may since
        have been forgotten, which only makes the novelty estimate slightly stale.)"""
        return list(self._recent)[-n:][::-1]

    def retrieve(self, query: Optional[str] = None, k: int = 5, session_id: Optional[str] = None,
                 tags: Optional[List[str]] = None, since: Optional[float] = None,
                 weights: Optional[RetrievalWeights] = None, reinforce: bool = True) -> List[ScoredMemory]:
        items = self.storage.list(self.namespace, Layer.EPISODIC, session_id=session_id)
        if tags:
            items = [i for i in items if set(tags) & set(i.tags)]
        if since:
            items = [i for i in items if i.created_at >= since]
        qv = self.embedder.embed(query) if query else None
        now = self.clock()
        hits = rank(items, qv, weights or self.weights, now=now, query_text=query)[:k]
        if reinforce:
            for h in hits:
                h.item.touch(now)
                self.storage.put(h.item)
        return hits

    def recent(self, n: int = 10, session_id: Optional[str] = None) -> List[MemoryItem]:
        return self.storage.list(self.namespace, Layer.EPISODIC, session_id=session_id, limit=n)

    def timeline(self, session_id: Optional[str] = None) -> List[MemoryItem]:
        return sorted(self.storage.list(self.namespace, Layer.EPISODIC, session_id=session_id),
                      key=lambda i: i.created_at)

    def forget(self, threshold: float = 0.15, max_items: Optional[int] = None,
               now: Optional[float] = None) -> int:
        """Drop low-retention episodes; optionally cap total size. Pinned items are kept."""
        now = now or self.clock()
        items = self.storage.list(self.namespace, Layer.EPISODIC)
        scored = sorted(((retention_score(i, now, self.weights), i) for i in items), key=lambda t: t[0])
        removed = 0
        keep = len(items)
        for s, it in scored:
            if it.pinned:
                continue
            over_cap = max_items is not None and keep > max_items
            if s < threshold or over_cap:
                self.storage.delete(it.id)
                removed += 1
                keep -= 1
        return removed

    def count(self) -> int:
        return self.storage.count(self.namespace, Layer.EPISODIC)
