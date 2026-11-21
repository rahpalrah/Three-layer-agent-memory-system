"""Layer 3 - Semantic memory: distilled, de-duplicated knowledge ("what is true").

Facts are learned from consolidation or directly. A new fact that closely
matches an existing one *reinforces* it (importance and evidence count go up)
instead of creating a duplicate. Facts can be superseded via ``key``
(e.g. key="user.city" - a newer value replaces the older one).
"""
from __future__ import annotations

import time
from typing import List, Optional

from .embeddings import Embedder, cosine
from .scoring import RetrievalWeights, rank
from .storage.base import StorageBackend
from .types import Layer, MemoryItem, ScoredMemory


class SemanticMemory:
    def __init__(self, storage: StorageBackend, embedder: Embedder, namespace: str,
                 dedup_threshold: float = 0.85, weights: Optional[RetrievalWeights] = None) -> None:
        self.storage = storage
        self.embedder = embedder
        self.namespace = namespace
        self.dedup_threshold = dedup_threshold
        # facts are durable: slow recency/importance decay by default
        self.weights = weights or RetrievalWeights(relevance=1.0, recency=0.1, importance=0.6,
                                                   importance_half_life_days=365.0)

    def facts(self) -> List[MemoryItem]:
        return self.storage.list(self.namespace, Layer.SEMANTIC)

    def learn(self, content: str, importance: float = 0.6, key: Optional[str] = None,
              source_ids: Optional[List[str]] = None, tags: Optional[List[str]] = None,
              pinned: bool = False, metadata: Optional[dict] = None) -> MemoryItem:
        vec = self.embedder.embed(content)
        existing = self.facts()
        now = time.time()

        # 1) keyed facts supersede older values
        if key:
            for f in existing:
                if f.metadata.get("key") == key:
                    if f.content.strip().lower() == content.strip().lower():
                        return self._reinforce(f, importance, source_ids, now)
                    history = f.metadata.get("history", []) + [{"content": f.content, "at": f.last_accessed}]
                    f.content, f.embedding = content, vec
                    f.importance = max(f.importance, importance)
                    f.metadata["history"] = history[-10:]
                    f.last_accessed = now
                    self._merge_sources(f, source_ids)
                    self.storage.put(f)
                    return f

        # 2) near-duplicates are reinforced, not duplicated
        best, best_sim = None, 0.0
        for f in existing:
            if f.embedding:
                s = cosine(vec, f.embedding)
                if s > best_sim:
                    best, best_sim = f, s
        if best is not None and best_sim >= self.dedup_threshold:
            return self._reinforce(best, importance, source_ids, now)

        meta = dict(metadata or {})
        meta.update({"evidence": 1, "sources": list(source_ids or [])})
        if key:
            meta["key"] = key
        item = MemoryItem(content=content, layer=Layer.SEMANTIC, importance=importance,
                          namespace=self.namespace, embedding=vec, tags=list(tags or []),
                          pinned=pinned, metadata=meta)
        self.storage.put(item)
        return item

    def _reinforce(self, f: MemoryItem, importance: float, source_ids, now: float) -> MemoryItem:
        f.metadata["evidence"] = int(f.metadata.get("evidence", 1)) + 1
        # each confirmation moves importance toward 1
        f.importance = min(1.0, max(f.importance, importance) + 0.1 * (1 - f.importance))
        f.last_accessed = now
        self._merge_sources(f, source_ids)
        self.storage.put(f)
        return f

    @staticmethod
    def _merge_sources(f: MemoryItem, source_ids) -> None:
        if source_ids:
            srcs = f.metadata.get("sources", [])
            f.metadata["sources"] = list(dict.fromkeys(srcs + list(source_ids)))[-50:]

    def retrieve(self, query: Optional[str] = None, k: int = 5, tags: Optional[List[str]] = None,
                 reinforce: bool = True) -> List[ScoredMemory]:
        items = self.facts()
        if tags:
            items = [i for i in items if set(tags) & set(i.tags)]
        qv = self.embedder.embed(query) if query else None
        hits = rank(items, qv, self.weights)[:k]
        if reinforce:
            now = time.time()
            for h in hits:
                h.item.touch(now)
                self.storage.put(h.item)
        return hits

    def get_by_key(self, key: str) -> Optional[MemoryItem]:
        return next((f for f in self.facts() if f.metadata.get("key") == key), None)

    def forget(self, item_id: str) -> bool:
        return self.storage.delete(item_id)

    def count(self) -> int:
        return self.storage.count(self.namespace, Layer.SEMANTIC)
