"""Stage-1 candidate generation: a persistent in-process index per namespace.

Industry two-stage retrieval: a cheap index proposes a few hundred candidates
(vector top-N  ∪  BM25 top-N from an inverted index), then the full ranker
(hybrid relevance x recency x importance, optional cross-encoder) runs only on
those. Recall no longer loads every memory from storage.

The index is kept in sync with writes made through :class:`IndexedStorage`, and
rebuilds itself when another process or connection changed the data (detected
through the storage's ``change_token``).

Vector backends: ``"numpy"`` (exact, default) or ``"faiss"`` (HNSW approximate
nearest neighbours, ``pip install faiss-cpu``).
"""
from __future__ import annotations

import math
import threading
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .embeddings import tokenize
from .storage.base import StorageBackend
from .types import Layer, MemoryItem, Session

INDEXED_LAYERS = (Layer.EPISODIC, Layer.SEMANTIC)


class _VectorStore:
    """Exact inner-product search over a growable float32 matrix."""

    def __init__(self, dim: int) -> None:
        import numpy as np
        self.np = np
        self.dim = dim
        self.mat = np.zeros((1024, dim), dtype=np.float32)
        self.n = 0
        self.alive = np.zeros(1024, dtype=bool)

    def add(self, vec: Sequence[float]) -> int:
        np = self.np
        if self.n == len(self.mat):
            self.mat = np.vstack([self.mat, np.zeros_like(self.mat)])
            self.alive = np.concatenate([self.alive, np.zeros_like(self.alive)])
        self.mat[self.n] = vec
        self.alive[self.n] = True
        self.n += 1
        return self.n - 1

    def remove(self, slot: int) -> None:
        self.alive[slot] = False

    def search(self, q: Sequence[float], n: int) -> Tuple[List[int], List[float]]:
        np = self.np
        if self.n == 0:
            return [], []
        sims = self.mat[: self.n] @ np.asarray(q, dtype=np.float32)
        sims[~self.alive[: self.n]] = -np.inf
        n = min(n, int(self.alive[: self.n].sum()))
        if n <= 0:
            return [], []
        top = np.argpartition(-sims, n - 1)[:n]
        top = top[np.argsort(-sims[top])]
        return top.tolist(), sims[top].tolist()

    def sims(self, q: Sequence[float], slots: Sequence[int]) -> List[float]:
        np = self.np
        if not slots:
            return []
        return (self.mat[list(slots)] @ np.asarray(q, dtype=np.float32)).tolist()


class _FaissHNSW(_VectorStore):
    """Approximate search with FAISS HNSW; exact vectors kept for re-scoring."""

    def __init__(self, dim: int, m: int = 32, ef_search: int = 128) -> None:
        super().__init__(dim)
        import faiss
        self.faiss = faiss
        self.index = faiss.IndexHNSWFlat(dim, m, faiss.METRIC_INNER_PRODUCT)
        self.index.hnsw.efSearch = ef_search
        self.index.hnsw.efConstruction = 80
        self._pending: List[int] = []

    def add(self, vec) -> int:
        slot = super().add(vec)
        self._pending.append(slot)
        return slot

    def _flush(self):
        if self._pending:
            self.index.add(self.mat[self._pending[0]: self._pending[-1] + 1])
            self._pending.clear()

    def search(self, q, n):
        np = self.np
        self._flush()
        if self.index.ntotal == 0:
            return [], []
        k = min(self.index.ntotal, n + int((~self.alive[: self.n]).sum()))
        self.index.hnsw.efSearch = max(self.index.hnsw.efSearch, k)
        d, i = self.index.search(np.asarray([q], dtype=np.float32), k)
        out = [(int(s), float(v)) for s, v in zip(i[0], d[0]) if s >= 0 and self.alive[s]][:n]
        return [s for s, _ in out], [v for _, v in out]


class MemoryIndex:
    def __init__(self, namespace: str, backend: str = "numpy") -> None:
        self.namespace = namespace
        self.backend = backend
        self.lock = threading.RLock()
        self._reset()

    def _reset(self) -> None:
        self.slot_of: Dict[str, int] = {}
        self.ids: List[Optional[str]] = []
        self.layer: List[Optional[Layer]] = []
        self.content_hash: List[int] = []
        self.vec: Optional[_VectorStore] = None
        self.postings: Dict[str, Dict[int, int]] = defaultdict(dict)
        self.doc_len: List[int] = []
        self.doc_terms: List[tuple] = []
        self.slot_tags: List[frozenset] = []
        self.tag_slots: Dict[str, set] = defaultdict(set)
        self.total_len = 0
        self.n_alive = 0
        self.synced_token = None

    # ------------------------------------------------------------ maintenance
    def _new_vec(self, dim: int) -> _VectorStore:
        if self.backend == "faiss":
            return _FaissHNSW(dim)
        if self.backend != "numpy":
            raise ValueError(f"unknown index backend {self.backend!r}")
        return _VectorStore(dim)

    def rebuild(self, items: Iterable[MemoryItem], token) -> None:
        with self.lock:
            self._reset()
            for it in items:
                self._add(it)
            self.synced_token = token

    def upsert(self, it: MemoryItem) -> None:
        with self.lock:
            if it.namespace != self.namespace or it.layer not in INDEXED_LAYERS:
                self.remove(it.id)
                return
            slot = self.slot_of.get(it.id)
            if slot is not None and self.content_hash[slot] == hash(it.content) and self.layer[slot] == it.layer:
                if it.metadata.get("superseded_by"):
                    self.remove(it.id)
                return  # only access stats / importance changed - nothing to index
            self.remove(it.id)
            if not it.metadata.get("superseded_by"):
                self._add(it)

    def _add(self, it: MemoryItem) -> None:
        if it.layer not in INDEXED_LAYERS or it.metadata.get("superseded_by"):
            return
        if it.embedding:
            if self.vec is None:
                self.vec = self._new_vec(len(it.embedding))
            if len(it.embedding) != self.vec.dim:
                return  # mixed embedders in one namespace: skip vector part
            slot = self.vec.add(it.embedding)
        else:
            if self.vec is None:
                return
            slot = self.vec.add([0.0] * self.vec.dim)
        while len(self.ids) <= slot:
            self.ids.append(None); self.layer.append(None); self.content_hash.append(0); self.doc_len.append(0)
            self.doc_terms.append(())
            self.slot_tags.append(frozenset())
        self.ids[slot], self.layer[slot], self.content_hash[slot] = it.id, it.layer, hash(it.content)
        toks = tokenize(it.content)
        for w in toks:
            p = self.postings[w]
            p[slot] = p.get(slot, 0) + 1
        self.doc_len[slot] = len(toks)
        self.doc_terms[slot] = tuple(set(toks))
        self.slot_tags[slot] = frozenset(it.tags)
        for t in it.tags:
            self.tag_slots[t].add(slot)
        self.total_len += len(toks)
        self.n_alive += 1
        self.slot_of[it.id] = slot

    def remove(self, item_id: str) -> None:
        with self.lock:
            slot = self.slot_of.pop(item_id, None)
            if slot is None:
                return
            if self.vec is not None:
                self.vec.remove(slot)
            for w in self.doc_terms[slot]:
                p = self.postings.get(w)
                if p is not None:
                    p.pop(slot, None)
                    if not p:
                        del self.postings[w]
            self.doc_terms[slot] = ()
            for t in self.slot_tags[slot]:
                self.tag_slots[t].discard(slot)
            self.slot_tags[slot] = frozenset()
            self.total_len -= self.doc_len[slot]
            self.doc_len[slot] = 0
            self.ids[slot] = None
            self.n_alive -= 1

    # ----------------------------------------------------------------- search
    def bm25(self, query: str, k1: float = 1.5, b: float = 0.75) -> Dict[int, float]:
        q = list(dict.fromkeys(tokenize(query or "")))
        n = max(self.n_alive, 1)
        avg = (self.total_len / n) or 1.0
        out: Dict[int, float] = defaultdict(float)
        for w in q:
            p = self.postings.get(w)
            if not p:
                continue
            idf = math.log(1 + (n - len(p) + 0.5) / (len(p) + 0.5))
            for slot, f in p.items():
                out[slot] += idf * f * (k1 + 1) / (f + k1 * (1 - b + b * self.doc_len[slot] / avg))
        return out

    def candidates(self, query_vec, query_text: Optional[str], n: int = 200,
                   layers: Sequence[Layer] = INDEXED_LAYERS, tags: Optional[Sequence[str]] = None):
        """Returns {item_id: (dense_sim, bm25)} for the union of vector and keyword top-n.

        ``tags`` restricts candidates *before* the top-n cut. Filtering after the cut loses
        the tagged items whenever other items outrank them (the bug this guards against)."""
        with self.lock:
            want = set(layers)
            allowed = None
            if tags:
                allowed = set().union(*(self.tag_slots.get(t, set()) for t in tags))
                if not allowed:
                    return {}
            dense: Dict[int, float] = {}
            if self.vec is not None and query_vec is not None and any(query_vec):
                if allowed is None:
                    slots, sims = self.vec.search(query_vec, n * 2 if len(want) < 2 else n)
                    dense = {s: v for s, v in zip(slots, sims) if self.layer[s] in want}
                else:
                    pool = [s for s in allowed if self.layer[s] in want]
                    sims = self.vec.sims(query_vec, pool)
                    dense = dict(sorted(zip(pool, sims), key=lambda t: t[1], reverse=True)[: n * 2])
            lex = self.bm25(query_text) if query_text else {}
            if allowed is not None:
                lex = {s: v for s, v in lex.items() if s in allowed}
            lex_top = sorted(((v, s) for s, v in lex.items() if self.layer[s] in want), reverse=True)[:n]
            slots = set(dense) | {s for _, s in lex_top}
            missing = [s for s in slots if s not in dense]
            if missing and self.vec is not None and query_vec is not None:
                dense.update(zip(missing, self.vec.sims(query_vec, missing)))
            return {self.ids[s]: (max(dense.get(s, 0.0), 0.0), lex.get(s, 0.0)) for s in slots if self.ids[s]}


class IndexedStorage(StorageBackend):
    """Wraps any backend and keeps a :class:`MemoryIndex` in sync with writes."""

    def __init__(self, inner: StorageBackend, index: MemoryIndex) -> None:
        self.inner = inner
        self.index = index

    def _write(self, fn, *args):
        before = self.inner.change_token()
        out = fn(*args)
        if before == self.index.synced_token:
            self.index.synced_token = self.inner.change_token()
        return out

    def ensure_fresh(self) -> None:
        tok = self.inner.change_token()
        if tok != self.index.synced_token:
            items = [i for layer in INDEXED_LAYERS for i in self.inner.list(self.index.namespace, layer)]
            self.index.rebuild(items, tok)

    def put(self, item: MemoryItem) -> None:
        self._write(self.inner.put, item)
        if item.namespace == self.index.namespace:
            self.index.upsert(item)

    def put_many(self, items) -> None:
        items = list(items)
        self._write(self.inner.put_many, items)
        for it in items:
            if it.namespace == self.index.namespace:
                self.index.upsert(it)

    def delete(self, item_id: str) -> bool:
        ok = self._write(self.inner.delete, item_id)
        self.index.remove(item_id)
        return ok

    def get(self, item_id): return self.inner.get(item_id)
    def get_many(self, ids): return self.inner.get_many(ids)
    def list(self, *a, **kw): return self.inner.list(*a, **kw)
    def count(self, *a, **kw): return self.inner.count(*a, **kw)
    def put_session(self, s: Session) -> None: self.inner.put_session(s)
    def get_session(self, sid): return self.inner.get_session(sid)
    def list_sessions(self, *a, **kw): return self.inner.list_sessions(*a, **kw)
    def change_token(self): return self.inner.change_token()

    def clear(self, namespace) -> None:
        self._write(self.inner.clear, namespace)
        if namespace == self.index.namespace:
            self.index.rebuild([], self.inner.change_token())

    def close(self) -> None:
        self.inner.close()
