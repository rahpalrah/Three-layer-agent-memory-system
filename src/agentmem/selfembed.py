"""A self-contained semantic embedder: no downloads, no third-party packages, no API.

Word meanings are learned from the text itself, unsupervised:
  1. count how often words co-occur within a window (sparse dictionaries);
  2. convert counts to positive pointwise mutual information (PPMI), the classic
     count-based route to word vectors (Levy & Goldberg 2014);
  3. project each word's PPMI row to ``dim`` dimensions with a deterministic random
     projection (Johnson-Lindenstrauss, a seeded hash per feature, so no matrix is stored);
  4. a sentence is the IDF-weighted, normalised mean of its word vectors.

Two words that appear in similar contexts ("car" and "vehicle" in a corpus that talks about
both) end up with similar vectors, without any labels.

    emb = SelfEmbedder(dim=256).fit(texts)
    mem = AgentMemory("memory.db", embedder=emb)

``fit`` may be called again later (for example with new conversation text); vectors that were
stored earlier then belong to the old space, so call ``AgentMemory.reindex()`` afterwards.
"""
from __future__ import annotations

import hashlib
import math
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Optional, Sequence

from .embeddings import tokenize

_MASK = (1 << 64) - 1


def _seed_sign_index(feature: int, seed: int, dim: int):
    h = hashlib.blake2b(f"{seed}:{feature}".encode(), digest_size=8).digest()
    v = int.from_bytes(h, "little")
    return v % dim, (1.0 if (v >> 63) & 1 else -1.0)


class SelfEmbedder:
    """Dependency-free word-vector sentence embedder fitted on unlabelled text."""

    def __init__(self, dim: int = 256, window: int = 4, min_count: int = 1,
                 max_vocab: int = 50_000, seed: int = 7) -> None:
        self.dim = dim
        self.window = window
        self.min_count = min_count
        self.max_vocab = max_vocab
        self.seed = seed
        self._cooc: Dict[str, Counter] = defaultdict(Counter)
        self._df: Counter = Counter()
        self._docs = 0
        self._wordvec: Dict[str, List[float]] = {}
        self._idf: Dict[str, float] = {}
        self._fitted = False

    # ----------------------------------------------------------------- fitting
    def fit(self, texts: Iterable[str]) -> "SelfEmbedder":
        """Accumulate co-occurrence statistics from ``texts`` and rebuild word vectors."""
        for text in texts:
            toks = tokenize(text)
            if not toks:
                continue
            self._docs += 1
            self._df.update(set(toks))
            for i, w in enumerate(toks):
                lo, hi = max(0, i - self.window), min(len(toks), i + self.window + 1)
                for j in range(lo, hi):
                    if j != i:
                        self._cooc[w][toks[j]] += 1
        self._rebuild()
        return self

    def _rebuild(self) -> None:
        total = sum(sum(c.values()) for c in self._cooc.values()) or 1
        row_sum = {w: sum(c.values()) for w, c in self._cooc.items()}
        col_sum: Counter = Counter()
        for c in self._cooc.values():
            col_sum.update(c)
        vocab = [w for w, _ in self._df.most_common(self.max_vocab) if self._df[w] >= self.min_count]
        keep = set(vocab)
        self._wordvec = {}
        for w in vocab:
            vec = [0.0] * self.dim
            for c_word, n in self._cooc.get(w, {}).items():
                if c_word not in keep:
                    continue
                pmi = math.log((n * total) / (row_sum[w] * col_sum[c_word]) + 1e-12)
                if pmi > 0:
                    idx, sign = _seed_sign_index(hash_key(c_word), self.seed, self.dim)
                    vec[idx] += sign * pmi
            norm = math.sqrt(sum(x * x for x in vec))
            if norm:
                self._wordvec[w] = [x / norm for x in vec]
        n = max(self._docs, 1)
        self._idf = {w: math.log(1 + n / (1 + self._df[w])) for w in self._wordvec}
        self._fitted = True

    @property
    def vocab_size(self) -> int:
        return len(self._wordvec)

    # ---------------------------------------------------------------- embedding
    def embed(self, text: str) -> List[float]:
        if not self._fitted:
            raise RuntimeError("SelfEmbedder.fit(texts) must be called before embedding")
        acc = [0.0] * self.dim
        hit = False
        for w in tokenize(text or ""):
            v = self._wordvec.get(w)
            if v is None:
                continue
            hit = True
            weight = self._idf[w]
            for k in range(self.dim):
                acc[k] += weight * v[k]
        if not hit:
            return acc
        norm = math.sqrt(sum(x * x for x in acc))
        return [x / norm for x in acc] if norm else acc

    def embed_many(self, texts: Sequence[str]) -> List[List[float]]:
        return [self.embed(t) for t in texts]


def hash_key(word: str) -> int:
    """Stable integer key for a word (independent of Python's per-process hash seed)."""
    return int.from_bytes(hashlib.blake2b(word.encode(), digest_size=8).digest(), "little")
