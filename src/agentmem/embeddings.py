"""Embedding backends.

The default :class:`HashingEmbedder` is dependency-free (feature hashing of
word unigrams + bigrams + character trigrams). It is good enough for keyword-ish
similarity and tests. For production, plug in any callable that maps text to a
vector, e.g. a sentence-transformers model or a hosted embeddings API.
"""
from __future__ import annotations

import hashlib
import math
import re
from typing import Callable, List, Protocol, Sequence, runtime_checkable

_WORD = re.compile(r"[a-z0-9']+")
_STOP = frozenset(
    "a an the is are was were be been am i you he she it we they me my your our "
    "to of in on at for and or but with that this these those do does did have has "
    "had will would can could should so as by from what which who about".split()
)


@runtime_checkable
class Embedder(Protocol):
    dim: int

    def embed(self, text: str) -> List[float]: ...


def tokenize(text: str) -> List[str]:
    return [t for t in _WORD.findall(text.lower()) if t not in _STOP]


class HashingEmbedder:
    def __init__(self, dim: int = 512) -> None:
        self.dim = dim

    def _bucket(self, feature: str) -> tuple:
        h = int.from_bytes(hashlib.blake2b(feature.encode(), digest_size=8).digest(), "little")
        return h % self.dim, (1.0 if (h >> 63) & 1 else -1.0)

    def embed(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        toks = tokenize(text)
        feats: List[tuple] = [(t, 1.0) for t in toks]
        feats += [(f"{a}_{b}", 0.7) for a, b in zip(toks, toks[1:])]
        for t in toks:
            # light stemming robustness via char trigrams
            padded = f"#{t}#"
            feats += [(f"c:{padded[i:i+3]}", 0.25) for i in range(len(padded) - 2)]
        for f, w in feats:
            idx, sign = self._bucket(f)
            vec[idx] += sign * w
        return normalize(vec)


class CallableEmbedder:
    """Wrap any ``fn(text) -> vector`` (OpenAI, Voyage, sentence-transformers...)."""

    def __init__(self, fn: Callable[[str], Sequence[float]], dim: int) -> None:
        self.fn = fn
        self.dim = dim

    def embed(self, text: str) -> List[float]:
        return normalize(list(self.fn(text)))


def normalize(vec: List[float]) -> List[float]:
    n = math.sqrt(sum(x * x for x in vec))
    return [x / n for x in vec] if n else vec


def cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0
