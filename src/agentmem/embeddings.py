"""Embedding backends.

The default :class:`HashingEmbedder` is dependency-free (feature hashing of
word unigrams + bigrams + character trigrams). It is good enough for keyword-ish
similarity and tests. For production, plug in any callable that maps text to a
vector, e.g. a sentence-transformers model or a hosted embeddings API.
"""
from __future__ import annotations

import hashlib
import math
import operator
import re
from typing import Callable, List, Optional, Protocol, Sequence, runtime_checkable

try:  # optional acceleration
    import numpy as _np
except ImportError:  # pragma: no cover
    _np = None

_WORD = re.compile(r"[^\W_]+(?:'[^\W_]+)*")
_CJK = re.compile(r"[\u3040-\u30ff\u3400-\u4dbf\u4e00-\u9fff\uac00-\ud7af]")
_STOP = frozenset(
    "a an the is are was were be been am i you he she it we they me my your our "
    "to of in on at for and or but with that this these those do does did have has "
    "had will would can could should so as by from what which who whom about where when why how "
    "does there their them its user user's users".split()  # "the user ..." prefixes every fact
)


def _stem(t: str) -> str:
    """Tiny suffix stripper so work/works/working/worked share a token."""
    for suf in ("ing", "ed", "es", "s"):
        if len(t) > len(suf) + 2 and t.endswith(suf) and not t.endswith("ss"):
            return t[: -len(suf)]
    return t


@runtime_checkable
class Embedder(Protocol):
    dim: int

    def embed(self, text: str) -> List[float]: ...


def tokenize(text: str) -> List[str]:
    """Unicode word tokens; scripts without spaces (CJK) become character bigrams."""
    out: List[str] = []
    for t in _WORD.findall(text.lower()):
        if _CJK.search(t):
            out += [t[i:i + 2] for i in range(max(1, len(t) - 1))]
        elif t not in _STOP:
            out.append(_stem(t))
    return out


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
    dot = sum(map(operator.mul, a, b))
    na = math.sqrt(sum(map(operator.mul, a, a)))
    nb = math.sqrt(sum(map(operator.mul, b, b)))
    return dot / (na * nb) if na and nb else 0.0


def cosine_many(query: Sequence[float], vectors: Sequence[Optional[Sequence[float]]]) -> List[float]:
    """Cosine of ``query`` against many vectors; uses numpy when installed."""
    if _np is not None and vectors:
        dim = len(query)
        ok = [i for i, v in enumerate(vectors) if v is not None and len(v) == dim]
        out = [0.0] * len(vectors)
        if ok:
            m = _np.asarray([vectors[i] for i in ok], dtype=_np.float32)
            q = _np.asarray(query, dtype=_np.float32)
            denom = _np.linalg.norm(m, axis=1) * (_np.linalg.norm(q) or 1.0)
            sims = (m @ q) / _np.where(denom == 0, 1.0, denom)
            for i, s in zip(ok, sims.tolist()):
                out[i] = s
        return out
    return [cosine(query, v) if v else 0.0 for v in vectors]
