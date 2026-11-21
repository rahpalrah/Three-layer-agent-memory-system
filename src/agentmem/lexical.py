"""BM25 keyword scoring over a candidate set, used for hybrid retrieval.

Dense similarity alone over-weights words that appear everywhere (speaker
names, "user", the domain's vocabulary). BM25's inverse document frequency
discounts them. Token statistics are cached per memory id + content.
"""
from __future__ import annotations

import math
from collections import Counter, OrderedDict
from typing import Dict, List, Sequence, Tuple

from .embeddings import tokenize
from .types import MemoryItem

_CACHE: "OrderedDict[Tuple[str, int], Tuple[Counter, int]]" = OrderedDict()
_CACHE_MAX = 200_000


def _stats(item: MemoryItem) -> Tuple[Counter, int]:
    key = (item.id, hash(item.content))
    hit = _CACHE.get(key)
    if hit is None:
        toks = tokenize(item.content)
        hit = (Counter(toks), len(toks))
        _CACHE[key] = hit
        if len(_CACHE) > _CACHE_MAX:
            _CACHE.popitem(last=False)
    return hit


def bm25_scores(query: str, items: Sequence[MemoryItem], k1: float = 1.5, b: float = 0.75) -> List[float]:
    q = list(dict.fromkeys(tokenize(query or "")))
    if not q or not items:
        return [0.0] * len(items)
    stats = [_stats(i) for i in items]
    n = len(items)
    avg = (sum(l for _, l in stats) / n) or 1.0
    df: Dict[str, int] = {w: sum(1 for tf, _ in stats if w in tf) for w in q}
    idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items() if c}
    out = []
    for tf, ln in stats:
        s = 0.0
        for w, wi in idf.items():
            f = tf.get(w)
            if f:
                s += wi * f * (k1 + 1) / (f + k1 * (1 - b + b * ln / avg))
        out.append(s)
    return out
