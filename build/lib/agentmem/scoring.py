"""Importance scoring and retrieval ranking.

Importance answers "how much should this be remembered?" at write time.
Retrieval score answers "how useful is this memory *now*, for *this* query?"
and combines relevance, recency and (decayed) importance, in the spirit of
Park et al., *Generative Agents* (2023).
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Optional, Protocol, Sequence

from .embeddings import cosine, cosine_many
from .types import MemoryItem


class ImportanceScorer(Protocol):
    def score(self, text: str, *, role: Optional[str] = None,
              context: Sequence[MemoryItem] = ()) -> float: ...


_SIGNALS = [
    # (pattern, weight) - explicit memory requests and durable personal facts
    (r"\b(remember|don't forget|do not forget|note that|keep in mind)\b", 0.35),
    (r"\b(my name is|i am|i'm|i work|i live|my (wife|husband|partner|son|daughter|kid|dog|cat|team|company|boss))\b", 0.25),
    (r"\b(i (really )?(like|love|prefer|hate|dislike|want|need)|favou?rite)\b", 0.2),
    (r"\b(always|never|must|important|critical|urgent|deadline|allergic|password|budget)\b", 0.2),
    (r"\b(decided|agreed|plan|goal|promise|committed)\b", 0.15),
    (r"\b(19|20)\d{2}\b|\b\d{1,2}[/:-]\d{1,2}\b|\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday|tomorrow)\b", 0.1),
    (r"\$\s?\d|\b\d+(\.\d+)?\s?(%|k|m|usd|eur|kg|km)\b", 0.1),
]
_TRIVIAL = re.compile(r"^(?:\s*(?:ok(?:ay)?|thanks?(?: you)?|thank you|thx|ty|hi|hello|hey|yes|yep|no|nope|sure|cool|great|nice|lol|bye|got it)[.!?, ]*)+$", re.I)


class HeuristicImportanceScorer:
    """Fast, dependency-free scorer combining lexical signals, role and novelty."""

    def __init__(self, base: float = 0.25, role_weights: Optional[dict] = None,
                 novelty_weight: float = 0.15, embed: Optional[Callable[[str], list]] = None) -> None:
        self.base = base
        self.role_weights = role_weights or {"user": 0.05, "assistant": -0.05, "system": 0.1, "tool": -0.1}
        self.novelty_weight = novelty_weight
        self.embed = embed
        self._patterns = [(re.compile(p, re.I), w) for p, w in _SIGNALS]

    def score(self, text: str, *, role: Optional[str] = None,
              context: Sequence[MemoryItem] = ()) -> float:
        if not text.strip() or _TRIVIAL.match(text):
            return 0.05
        s = self.base
        s += sum(w for p, w in self._patterns if p.search(text))
        s += self.role_weights.get(role or "", 0.0)
        s += min(len(text) / 2000.0, 0.1)  # substance, capped
        if text.count("!") >= 2:
            s += 0.05
        if self.embed and context:
            v = self.embed(text)
            sims = [cosine(v, m.embedding) for m in context if m.embedding]
            if sims:
                s += self.novelty_weight * (1.0 - max(sims))
        return max(0.0, min(1.0, s))


class CallableImportanceScorer:
    """Delegate to any ``fn(text, role) -> float`` (e.g. an LLM rating 1-10)."""

    def __init__(self, fn: Callable[[str, Optional[str]], float],
                 fallback: Optional[ImportanceScorer] = None) -> None:
        self.fn = fn
        self.fallback = fallback or HeuristicImportanceScorer()

    def score(self, text: str, *, role: Optional[str] = None,
              context: Sequence[MemoryItem] = ()) -> float:
        try:
            return max(0.0, min(1.0, float(self.fn(text, role))))
        except Exception:
            return self.fallback.score(text, role=role, context=context)


@dataclass
class RetrievalWeights:
    relevance: float = 1.0
    recency: float = 0.5
    importance: float = 0.7
    recency_half_life_hours: float = 72.0
    importance_half_life_days: float = 30.0   # importance decays unless reinforced
    access_boost: float = 0.03                # per past retrieval, capped
    min_relevance: float = 0.0                # hard filter on raw cosine when a query is given
    relevance_gate: float = 0.75              # 0 = additive (Generative Agents); 1 = recency/importance
                                              # only count in proportion to relevance


def recency_score(item: MemoryItem, now: float, half_life_hours: float) -> float:
    age_h = max(0.0, (now - item.last_accessed) / 3600.0)
    return 0.5 ** (age_h / half_life_hours) if half_life_hours > 0 else 1.0


def effective_importance(item: MemoryItem, now: float, w: RetrievalWeights) -> float:
    """Importance after time decay, plus reinforcement from repeated access."""
    if item.pinned:
        return 1.0
    age_d = max(0.0, (now - item.created_at) / 86400.0)
    decay = 0.5 ** (age_d / w.importance_half_life_days) if w.importance_half_life_days > 0 else 1.0
    boost = min(item.access_count * w.access_boost, 0.3)
    return max(0.0, min(1.0, item.importance * (0.5 + 0.5 * decay) + boost))


def rank(items: Iterable[MemoryItem], query_vec: Optional[Sequence[float]],
         weights: RetrievalWeights, now: Optional[float] = None,
         layer_weights: Optional[Dict[object, RetrievalWeights]] = None):
    """Score items for a query.

    Relevance is min-max normalised across the candidate set (so the score
    scale doesn't depend on the embedder), and recency/importance are scaled by
    ``relevance_gate`` so an important-but-unrelated memory can't outrank a
    relevant one. With no query, ranking is by recency and importance only.
    ``layer_weights`` lets each layer use its own decay settings.
    """
    from .types import ScoredMemory
    now = now or time.time()
    items = list(items)
    raw = cosine_many(query_vec, [i.embedding for i in items]) if query_vec is not None else [0.0] * len(items)
    raw = [max(r, 0.0) for r in raw]
    hi = max(raw, default=0.0)
    out = []
    for it, rel in zip(items, raw):
        w = (layer_weights or {}).get(it.layer, weights)
        if query_vec is not None and rel < w.min_relevance:
            continue
        rec = recency_score(it, now, w.recency_half_life_hours)
        imp = effective_importance(it, now, w)
        if query_vec is not None:
            rel_n = rel / hi if hi > 0 else 0.0
            gate = (1 - w.relevance_gate) + w.relevance_gate * rel_n
            total = (w.relevance + w.recency + w.importance) or 1.0
            score = (w.relevance * rel_n + gate * (w.recency * rec + w.importance * imp)) / total
        else:
            total = (w.recency + w.importance) or 1.0
            score = (w.recency * rec + w.importance * imp) / total
        out.append(ScoredMemory(it, score, rel, rec, imp))
    out.sort(key=lambda s: s.score, reverse=True)
    return out


def retention_score(item: MemoryItem, now: float, w: RetrievalWeights) -> float:
    """Used by forgetting: blend of decayed importance and recency."""
    return 0.7 * effective_importance(item, now, w) + 0.3 * recency_score(item, now, w.recency_half_life_hours)


__all__ = [
    "ImportanceScorer", "HeuristicImportanceScorer", "CallableImportanceScorer",
    "RetrievalWeights", "rank", "recency_score", "effective_importance", "retention_score",
]
