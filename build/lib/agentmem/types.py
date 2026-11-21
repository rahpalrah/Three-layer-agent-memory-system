"""Core data types shared by every memory layer."""
from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class Layer(str, Enum):
    WORKING = "working"
    EPISODIC = "episodic"
    SEMANTIC = "semantic"


def _new_id() -> str:
    return uuid.uuid4().hex


@dataclass
class MemoryItem:
    """A single unit of memory.

    ``importance`` is in [0, 1]. ``pinned`` items are never forgotten.
    ``metadata`` is free-form and JSON-serialisable (role, source, evidence...).
    """

    content: str
    layer: Layer = Layer.EPISODIC
    importance: float = 0.5
    namespace: str = "default"
    session_id: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    pinned: bool = False
    access_count: int = 0
    id: str = field(default_factory=_new_id)
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)

    def touch(self, now: Optional[float] = None) -> None:
        """Mark as accessed (retrieval reinforces memory)."""
        self.access_count += 1
        self.last_accessed = now if now is not None else time.time()

    def to_dict(self, include_embedding: bool = False) -> Dict[str, Any]:
        d = asdict(self)
        d["layer"] = self.layer.value
        if not include_embedding:
            d.pop("embedding", None)
        return d

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "MemoryItem":
        d = dict(d)
        d["layer"] = Layer(d.get("layer", "episodic"))
        return cls(**d)


@dataclass
class ScoredMemory:
    """A retrieval hit with its score breakdown (useful for debugging/tuning)."""

    item: MemoryItem
    score: float
    relevance: float = 0.0
    recency: float = 0.0
    importance: float = 0.0

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"ScoredMemory(score={self.score:.3f}, layer={self.item.layer.value}, content={self.item.content[:60]!r})"


@dataclass
class Session:
    id: str
    namespace: str = "default"
    started_at: float = field(default_factory=time.time)
    ended_at: Optional[float] = None
    summary: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
