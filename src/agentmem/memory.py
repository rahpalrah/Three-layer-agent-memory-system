"""``AgentMemory`` - the single entry point applications integrate with."""
from __future__ import annotations

import json
import time
import uuid
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Union

from .consolidation import ConsolidationReport, FactExtractor, RuleBasedExtractor
from .embeddings import Embedder, HashingEmbedder
from .episodic import EpisodicMemory
from .scoring import HeuristicImportanceScorer, ImportanceScorer, RetrievalWeights
from .semantic import SemanticMemory
from .storage import InMemoryStorage, SQLiteStorage, StorageBackend
from .types import Layer, MemoryItem, ScoredMemory, Session
from .working import WorkingMemory

Hook = Callable[[str, Dict[str, Any]], None]


class AgentMemory:
    """Three-layer memory for an agent.

    >>> mem = AgentMemory(path="memory.db", namespace="user-42")
    >>> with mem.session():
    ...     mem.add("My name is Ada and I prefer concise answers.", role="user")
    >>> mem.recall("what is the user's name?")   # works in any later session/process

    ``namespace`` isolates tenants (one per user, agent or user+agent pair).
    """

    def __init__(self, path: Optional[Union[str, Path]] = None, *, namespace: str = "default",
                 storage: Optional[StorageBackend] = None, embedder: Union[Embedder, str, None] = None,
                 scorer: Optional[ImportanceScorer] = None, extractor: Optional[FactExtractor] = None,
                 working_capacity: int = 20, working_token_budget: Optional[int] = 4000,
                 discard_below: float = 0.1, auto_consolidate: bool = True,
                 weights: Optional[RetrievalWeights] = None,
                 clock: Callable[[], float] = time.time) -> None:
        """``clock`` returns "now" as a UNIX timestamp; inject one to replay
        historical conversations or to simulate the passage of time in tests."""
        self.clock = clock
        if storage is None:
            storage = SQLiteStorage(path) if path else InMemoryStorage()
        self.storage = storage
        self.namespace = namespace
        if embedder == "minilm":  # recommended: semantic + hybrid search, local and free
            from .integrations.onnx import MiniLMEmbedder
            embedder = MiniLMEmbedder()
        elif embedder in (None, "hashing"):
            embedder = HashingEmbedder()
        elif isinstance(embedder, str):
            raise ValueError(f"unknown embedder {embedder!r}; use 'minilm', 'hashing' or an Embedder")
        self.embedder = embedder
        self.scorer = scorer or HeuristicImportanceScorer(embed=self.embedder.embed)
        self.extractor = extractor or RuleBasedExtractor()
        self.discard_below = discard_below
        self.auto_consolidate = auto_consolidate
        self._hooks: List[Hook] = []

        # Hybrid keyword (BM25) + vector relevance. The built-in HashingEmbedder is itself
        # keyword-based, so BM25 carries most of the weight; with a real semantic embedder an
        # even split was best on the LoCoMo dev split (benchmarks/tune.py).
        lexical = 0.8 if isinstance(self.embedder, HashingEmbedder) else 0.5
        if weights is None:
            weights = RetrievalWeights(lexical=lexical)
        self.episodic = EpisodicMemory(storage, self.embedder, namespace, weights, clock=clock)
        self.semantic = SemanticMemory(storage, self.embedder, namespace, clock=clock)
        self.semantic.weights.lexical = weights.lexical
        self.working = WorkingMemory(storage, namespace, working_capacity, working_token_budget,
                                     on_evict=self._on_evict)
        self.current_session: Optional[Session] = None

    # ------------------------------------------------------------------ hooks
    def on(self, hook: Hook) -> Hook:
        """Register ``hook(event, payload)``; events: add, evict, discard, learn,
        session_start, session_end, consolidate, forget. Usable as a decorator."""
        self._hooks.append(hook)
        return hook

    def _emit(self, event: str, **payload: Any) -> None:
        for h in self._hooks:
            try:
                h(event, payload)
            except Exception:  # hooks must never break memory
                pass

    # --------------------------------------------------------------- sessions
    def start_session(self, session_id: Optional[str] = None, **metadata: Any) -> Session:
        if self.current_session:
            self.end_session()
        existing = self.storage.get_session(session_id) if session_id else None
        if existing:  # resume an interrupted session
            existing.ended_at = None
            self.current_session = existing
            self.working.restore(existing.id)
        else:
            self.current_session = Session(id=session_id or uuid.uuid4().hex, namespace=self.namespace,
                                           metadata=metadata, started_at=self.clock())
        self.storage.put_session(self.current_session)
        self._emit("session_start", session=self.current_session)
        return self.current_session

    def end_session(self, summary: Optional[str] = None) -> Optional[ConsolidationReport]:
        s = self.current_session
        if not s:
            return None
        self.working.flush()
        report = self.consolidate(session_id=s.id) if self.auto_consolidate else None
        s.ended_at = self.clock()
        s.summary = summary or s.summary
        self.storage.put_session(s)
        self._emit("session_end", session=s, report=report)
        self.current_session = None
        return report

    @contextmanager
    def session(self, session_id: Optional[str] = None, **metadata: Any) -> Iterator[Session]:
        s = self.start_session(session_id, **metadata)
        try:
            yield s
        finally:
            self.end_session()

    def sessions(self, limit: Optional[int] = None) -> List[Session]:
        return self.storage.list_sessions(self.namespace, limit)

    # ----------------------------------------------------------------- writes
    def add(self, content: str, *, role: str = "user", importance: Optional[float] = None,
            tags: Optional[List[str]] = None, metadata: Optional[Dict[str, Any]] = None,
            pinned: bool = False) -> MemoryItem:
        """Observe something in the current conversation (goes to working memory).

        Importance is scored automatically unless given. Starts a session
        implicitly if none is active.
        """
        if not self.current_session:
            self.start_session()
        recent = self.episodic.recent(20) + self.working.items()
        imp = importance if importance is not None else self.scorer.score(content, role=role, context=recent)
        meta = dict(metadata or {}, role=role)
        now = self.clock()
        item = MemoryItem(content=content, importance=imp, namespace=self.namespace,
                          session_id=self.current_session.id, tags=list(tags or []),
                          metadata=meta, pinned=pinned, embedding=self.embedder.embed(content),
                          created_at=now, last_accessed=now)
        self.working.add(item)
        self._emit("add", item=item)
        return item

    def _on_evict(self, item: MemoryItem) -> None:
        if item.importance < self.discard_below and not item.pinned:
            self._emit("discard", item=item)
            return
        self.episodic.record(item)
        self._emit("evict", item=item)

    def remember(self, content: str, *, importance: float = 0.5, tags: Optional[List[str]] = None,
                 metadata: Optional[Dict[str, Any]] = None, pinned: bool = False) -> MemoryItem:
        """Write an episode directly (bypassing working memory), e.g. from a tool or event log."""
        now = self.clock()
        item = MemoryItem(content=content, importance=importance, tags=list(tags or []),
                          metadata=dict(metadata or {}), pinned=pinned, created_at=now, last_accessed=now,
                          session_id=self.current_session.id if self.current_session else None)
        return self.episodic.record(item)

    def learn(self, fact: str, *, importance: float = 0.7, key: Optional[str] = None,
              tags: Optional[List[str]] = None, pinned: bool = False) -> MemoryItem:
        """Write a fact straight into semantic memory."""
        item = self.semantic.learn(fact, importance=importance, key=key, tags=tags, pinned=pinned)
        self._emit("learn", item=item)
        return item

    def forget(self, item_id: str) -> bool:
        ok = self.storage.delete(item_id)
        if ok:
            self.working._items = [i for i in self.working._items if i.id != item_id]
            self._emit("forget", ids=[item_id])
        return ok

    # ------------------------------------------------------------------ reads
    def recall(self, query: Optional[str] = None, k: int = 8,
               layers: Optional[List[Layer]] = None, tags: Optional[List[str]] = None,
               min_score: float = 0.0, reinforce: bool = True, expand: int = 0) -> List[ScoredMemory]:
        """Search across layers; one joint ranking by relevance, recency and importance.

        ``expand=n`` adds the n episodes before/after each episodic hit from the same
        session (sentence-window retrieval), within the same ``k`` budget.
        """
        from .scoring import rank
        layers = layers or [Layer.WORKING, Layer.EPISODIC, Layer.SEMANTIC]
        items: List[MemoryItem] = []
        if Layer.SEMANTIC in layers:
            items += self.semantic.facts()
        if Layer.EPISODIC in layers:
            items += self.storage.list(self.namespace, Layer.EPISODIC)
        if Layer.WORKING in layers:
            items += self.working.items()
        if tags:
            items = [i for i in items if set(tags) & set(i.tags)]
        qv = self.embedder.embed(query) if query else None
        now = self.clock()
        ranked = rank(items, qv, self.episodic.weights, now=now,
                      layer_weights={Layer.SEMANTIC: self.semantic.weights}, query_text=query)
        seen, out = set(), []
        for h in ranked:
            key = h.item.content.strip().lower()
            if key in seen or h.score < min_score:
                continue
            seen.add(key)
            out.append(h)
            if len(out) >= k:
                break
        if expand > 0:
            out = self._expand(out, k, expand)
        if reinforce:
            for h in out:
                if h.item.layer != Layer.WORKING:
                    h.item.touch(now)
                    self.storage.put(h.item)
        return out

    def _expand(self, hits: List[ScoredMemory], k: int, n: int) -> List[ScoredMemory]:
        timelines: Dict[str, List[MemoryItem]] = {}
        out: List[ScoredMemory] = []
        ids = set()
        for h in hits:
            if len(out) >= k:
                break
            if h.item.id in ids:
                continue
            group = [h]
            if h.item.layer == Layer.EPISODIC and h.item.session_id:
                tl = timelines.get(h.item.session_id)
                if tl is None:
                    tl = sorted(self.storage.list(self.namespace, Layer.EPISODIC, session_id=h.item.session_id),
                                key=lambda i: i.created_at)
                    timelines[h.item.session_id] = tl
                pos = next((i for i, e in enumerate(tl) if e.id == h.item.id), None)
                if pos is not None:
                    for j in list(range(pos - n, pos)) + list(range(pos + 1, pos + n + 1)):
                        if 0 <= j < len(tl):
                            group.append(ScoredMemory(tl[j], h.score * 0.9, h.relevance, h.recency, h.importance))
            for g in group:
                if g.item.id not in ids and len(out) < k:
                    ids.add(g.item.id)
                    out.append(g)
        return out

    def build_context(self, query: Optional[str] = None, *, k_facts: int = 6, k_episodes: int = 4,
                      k_profile: int = 8, max_chars: int = 3000) -> str:
        """A compact text block to inject into an LLM system prompt.

        Sections: the core profile (keyed facts, always included), facts relevant
        to ``query``, and relevant past events.
        """
        profile = self.semantic.profile(k_profile)
        pids = {f.id for f in profile}
        facts = [h for h in self.semantic.retrieve(query, k=k_facts + len(pids)) if h.item.id not in pids][:k_facts]
        eps = self.episodic.retrieve(query, k=k_episodes)
        lines: List[str] = []
        if profile:
            lines.append("User profile:")
            lines += [f"- {f.content}" for f in profile]
        if facts:
            lines.append("Relevant facts (long-term):")
            lines += [f"- {h.item.content}" for h in facts]
        if eps:
            lines.append("Relevant past events:")
            for h in eps:
                when = time.strftime("%Y-%m-%d", time.localtime(h.item.created_at))  # noqa
                lines.append(f"- [{when}] ({h.item.metadata.get('role', 'event')}) {h.item.content}")
        text = "\n".join(lines)
        return text[:max_chars]

    # ---------------------------------------------------------- maintenance
    def consolidate(self, session_id: Optional[str] = None,
                    forget_threshold: Optional[float] = 0.1) -> ConsolidationReport:
        """Distil episodes into semantic facts, then forget faded episodes."""
        eps = self.episodic.timeline(session_id=session_id)
        eps = [e for e in eps if not e.metadata.get("consolidated")]
        report = ConsolidationReport(episodes_considered=len(eps))
        before = {f.id for f in self.semantic.facts()}
        for fact in self.extractor.extract(eps):
            item = self.semantic.learn(fact.content, importance=fact.importance, key=fact.key,
                                       source_ids=fact.source_ids, tags=fact.tags,
                                       slot=getattr(fact, "slot", None), update=getattr(fact, "update", False))
            report.fact_ids.append(item.id)
            if item.id in before:
                report.facts_reinforced += 1
            else:
                report.facts_learned += 1
                before.add(item.id)
            self._emit("learn", item=item)
        for e in eps:
            e.metadata["consolidated"] = True
            self.storage.put(e)
        if forget_threshold is not None:
            report.episodes_forgotten = self.episodic.forget(forget_threshold)
        self._emit("consolidate", report=report)
        return report

    def stats(self) -> Dict[str, Any]:
        return {
            "namespace": self.namespace,
            "working": len(self.working),
            "working_tokens": self.working.tokens,
            "episodic": self.episodic.count(),
            "semantic": self.semantic.count(),
            "sessions": len(self.sessions()),
            "current_session": self.current_session.id if self.current_session else None,
        }

    # ---------------------------------------------------------- portability
    def export(self) -> Dict[str, Any]:
        return {
            "namespace": self.namespace,
            "memories": [i.to_dict(include_embedding=False) for i in self.storage.list(self.namespace)],
            "sessions": [s.__dict__ for s in self.sessions()],
        }

    def export_json(self, path: Union[str, Path]) -> None:
        Path(path).write_text(json.dumps(self.export(), indent=2, default=str))

    def import_data(self, data: Dict[str, Any]) -> int:
        n = 0
        for d in data.get("memories", []):
            item = MemoryItem.from_dict(d)
            item.namespace = self.namespace
            item.embedding = self.embedder.embed(item.content)
            self.storage.put(item)
            n += 1
        for s in data.get("sessions", []):
            self.storage.put_session(Session(**{**s, "namespace": self.namespace}))
        return n

    def reset(self) -> None:
        self.working._items.clear()
        self.storage.clear(self.namespace)
        self.current_session = None

    def close(self) -> None:
        if self.current_session:
            self.end_session()
        self.storage.close()

    def __enter__(self) -> "AgentMemory":
        return self

    def __exit__(self, *exc) -> None:
        self.close()
