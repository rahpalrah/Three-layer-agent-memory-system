"""Consolidation: episodic -> semantic ("sleep" for the agent).

A ``FactExtractor`` turns a batch of episodes into durable facts. The default
:class:`RuleBasedExtractor` needs no LLM; swap in an LLM-backed extractor
(see ``agentmem.integrations.anthropic``) for much richer distillation.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Protocol, Sequence

from .types import MemoryItem


@dataclass
class Fact:
    content: str
    importance: float = 0.6
    key: Optional[str] = None
    source_ids: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    slot: Optional[str] = None     # e.g. "pref:coffee" - what the fact is about
    update: bool = False           # the speaker signalled a change ("now", "anymore", ...)


class FactExtractor(Protocol):
    def extract(self, episodes: Sequence[MemoryItem]) -> List[Fact]: ...


# (regex on the user's text, template, key) - first-person facts about the user
_RULES = [
    (r"\bmy name is ((?-i:[A-Z][\w'-]*(?: [A-Z][\w'-]*)?))", "The user's name is {0}.", "user.name"),
    (r"\bcall me ((?-i:[A-Z][\w'-]*))", "The user prefers to be called {0}.", "user.name"),
    (r"\bi live in ((?-i:[A-Z][\w'-]*(?: [A-Z][\w'-]*)*))(?:[.,!?]|$| and )", "The user lives in {0}.", "user.location"),
    (r"\bi(?:'m| am) from ((?-i:[A-Z][\w'-]*(?: [A-Z][\w'-]*)*))(?:[.,!?]|$| and )", "The user is from {0}.", "user.origin"),
    (r"\bi work (?:at|for) ((?-i:[A-Z][\w&'-]*(?: [A-Z][\w&'-]*)*))", "The user works at {0}.", "user.employer"),
    (r"\bi(?:'m| am) an? ([a-z][\w -]{2,40}?)(?:[.,!?]|$| at | and )", "The user is a {0}.", "user.role"),
    (r"\bi(?: really)? (?:like|love|enjoy) ([\w ,'-]{2,60}?)(?:[.!?]|$)", "The user likes {0}.", None),
    (r"\bi prefer ([\w ,'-]{2,60}?)(?:[.!?]|$)", "The user prefers {0}.", None),
    (r"\bi (?:hate|dislike|don't like|do not like) ([\w ,'-]{2,60}?)(?:[.!?]|$)", "The user dislikes {0}.", None),
    (r"\bi(?:'m| am) allergic to ([\w ,'-]{2,40}?)(?:[.!?]|$)", "The user is allergic to {0}.", None),
    (r"\bmy favou?rite ([a-z ]{2,20}) is ([\w ,'-]{2,40}?)(?:[.!?]|$)", "The user's favourite {0} is {1}.", "user.favourite.{0}"),
    (r"\b(?:remember|don't forget|note) that ([^.!?]{4,200})", "{0}.", None),
]


_UPDATE = re.compile(r"\b(now|anymore|any more|instead|changed my mind|no longer|from now on|these days|switched)\b", re.I)
_PREPS = re.compile(r"\s+(?:for|with|in|on|at|to|from|of|because|but|and|so|when)\b.*$", re.I)
_FILLER_TAIL = re.compile(r"\b(?:now|anymore|any more|instead|these days|a lot|very much|too)\s*$", re.I)
_PREF_TEMPLATES = ("The user likes", "The user prefers", "The user dislikes")


def preference_slot(obj: str) -> Optional[str]:
    """Head noun of a preference object: 'dark roast coffee' -> 'coffee', 'whole beans for x' -> 'beans'."""
    o = _PREPS.sub("", obj.strip())
    o = _FILLER_TAIL.sub("", o).strip(" ,.")
    words = re.findall(r"[^\W_]+", o.lower())
    return f"pref:{words[-1]}" if words else None


_PRONOUNS = [(r"\bi am\b|\bi'm\b", "the user is"), (r"\bmy\b", "the user's"),
             (r"\bmine\b", "the user's"), (r"\bme\b", "the user"), (r"\bi\b", "the user")]


def third_person(text: str) -> str:
    """Rewrite first-person text about the user into a third-person fact."""
    for pat, rep in _PRONOUNS:
        text = re.sub(pat, rep, text, flags=re.I)
    return text[:1].upper() + text[1:]


class RuleBasedExtractor:
    """Pattern-based extraction + promotion of highly important episodes."""

    def __init__(self, promote_threshold: float = 0.8) -> None:
        self.promote_threshold = promote_threshold
        self._rules = [(re.compile(p, re.I), t, k) for p, t, k in _RULES]

    def extract(self, episodes: Sequence[MemoryItem]) -> List[Fact]:
        facts: List[Fact] = []
        for ep in episodes:
            role = ep.metadata.get("role")
            matched = False
            if role in (None, "user"):
                for rx, tmpl, key in self._rules:
                    for m in rx.finditer(ep.content):
                        groups = [g.strip() for g in m.groups()]
                        k = key.format(*[g.lower().replace(" ", "_") for g in groups]) if key else None
                        text = tmpl.format(*groups)
                        if tmpl == "{0}.":
                            text = third_person(text)
                        slot = preference_slot(groups[0]) if tmpl.startswith(_PREF_TEMPLATES) else None
                        facts.append(Fact(text, importance=max(0.6, ep.importance), key=k,
                                          source_ids=[ep.id], tags=list(ep.tags), slot=slot,
                                          update=bool(_UPDATE.search(ep.content))))
                        matched = True
            if not matched and ep.importance >= self.promote_threshold:
                facts.append(Fact(ep.content, importance=ep.importance, source_ids=[ep.id], tags=list(ep.tags)))
        return facts


class CallableExtractor:
    """Wrap ``fn(episodes) -> list[Fact | str]`` (e.g. an LLM call)."""

    def __init__(self, fn, fallback: Optional[FactExtractor] = None) -> None:
        self.fn = fn
        self.fallback = fallback or RuleBasedExtractor()

    def extract(self, episodes: Sequence[MemoryItem]) -> List[Fact]:
        try:
            out = self.fn(list(episodes))
        except Exception:
            return self.fallback.extract(episodes)
        ids = [e.id for e in episodes]
        return [f if isinstance(f, Fact) else Fact(str(f), source_ids=ids) for f in out]


@dataclass
class ConsolidationReport:
    episodes_considered: int = 0
    facts_learned: int = 0
    facts_reinforced: int = 0
    episodes_forgotten: int = 0
    fact_ids: List[str] = field(default_factory=list)
