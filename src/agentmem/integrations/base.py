"""Provider-agnostic pieces shared by every LLM integration.

To support a new model provider you only need ``complete(system, messages) -> str``.
"""
from __future__ import annotations

import json
import re
from typing import Callable, List, Optional, Sequence

from ..consolidation import CallableExtractor, Fact
from ..memory import AgentMemory
from ..scoring import CallableImportanceScorer
from ..types import MemoryItem

Complete = Callable[[str, List[dict]], str]  # (system, messages) -> assistant text

IMPORTANCE_PROMPT = (
    "Rate how important it is for a personal assistant to remember this message long-term, "
    "from 1 (trivial chit-chat) to 10 (core fact, preference, commitment). Reply with the number only.")

EXTRACT_PROMPT = (
    "Extract durable facts worth remembering across conversations (identity, preferences, goals, "
    "relationships, decisions, constraints, open issues). Skip small talk and anything only relevant "
    "to this moment. Return ONLY a JSON array of objects: "
    '{"fact": str, "importance": 0-1, "key": optional dotted key like "user.city" for single-valued '
    'attributes, "sources": [episode ids]}. Return [] if nothing qualifies.')


def alternate(messages: List[dict]) -> List[dict]:
    """Most chat APIs want user-first, strictly alternating roles: merge/trim to comply."""
    out: List[dict] = []
    for m in messages:
        role = "assistant" if m["role"] == "assistant" else "user"
        if out and out[-1]["role"] == role:
            out[-1]["content"] += "\n\n" + m["content"]
        else:
            out.append({"role": role, "content": m["content"]})
    while out and out[0]["role"] != "user":
        out.pop(0)
    return out


def importance_scorer(complete: Complete) -> CallableImportanceScorer:
    def fn(text: str, role: Optional[str]) -> float:
        reply = complete(IMPORTANCE_PROMPT, [{"role": "user", "content": f"[{role or 'user'}] {text}"}])
        m = re.search(r"\d+(\.\d+)?", reply)
        return (min(10.0, max(1.0, float(m.group()))) - 1) / 9 if m else 0.5
    return CallableImportanceScorer(fn)


def parse_facts(raw: str) -> List[Fact]:
    m = re.search(r"\[.*\]", raw, re.S)
    data = json.loads(m.group()) if m else []
    return [Fact(d["fact"], float(d.get("importance", 0.6)), d.get("key"), list(d.get("sources", [])))
            for d in data if isinstance(d, dict) and d.get("fact")]


def fact_extractor(complete: Complete) -> CallableExtractor:
    def fn(episodes: Sequence[MemoryItem]) -> List[Fact]:
        if not episodes:
            return []
        transcript = "\n".join(f"[{e.id}] {e.metadata.get('role', 'event')}: {e.content}" for e in episodes)
        return parse_facts(complete(EXTRACT_PROMPT, [{"role": "user", "content": transcript}]))
    return CallableExtractor(fn)


class ChatAgent:
    """Memory-backed chat loop over any ``complete(system, messages)`` function."""

    def __init__(self, complete: Complete, memory: AgentMemory,
                 system: str = "You are a helpful assistant with long-term memory.",
                 use_llm_memory: bool = False) -> None:
        self.complete = complete
        self.memory = memory
        self.system = system
        if use_llm_memory:
            memory.scorer = importance_scorer(complete)
            memory.extractor = fact_extractor(complete)

    def build_system(self, user_message: str) -> str:
        context = self.memory.build_context(user_message)
        if not context:
            return self.system
        return (self.system + "\n\n<memory>\n" + context + "\n</memory>\n"
                "Use these memories when relevant; don't mention the memory system unless asked.")

    def chat(self, user_message: str) -> str:
        system = self.build_system(user_message)
        self.memory.add(user_message, role="user")
        reply = self.complete(system, alternate(self.memory.working.as_messages()))
        self.memory.add(reply, role="assistant")
        return reply
