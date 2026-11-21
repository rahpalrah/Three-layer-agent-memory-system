"""Claude integration: a memory-backed chat agent plus LLM-powered scoring
and fact extraction.  Requires ``pip install agentmem[anthropic]``.

    from anthropic import Anthropic
    from agentmem import AgentMemory
    from agentmem.integrations.anthropic import MemoryAgent

    mem = AgentMemory("memory.db", namespace="user-42")
    agent = MemoryAgent(Anthropic(), mem)
    print(agent.chat("Hi, I'm Ada and I'm vegetarian."))
"""
from __future__ import annotations

import json
import re
from typing import Any, List, Optional, Sequence

from ..consolidation import CallableExtractor, Fact
from ..memory import AgentMemory
from ..scoring import CallableImportanceScorer
from ..types import MemoryItem

DEFAULT_MODEL = "claude-sonnet-5-5"


def _text(resp: Any) -> str:
    return "".join(getattr(b, "text", "") for b in resp.content if getattr(b, "type", "") == "text")


def _alternate(messages: List[dict]) -> List[dict]:
    """Claude requires user-first, alternating roles: merge/trim to comply."""
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


def claude_importance_scorer(client: Any, model: str = DEFAULT_MODEL) -> CallableImportanceScorer:
    def fn(text: str, role: Optional[str]) -> float:
        resp = client.messages.create(
            model=model, max_tokens=8,
            system=("Rate how important it is for a personal assistant to remember this message "
                    "long-term, from 1 (trivial chit-chat) to 10 (core fact, preference, commitment). "
                    "Reply with the number only."),
            messages=[{"role": "user", "content": f"[{role or 'user'}] {text}"}],
        )
        m = re.search(r"\d+(\.\d+)?", _text(resp))
        return (float(m.group()) - 1) / 9 if m else 0.5
    return CallableImportanceScorer(fn)


def claude_fact_extractor(client: Any, model: str = DEFAULT_MODEL) -> CallableExtractor:
    def fn(episodes: Sequence[MemoryItem]) -> List[Fact]:
        if not episodes:
            return []
        transcript = "\n".join(f"[{e.id}] {e.metadata.get('role', 'event')}: {e.content}" for e in episodes)
        resp = client.messages.create(
            model=model, max_tokens=1500,
            system=("Extract durable facts worth remembering across conversations (identity, "
                    "preferences, goals, relationships, decisions, constraints). Skip small talk and "
                    "anything only relevant to this moment. Return ONLY a JSON array of objects: "
                    '{"fact": str, "importance": 0-1, "key": optional dotted key like "user.city" for '
                    'single-valued attributes, "sources": [episode ids]}. Return [] if nothing qualifies.'),
            messages=[{"role": "user", "content": transcript}],
        )
        raw = _text(resp)
        m = re.search(r"\[.*\]", raw, re.S)
        data = json.loads(m.group()) if m else []
        return [Fact(d["fact"], float(d.get("importance", 0.6)), d.get("key"),
                     list(d.get("sources", []))) for d in data if d.get("fact")]
    return CallableExtractor(fn)


class MemoryAgent:
    """Chat loop that reads from and writes to all three memory layers."""

    def __init__(self, client: Any, memory: AgentMemory, model: str = DEFAULT_MODEL,
                 system: str = "You are a helpful assistant with long-term memory.",
                 max_tokens: int = 1024, use_llm_memory: bool = False) -> None:
        self.client = client
        self.memory = memory
        self.model = model
        self.system = system
        self.max_tokens = max_tokens
        if use_llm_memory:
            memory.scorer = claude_importance_scorer(client, model)
            memory.extractor = claude_fact_extractor(client, model)

    def chat(self, user_message: str, **create_kwargs: Any) -> str:
        mem = self.memory
        context = mem.build_context(user_message)
        mem.add(user_message, role="user")
        system = self.system
        if context:
            system += ("\n\n<memory>\n" + context + "\n</memory>\n"
                       "Use these memories when relevant; don't mention the memory system unless asked.")
        resp = self.client.messages.create(
            model=self.model, max_tokens=self.max_tokens, system=system,
            messages=_alternate(mem.working.as_messages()), **create_kwargs)
        reply = _text(resp)
        mem.add(reply, role="assistant")
        return reply
