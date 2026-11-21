"""Claude integration. Requires ``pip install agentmem[anthropic]``.

    from anthropic import Anthropic
    from agentmem import AgentMemory
    from agentmem.integrations.anthropic import MemoryAgent

    mem = AgentMemory("memory.db", namespace="user-42")
    agent = MemoryAgent(Anthropic(), mem)
    print(agent.chat("Hi, I'm Ada and I'm vegetarian."))
"""
from __future__ import annotations

from typing import Any, List

from ..memory import AgentMemory
from .base import ChatAgent, alternate as _alternate, fact_extractor, importance_scorer

DEFAULT_MODEL = "claude-sonnet-5-5"


def _text(resp: Any) -> str:
    return "".join(getattr(b, "text", "") for b in resp.content if getattr(b, "type", "") == "text")


def claude_complete(client: Any, model: str = DEFAULT_MODEL, max_tokens: int = 1024, **kwargs: Any):
    def complete(system: str, messages: List[dict]) -> str:
        return _text(client.messages.create(model=model, max_tokens=max_tokens, system=system,
                                            messages=messages, **kwargs))
    return complete


def claude_importance_scorer(client: Any, model: str = DEFAULT_MODEL):
    return importance_scorer(claude_complete(client, model, max_tokens=8))


def claude_fact_extractor(client: Any, model: str = DEFAULT_MODEL):
    return fact_extractor(claude_complete(client, model, max_tokens=1500))


class MemoryAgent(ChatAgent):
    """Chat with Claude using all three memory layers."""

    def __init__(self, client: Any, memory: AgentMemory, model: str = DEFAULT_MODEL,
                 system: str = "You are a helpful assistant with long-term memory.",
                 max_tokens: int = 1024, use_llm_memory: bool = False) -> None:
        super().__init__(claude_complete(client, model, max_tokens), memory, system)
        self.client, self.model = client, model
        if use_llm_memory:
            memory.scorer = claude_importance_scorer(client, model)
            memory.extractor = claude_fact_extractor(client, model)


__all__ = ["MemoryAgent", "claude_complete", "claude_importance_scorer", "claude_fact_extractor",
           "_alternate", "DEFAULT_MODEL"]
