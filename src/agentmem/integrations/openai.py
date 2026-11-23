"""OpenAI integration: embeddings and chat models, standard library only.

    export OPENAI_API_KEY=sk-...        # or let the environment's proxy inject it
    from agentmem import AgentMemory
    from agentmem.integrations.openai import OpenAIEmbedder, OpenAIAgent

    mem = AgentMemory("memory.db", namespace="user-42", embedder=OpenAIEmbedder())
    agent = OpenAIAgent(mem, model="gpt-4.1-mini")

The key is read from OPENAI_API_KEY. If it is unset, the placeholder "proxy-injected" is sent and
a network proxy that holds the real credential may replace the header. Nothing is logged.
"""
from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any, List, Optional, Sequence

from ..embeddings import normalize
from .base import ChatAgent

BASE_URL = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
DEFAULT_CHAT_MODEL = "gpt-4.1-mini"
DEFAULT_EMBED_MODEL = "text-embedding-3-small"


class OpenAIError(RuntimeError):
    pass


def _key() -> str:
    return os.environ.get("OPENAI_API_KEY") or "proxy-injected"


def _post(path: str, payload: dict, timeout: float = 120.0) -> dict:
    req = urllib.request.Request(
        BASE_URL + path, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {_key()}"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read())
    except urllib.error.HTTPError as e:
        # error bodies name the problem (bad model, quota) but never echo the key
        raise OpenAIError(f"OpenAI {path} -> HTTP {e.code}: {e.read()[:300]!r}") from e
    except urllib.error.URLError as e:
        raise OpenAIError(f"Cannot reach {BASE_URL}: {e.reason}") from e


class OpenAIEmbedder:
    """Semantic embeddings from OpenAI (text-embedding-3-small: 1536 dims, or fewer via ``dim``)."""

    def __init__(self, model: str = DEFAULT_EMBED_MODEL, dim: Optional[int] = None,
                 batch_size: int = 256) -> None:
        self.model = model
        self.dim_arg = dim
        self.batch_size = batch_size
        self.dim = dim or 1536
        self._cache: dict = {}

    def embed_many(self, texts: Sequence[str]) -> List[List[float]]:
        out: List[List[float]] = []
        for i in range(0, len(texts), self.batch_size):
            batch = [t if t.strip() else " " for t in texts[i:i + self.batch_size]]
            payload: dict = {"model": self.model, "input": batch}
            if self.dim_arg:
                payload["dimensions"] = self.dim_arg
            data = _post("/embeddings", payload)
            rows = sorted(data["data"], key=lambda r: r["index"])
            out += [normalize(r["embedding"]) for r in rows]
        return out

    def embed(self, text: str) -> List[float]:
        v = self._cache.get(text)
        if v is None:
            v = self.embed_many([text])[0]
            if len(self._cache) < 10_000:
                self._cache[text] = v
        return v


def openai_complete(model: str = DEFAULT_CHAT_MODEL, max_tokens: int = 1024, temperature: float = 0.0):
    """Returns ``complete(system, messages) -> str`` for any ChatAgent or benchmark."""
    def complete(system: str, messages: List[dict]) -> str:
        msgs = ([{"role": "system", "content": system}] if system else []) + messages
        payload = {"model": model, "messages": msgs, "max_completion_tokens": max_tokens}
        if not model.startswith(("o1", "o3", "o4", "gpt-5")):  # reasoning models ignore temperature
            payload["temperature"] = temperature
        data = _post("/chat/completions", payload)
        return data["choices"][0]["message"].get("content") or ""
    return complete


class OpenAIAgent(ChatAgent):
    """Memory-backed chat with an OpenAI model."""

    def __init__(self, memory, model: str = DEFAULT_CHAT_MODEL, max_tokens: int = 1024,
                 system: str = "You are a helpful assistant with long-term memory.",
                 use_llm_memory: bool = False) -> None:
        super().__init__(openai_complete(model, max_tokens), memory, system, use_llm_memory)
        self.model = model
