"""Free, local LLMs via Ollama (https://ollama.com). No extra Python dependencies.

    ollama pull llama3.2 && ollama pull nomic-embed-text

    from agentmem import AgentMemory
    from agentmem.integrations.ollama import OllamaAgent, OllamaEmbedder

    mem = AgentMemory("memory.db", namespace="user-42", embedder=OllamaEmbedder())
    agent = OllamaAgent(mem, model="llama3.2", use_llm_memory=True)
    print(agent.chat("Hi, I'm Ada and I'm vegetarian."))
"""
from __future__ import annotations

import json
import os
import urllib.request
from typing import Any, Dict, List, Optional

from ..embeddings import normalize
from ..memory import AgentMemory
from .base import ChatAgent

DEFAULT_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
DEFAULT_MODEL = "llama3.2"
DEFAULT_EMBED_MODEL = "nomic-embed-text"


class OllamaError(RuntimeError):
    pass


class OllamaClient:
    def __init__(self, host: str = DEFAULT_HOST, timeout: float = 120.0) -> None:
        if not host.startswith("http"):
            host = "http://" + host
        self.host = host.rstrip("/")
        self.timeout = timeout

    def _post(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        req = urllib.request.Request(self.host + path, data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            raise OllamaError(f"Ollama {path} -> HTTP {e.code}: {e.read()[:300]!r}") from e
        except urllib.error.URLError as e:
            raise OllamaError(f"Cannot reach Ollama at {self.host} ({e.reason}). Is `ollama serve` running?") from e

    def chat(self, model: str, messages: List[dict], options: Optional[dict] = None) -> str:
        data = self._post("/api/chat", {"model": model, "messages": messages, "stream": False,
                                        "options": options or {}})
        return data.get("message", {}).get("content", "")

    def embed(self, model: str, text: str) -> List[float]:
        data = self._post("/api/embed", {"model": model, "input": text})
        return data["embeddings"][0]


def ollama_complete(model: str = DEFAULT_MODEL, client: Optional[OllamaClient] = None,
                    options: Optional[dict] = None):
    client = client or OllamaClient()

    def complete(system: str, messages: List[dict]) -> str:
        return client.chat(model, [{"role": "system", "content": system}] + messages, options)
    return complete


class OllamaEmbedder:
    """Semantic embeddings from a local model (e.g. nomic-embed-text, mxbai-embed-large)."""

    def __init__(self, model: str = DEFAULT_EMBED_MODEL, client: Optional[OllamaClient] = None) -> None:
        self.model = model
        self.client = client or OllamaClient()
        self.dim = len(self.client.embed(model, "dimension probe"))

    def embed(self, text: str) -> List[float]:
        return normalize(self.client.embed(self.model, text or " "))


class OllamaAgent(ChatAgent):
    def __init__(self, memory: AgentMemory, model: str = DEFAULT_MODEL,
                 client: Optional[OllamaClient] = None,
                 system: str = "You are a helpful assistant with long-term memory.",
                 use_llm_memory: bool = False, options: Optional[dict] = None) -> None:
        super().__init__(ollama_complete(model, client, options), memory, system, use_llm_memory)
