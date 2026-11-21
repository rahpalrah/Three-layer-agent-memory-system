"""Ollama adapter against a local HTTP server that mimics Ollama's API."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from agentmem import AgentMemory
from agentmem.integrations.ollama import OllamaAgent, OllamaClient, OllamaEmbedder, OllamaError

VOCAB = ["vegetarian", "recipe", "dinner", "coffee", "name", "ada", "food"]


class FakeOllama(BaseHTTPRequestHandler):
    requests = []

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeOllama.requests.append((self.path, body))
        if self.path == "/api/embed":
            t = body["input"].lower()
            out = {"embeddings": [[float(w in t) + 0.01 for w in VOCAB]]}
        elif self.path == "/api/chat":
            system = body["messages"][0]["content"]
            last = body["messages"][-1]["content"]
            if system.startswith("Rate how important"):
                text = "9" if "vegetarian" in last else "2"
            elif system.startswith("Extract durable facts"):
                eid = last.split("]")[0][1:]
                text = json.dumps([{"fact": "The user is vegetarian.", "importance": 0.9,
                                    "key": "user.diet", "sources": [eid]}])
            else:
                text = "Try a mushroom risotto." if "vegetarian" in system else "Hello!"
            out = {"model": body["model"], "message": {"role": "assistant", "content": text}, "done": True}
        else:
            self.send_response(404); self.end_headers(); return
        data = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture
def server():
    FakeOllama.requests = []
    srv = HTTPServer(("127.0.0.1", 0), FakeOllama)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def test_ollama_agent_end_to_end(server, tmp_path):
    client = OllamaClient(server)
    emb = OllamaEmbedder(client=client)
    assert emb.dim == len(VOCAB)
    db = tmp_path / "m.db"
    mem = AgentMemory(db, namespace="u", embedder=emb)
    with mem.session():
        assert OllamaAgent(mem, client=client, use_llm_memory=True).chat("Hi, I'm vegetarian.") == "Hello!"
    mem.close()

    mem2 = AgentMemory(db, namespace="u", embedder=emb)
    with mem2.session():
        reply = OllamaAgent(mem2, client=client, use_llm_memory=True).chat("Suggest a dinner recipe")
    assert reply == "Try a mushroom risotto."
    assert mem2.semantic.get_by_key("user.diet").content == "The user is vegetarian."
    chats = [b for p, b in FakeOllama.requests if p == "/api/chat" and b["messages"][0]["content"].startswith("You are")]
    assert chats[-1]["messages"][0]["role"] == "system" and chats[-1]["stream"] is False
    assert "<memory>" in chats[-1]["messages"][0]["content"]


def test_ollama_unreachable_gives_clear_error():
    with pytest.raises(OllamaError, match="Is `ollama serve` running"):
        OllamaClient("http://127.0.0.1:1", timeout=2).chat("llama3.2", [{"role": "user", "content": "hi"}])
