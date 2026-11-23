"""OpenAI integration against a local mock of the OpenAI HTTP API (no key, no network)."""
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from agentmem import AgentMemory
from agentmem.integrations import openai as O


class FakeOpenAI(BaseHTTPRequestHandler):
    seen = []

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeOpenAI.seen.append((self.path, self.headers.get("Authorization"), body))
        if self.path == "/v1/embeddings":
            vocab = ["vegetarian", "recipe", "dinner", "car", "vehicle"]
            inputs = body["input"] if isinstance(body["input"], list) else [body["input"]]
            data = [{"index": i, "embedding": [float(w in t.lower()) + 0.01 for w in vocab]}
                    for i, t in enumerate(inputs)]
            out = {"data": list(reversed(data)), "model": body["model"]}  # out of order on purpose
        elif self.path == "/v1/chat/completions":
            text = "Try a mushroom risotto." if "vegetarian" in json.dumps(body["messages"]) else "Hello!"
            out = {"choices": [{"message": {"role": "assistant", "content": text}}]}
        else:
            self.send_response(404); self.end_headers(); return
        data = json.dumps(out).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture
def server(monkeypatch):
    FakeOpenAI.seen = []
    srv = HTTPServer(("127.0.0.1", 0), FakeOpenAI)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    monkeypatch.setattr(O, "BASE_URL", f"http://127.0.0.1:{srv.server_port}/v1")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    yield srv
    srv.shutdown()


def test_embeddings_are_ordered_and_normalised(server):
    e = O.OpenAIEmbedder(batch_size=2)
    vecs = e.embed_many(["vegetarian dinner", "car", "recipe"])
    assert len(vecs) == 3 and len(vecs[0]) == 5
    assert vecs[1][3] > vecs[1][0]                      # "car" row landed at index 1 despite reversed response
    assert abs(sum(x * x for x in vecs[0]) - 1.0) < 1e-6
    assert FakeOpenAI.seen[0][1] == "Bearer proxy-injected"   # placeholder, never a real key


def test_real_key_from_env_is_used(server, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-123")
    O.OpenAIEmbedder().embed("car")
    assert FakeOpenAI.seen[-1][1] == "Bearer sk-test-123"


def test_memory_agent_end_to_end_with_openai(server, tmp_path):
    emb = O.OpenAIEmbedder()
    db = tmp_path / "m.db"
    mem = AgentMemory(db, namespace="u", embedder=emb)
    with mem.session():
        O.OpenAIAgent(mem, model="gpt-4.1-mini").chat("Hi, I'm vegetarian.")
    mem.close()
    mem2 = AgentMemory(db, namespace="u", embedder=emb)
    with mem2.session():
        assert O.OpenAIAgent(mem2, model="gpt-4.1-mini").chat("Suggest a dinner recipe") == "Try a mushroom risotto."
    chats = [b for p, _, b in FakeOpenAI.seen if p == "/v1/chat/completions"]
    assert chats[-1]["messages"][0]["role"] == "system" and "<memory>" in chats[-1]["messages"][0]["content"]
    assert "max_completion_tokens" in chats[-1] and chats[-1]["temperature"] == 0.0


def test_errors_are_clear_and_do_not_echo_the_key(server, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-secret-999")
    monkeypatch.setattr(O, "BASE_URL", "http://127.0.0.1:1/v1")
    with pytest.raises(O.OpenAIError, match="Cannot reach"):
        O.openai_complete()("", [{"role": "user", "content": "hi"}])
    assert "sk-secret-999" not in "".join(str(x) for x in FakeOpenAI.seen)
