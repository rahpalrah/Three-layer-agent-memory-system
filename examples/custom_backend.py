"""Plug in your own embedder, scorer and storage."""
from agentmem import AgentMemory, CallableEmbedder, CallableImportanceScorer, InMemoryStorage

# e.g. sentence-transformers:  model.encode(text).tolist(), dim=384
fake_embed = CallableEmbedder(lambda t: [float(len(t)), float(t.count(" ")), 1.0], dim=3)

# e.g. business rules: anything mentioning an order id is important
scorer = CallableImportanceScorer(lambda text, role: 0.9 if "order" in text.lower() else 0.3)

mem = AgentMemory(storage=InMemoryStorage(), embedder=fake_embed, scorer=scorer, namespace="shop")

@mem.on
def audit(event, payload):
    if event in ("learn", "evict"):
        print(f"[audit] {event}: {payload['item'].content}")

with mem.session():
    mem.add("Where is my order #8812?", role="user")
    mem.add("Nice weather today", role="user")
print(mem.stats())
