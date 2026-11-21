"""Run twice:  python examples/quickstart.py  - the second run remembers the first."""
from agentmem import AgentMemory

mem = AgentMemory("demo_memory.db", namespace="user-42")
print("Before:", mem.stats())

with mem.session(channel="cli"):
    mem.add("Hi, my name is Ada and I live in London.", role="user")
    mem.add("Nice to meet you, Ada!", role="assistant")
    mem.add("I prefer concise answers. Remember that my thesis deadline is June 3.", role="user")
    mem.add("ok thanks", role="user")  # low importance - gets discarded

print("After:", mem.stats())
print("\nRecall 'when is the deadline?':")
for hit in mem.recall("when is the deadline?", k=3):
    print(f"  {hit.score:.2f} [{hit.item.layer.value}] {hit.item.content}")

print("\nContext block for an LLM prompt:\n" + mem.build_context("tell me about the user"))
mem.close()
