"""Interactive chat with Claude that remembers you across runs.

    pip install "agentmem[anthropic]"
    export ANTHROPIC_API_KEY=...
    python examples/claude_chat.py
"""
from anthropic import Anthropic

from agentmem import AgentMemory
from agentmem.integrations.anthropic import MemoryAgent

mem = AgentMemory("chat_memory.db", namespace="me")
agent = MemoryAgent(Anthropic(), mem, use_llm_memory=True)  # Claude scores importance + extracts facts

with mem.session():
    print("Chat (Ctrl-D to quit). Known facts:", mem.semantic.count())
    try:
        while True:
            print("Claude:", agent.chat(input("You: ")))
    except (EOFError, KeyboardInterrupt):
        pass
print("\nSession consolidated. Stats:", mem.stats())
