# agentmem: three-layer memory for AI agents

A zero-dependency Python SDK that gives an agent **long-term memory** that persists across sessions:

```
           add()                 evict / session end              consolidate()
 user ─▶ ┌──────────────┐ ───────────────────────▶ ┌──────────────┐ ──────────────▶ ┌──────────────┐
 turns   │   WORKING    │   (trivia < 0.1 dropped)  │   EPISODIC   │  facts extracted │   SEMANTIC   │
         │ bounded, by  │                           │ time-stamped │  + de-duplicated │ durable facts│
         │ items+tokens │                           │ per session  │                  │ keyed/merged │
         └──────────────┘                           └──────┬───────┘                  └──────────────┘
                 ▲                                          │ forget(): decayed, low-importance episodes
                 └────────── recall() / build_context() ◀───┴─ ranked by relevance × recency × importance
```

| Layer | Holds | Lifetime |
|---|---|---|
| **Working** | the last N turns of the current conversation (item + token budget) | the session. It is mirrored to storage, so a crashed session can be resumed |
| **Episodic** | what happened: content, role, session, timestamp, importance, access stats | weeks; it fades unless it is important or keeps getting recalled |
| **Semantic** | what is true: distilled facts with evidence counts, sources and `key` versioning | indefinitely |

**Importance scoring.** Every write is scored from 0 to 1. The built-in heuristic scores explicit "remember…" requests, personal facts, preferences, commitments, dates, amounts, role and novelty. You can swap in an LLM scorer. Importance **decays** with a half-life and is **reinforced** each time the memory is recalled. Pinned items never fade.

**Cross-session persistence.** SQLite is used by default, and you can implement your own backend with `StorageBackend`. Namespaces isolate users and tenants.

## Install

```bash
pip install -e .                 # core, no dependencies
pip install -e ".[anthropic]"    # + Claude integration
```

## Quickstart

```python
from agentmem import AgentMemory

mem = AgentMemory("memory.db", namespace="user-42")

with mem.session():                                   # session end → flush + consolidate
    mem.add("Hi, my name is Ada and I live in London.", role="user")
    mem.add("I prefer concise answers. Remember that my deadline is June 3.", role="user")

# ...later, in a new process...
mem = AgentMemory("memory.db", namespace="user-42")
mem.recall("when is the deadline?")      # → [ScoredMemory(semantic: "The user's deadline is June 3."), ...]
mem.build_context("tell me about the user")   # → text block for your system prompt
```

## Core API

| Call | What it does |
|---|---|
| `add(text, role=, importance=, tags=, pinned=)` | observe a conversation turn (working memory) |
| `remember(text, importance=)` | write an episode directly (events, tool results) |
| `learn(fact, key=)` | write a fact; with `key="user.city"`, newer values replace older ones and the history is kept |
| `recall(query, k=, layers=, tags=)` | ranked search across all layers |
| `build_context(query)` | prompt-ready memory block |
| `session()` / `start_session(id)` / `end_session()` | session lifecycle; restarting with an existing id resumes it |
| `consolidate()` | episodic → semantic, then forget faded episodes |
| `on(hook)` | events: `add, evict, discard, learn, consolidate, session_start, session_end, forget` |
| `export_json()` / `import_data()` / `forget(id)` / `reset()` | portability, GDPR-style deletion |
| `stats()` | counts for each layer |

Each layer can also be used on its own: `mem.working`, `mem.episodic`, `mem.semantic`.

## Plug in your own components

```python
AgentMemory(
    storage=MyPostgresStorage(),                         # subclass StorageBackend
    embedder=CallableEmbedder(model.encode, dim=384),    # any text → vector
    scorer=CallableImportanceScorer(my_fn),              # fn(text, role) → 0..1
    extractor=CallableExtractor(my_llm_fn),              # fn(episodes) → [Fact | str]
    weights=RetrievalWeights(relevance=1, recency=.5, importance=.7, recency_half_life_hours=72),
)
```

The default `HashingEmbedder` has no dependencies and works on keywords. Use a real embedding model when you need semantic recall in production.

## Claude integration

```python
from anthropic import Anthropic
from agentmem.integrations.anthropic import MemoryAgent

agent = MemoryAgent(Anthropic(), mem, use_llm_memory=True)  # Claude also scores importance + extracts facts
with mem.session():
    agent.chat("I'm vegetarian, by the way.")
with mem.session():                                          # days later
    agent.chat("Suggest a dinner recipe")                    # → respects the vegetarian fact
```

## CLI

```bash
agentmem --db memory.db --ns user-42 stats
agentmem --db memory.db --ns user-42 search "deadline"
agentmem --db memory.db --ns user-42 list --layer semantic
agentmem --db memory.db --ns user-42 export dump.json
```

## Development

```bash
pip install -e ".[dev]" && pytest
```

The examples are in `examples/`: `quickstart.py` (run it twice), `claude_chat.py` and `custom_backend.py`.
