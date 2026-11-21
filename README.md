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
pip install -e ".[fast]"         # + numpy-accelerated similarity search
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

## Results

The full report is in [`benchmarks/RESULTS.md`](benchmarks/RESULTS.md). All of it can be reproduced for free.

- **Business use case** (`examples/support_desk.py`, coffee-shop support desk with 20 customers over 8 weeks): the answer was in the model's context for **100%** of week-8 questions, against **27%** for a last-20-messages chat window and **0%** for a stateless bot. That includes allergies and changes of mind.
- **LoCoMo** public benchmark (`benchmarks/locomo.py`, 1,536 questions over months-long conversations): **56.2% evidence recall@10**, against 1.6% for a chat window and 57.9% for BM25.

## Free local LLMs (Ollama)

```python
from agentmem.integrations.ollama import OllamaAgent, OllamaEmbedder

mem = AgentMemory("memory.db", namespace="user-42", embedder=OllamaEmbedder("nomic-embed-text"))
agent = OllamaAgent(mem, model="llama3.2", use_llm_memory=True)
```

This has no extra Python dependencies; it talks to `ollama serve` over HTTP. To support another provider, pass any `complete(system, messages) -> str` to `agentmem.integrations.base.ChatAgent`.

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

## Retrieval

`recall()` ranks every candidate from all layers together:

```
score = w_rel·rel' + gate(rel')·(w_rec·recency + w_imp·importance)
        rel' = relevance min-max normalised across candidates
        gate = (1 − relevance_gate) + relevance_gate·rel'
```

Relevance is normalised, so scores don't depend on the embedder's scale. The gate stops an important but unrelated memory from outranking a relevant one; set `relevance_gate=0` to get the purely additive *Generative Agents* formula. Each layer has its own decay: semantic facts fade over about a year, episodes over about a month.

## Testing

```bash
pip install -e ".[dev]" && pytest               # 24 unit tests
python tests/e2e/run_e2e.py                     # 14 end-to-end system scenarios
python examples/support_desk.py                 # business use-case scorecard
python benchmarks/locomo.py                     # public LoCoMo benchmark (downloads 2.8 MB)
```

The end-to-end suite runs against the **installed** package and uses real OS processes. It covers:
- persistence across processes
- recovery after `kill -9` mid-session
- 6 concurrent writer processes and 8 threads on one database
- the full CLI workflow
- a 600-turn conversation
- decay and forgetting
- a 5,000-memory scale run
- the support-desk business run
- the Ollama adapter against a mock server
- the Claude integration through the real `anthropic` SDK with a mocked HTTP transport
- the examples
- hostile input (Unicode/CJK, 200 KB text, NUL bytes, SQL-like strings)

CI runs both suites on Python 3.9–3.13, with and without the optional extras.

## Performance and limits

On the end-to-end machine, 5,000 episodes take about 1.4 s to write, and `recall` takes about 350 ms per query with `[fast]`. Recall does an exhaustive scan of the namespace, which is fine up to about 10,000 memories per namespace. Beyond that, implement a `StorageBackend` on a vector index (pgvector, Qdrant, ...). Retrieval is hybrid: BM25 keywords plus embeddings (`RetrievalWeights.lexical`). The default `HashingEmbedder` matches on keywords with light stemming. Plug in a real embedding model for semantic recall (car ≈ vehicle).

The examples are in `examples/`: `quickstart.py` (run it twice), `claude_chat.py` and `custom_backend.py`.
