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
pip install -e ".[onnx]"         # + local semantic embeddings and cross-encoder reranker (recommended)
pip install -e ".[faiss]"        # + FAISS HNSW approximate-nearest-neighbour index
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

| Public benchmark (held out; settings tuned on one LoCoMo conversation only) | chat history | BM25 | **agentmem + reranker** |
|---|---|---|---|
| LoCoMo evidence recall@10 (1,386 questions) | 1.6% | 57.8% | **68.9%** |
| LongMemEval-S session-any@10 (470 questions, ~494-turn histories) | 11.5% | 96.6% | **98.7%** |
| LongMemEval-S session-all@10: every answer session found | 2.6% | 84.7% | **88.7%** |
| Coffee-shop support desk: answer in the model's context after 8 weeks | 27% | — | **100%** |

On LongMemEval the gains over BM25 are statistically significant (paired bootstrap, 95% CI); the report lists the significance of every comparison and the cases that still fail.

## Recommended setup

```bash
pip install "agentmem[onnx]"     # adds local semantic embeddings (all-MiniLM-L6-v2, ~90 MB, CPU)
```

```python
mem = AgentMemory("memory.db", namespace="user-42", embedder="minilm")
```

This uses hybrid retrieval (BM25 keywords plus MiniLM vectors), with defaults tuned on a LoCoMo dev split and validated on held-out data. The model is downloaded once from a public mirror and verified by SHA-256. Set `AGENTMEM_MODEL_DIR` to choose where it is cached.

## OpenAI

```python
from agentmem import AgentMemory
from agentmem.integrations.openai import OpenAIEmbedder, OpenAIAgent

mem = AgentMemory("memory.db", embedder=OpenAIEmbedder())         # text-embedding-3-small
agent = OpenAIAgent(mem, model="gpt-4.1-mini")
```

Set `OPENAI_API_KEY`. No extra dependency is needed. Benchmarks accept `--llm openai:gpt-4.1-mini`.

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

Retrieval follows the standard production pattern: **index → hybrid ranker → cross-encoder**.

| Stage | What it does | Configure |
|---|---|---|
| 1. Candidate index | Vector top-N ∪ BM25 top-N from an inverted index. It stays in sync with writes and rebuilds itself when another process changes the database | `index="auto"\|"numpy"\|"faiss"\|None`, `candidates=200` |
| 2. Hybrid ranker | Relevance-gated recency and importance | `RetrievalWeights(...)` |
| 3. Cross-encoder rerank (optional) | Re-scores the top `rerank_depth` with ms-marco-MiniLM-L-6-v2 | `reranker=CrossEncoderReranker()`, `rerank_depth=30`, `rerank_weight=0.7` |

Stage 2 ranks with:

```
relevance = minmax( (1−λ)·minmax(vector similarity) + λ·minmax(BM25) )     # hybrid search
score     = w_rel·relevance + relevance·(w_rec·recency + w_imp·importance)  # relevance-gated
```

- **Defaults** (`RetrievalWeights`): λ = 0.5 with semantic embeddings and 0.8 with the built-in keyword embedder; `w_rec` = 0.25 and `w_imp` = 0.35. They were tuned on a LoCoMo dev split.
- **Fusion:** `fusion="rrf"` switches to reciprocal rank fusion.
- **Neighbours:** `recall(..., expand=1)` also returns the turns next to each hit (sentence-window retrieval).
- **Decay:** each layer has its own. Semantic facts fade over about a year, episodes over about a month.
- **Bulk ingestion:** `remember_many(texts)` embeds in batches and writes in one transaction.

```python
from agentmem import AgentMemory
from agentmem.rerank import CrossEncoderReranker

mem = AgentMemory("memory.db", embedder="minilm", index="faiss",     # pip install "agentmem[onnx,faiss]"
                  reranker=CrossEncoderReranker())
```

### Downloading LongMemEval

```bash
mkdir -p benchmarks/data/longmemeval && cd benchmarks/data/longmemeval
curl -L -O https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json   # 277 MB
```

## Independence: what needs nothing else

The core library has **no required dependencies**, no API keys and no model downloads. Verified in a clean environment with only the package installed (no numpy, no ML packages): storage, layers, consolidation, retrieval, persistence and the CLI all work.

| Component | Needs | Notes |
|---|---|---|
| Storage, working/episodic/semantic memory, decay, forgetting, consolidation | nothing | pure Python, SQLite from the standard library |
| Default embedder (`HashingEmbedder`) | nothing | keyword-based |
| **Self-contained semantic embedder (`SelfEmbedder`)** | nothing | learns word similarity from the memory's own text (PPMI word vectors, seeded random projection). No download, no package |
| BM25 keyword retrieval, hybrid ranking | nothing | pure Python |
| Vector index | optional numpy | falls back to exact full scan without it |
| MiniLM embeddings, cross-encoder reranker | optional `onnxruntime` + `tokenizers`; weights via `agentmem fetch-models` (SHA-256 verified) | higher quality, not required |
| Extractive answer reader (`ExtractiveReader`) | nothing | copies answers from memory; a weak baseline, F1 6.0% on LoCoMo |
| Claude, OpenAI, Ollama chat | optional API key or local server | only for LLM-based importance scoring, fact extraction and agents |

```python
from agentmem import AgentMemory, SelfEmbedder

emb = SelfEmbedder(dim=256).fit(all_text_you_have)   # unlabelled text from the memory itself
mem = AgentMemory("memory.db", embedder=emb)
```

Measured on LoCoMo held-out conversations 2-10 (`benchmarks/locomo.py --embedders self`): the self-contained embedder reaches 58.3% evidence recall@10, against 57.5% for the built-in keyword embedder, 57.8% for BM25 and 63.3% for MiniLM. It learns from each conversation's own text, with no labels.

Check what is available on your machine:

```bash
agentmem doctor
agentmem fetch-models --minilm --reranker     # optional, downloads and verifies the weights once
```

## Licences and dataset restrictions

The code is MIT-licensed (`LICENSE`). Benchmark datasets are **not** included and carry their own licences; see [`NOTICE.md`](NOTICE.md).

**Customer Support on Twitter** (Kaggle) is licensed **CC BY-NC-SA 4.0: non-commercial use only**. Any results you build from it inherit that restriction. The repository publishes only aggregate statistics, never tweet text.

## Testing

```bash
pip install -e ".[dev]" && pytest               # 40 unit tests
python tests/e2e/run_e2e.py                     # 18 end-to-end system scenarios
python examples/support_desk.py                 # business use-case scorecard
python benchmarks/tune.py [--rerank]            # tune retrieval / reranker on the LoCoMo dev split
python benchmarks/longmemeval.py --rerank       # LongMemEval retrieval (download the dataset first, see below)
python benchmarks/longmemeval_qa.py --llm llamacpp:model.gguf --judge   # LongMemEval answer accuracy
python benchmarks/scale.py                      # latency at 1k-100k memories
python benchmarks/twcs.py [--e2e]               # Customer Support on Twitter: returning customers (Kaggle, see RESULTS.md section 10)
python benchmarks/twcs.py --brand AmazonHelp    # one brand inbox in one shared memory, tag isolation
python benchmarks/twcs_scale.py                 # all 2.7M tweets, 702k tenants, one SQLite file
python benchmarks/locomo_qa.py --llm ollama:llama3.2 --judge   # answer accuracy (F1, LLM judge)
python benchmarks/locomo.py --conversations 2-10 --embedders hashing minilm   # held-out test
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
- MiniLM semantic recall
- the Claude integration through the real `anthropic` SDK with a mocked HTTP transport
- the examples
- hostile input (Unicode/CJK, 200 KB text, NUL bytes, SQL-like strings)

CI runs both suites on Python 3.9–3.13, with and without the optional extras.

## Performance and limits

See `benchmarks/results/scale.json`:
- **Latency:** with the stage-1 index, `recall` takes about 7 ms at 1,000 memories and about 36 ms at 100,000 (p50, CPU). A full scan takes 8.7 s at 100,000.
- **Cold start:** the index is built in memory once per process, taking about 1 s per 10,000 memories.
- **Beyond a single process:** for millions of memories, or several application servers sharing one index, implement `StorageBackend` on a vector database (pgvector, Qdrant, ...).
