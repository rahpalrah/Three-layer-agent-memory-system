# agentmem: test results

Everything here runs for free, with no API keys and no paid services. The numbers come from the commands shown, run on 2026-11-21 with Python 3.13 and numpy installed.

## 1. Business use case: coffee-shop support desk

`python examples/support_desk.py`

The simulation has 20 customers, each with 4 conversations over 8 simulated weeks (812 turns in total). The conversations include:
- introductions
- grind and roast preferences
- a food allergy
- a damaged order with a promised replacement
- a change of mind on roast for half the customers ("I prefer medium roast now")
- filler such as "ok thanks"

In week 8, each customer asks 5–6 questions that only memory can answer.

The table shows the share of week-8 questions answered correctly:

| | stateless bot | last-20-messages window | agentmem: top-3 search | agentmem: answer in the prompt context |
|---|---|---|---|---|
| **Direct questions** ("Am I allergic to anything?") | 0% | 27% | **100%** | **100%** |
| **Paraphrased questions** ("Is there any ingredient I can't have?") | 0% | 27% | 47% | **100%** |

| category (paraphrased) | window-20 | agentmem top-3 | agentmem context |
|---|---|---|---|
| identity (name, city) | 2% | 2% | 100% |
| safety (allergy) | 0% | 0% | 100% |
| preference (grind) | 0% | 75% | 100% |
| current preference, after a change of mind | 50% | 100% | 100% |
| open issue (damaged order #) | 100% | 95% | 100% |

- The prompt context is about **550 characters**, against **about 1,050** for the full history. The gap widens as history grows, because memory is bounded and history is not.
- 812 turns were observed; 638 episodes were kept (filler dropped) and 116 active facts learned. Stale preferences are superseded but kept for audit.
- **How to read this:** paraphrased *search* is limited by the built-in keyword embedder. Answers still reach the model because the core profile (keyed facts such as name and city) and the most important facts are always injected. For semantic search, plug in `OllamaEmbedder` (free, local).
- **Caveat:** the data is synthetic. The test was written to stress layer behaviour (supersession, safety facts, filler), and part of the design was fixed after the first run exposed problems (section 4).

## 2. Public benchmark: LoCoMo (long-term conversational memory)

`python benchmarks/locomo.py`. This downloads [LoCoMo](https://github.com/snap-research/locomo) (Maharana et al., ACL 2024).

The dataset has 10 real-length conversations: 272 sessions spanning months, 5,882 turns and 1,536 questions (the adversarial category is excluded). Each conversation is replayed with its real session dates through an injected clock. The metric is **evidence recall@k**: the share of the annotated answer turns that retrieval returns. No LLM is involved.

| system | recall@5 | recall@10 | recall@25 | hit@10 |
|---|---|---|---|---|
| random | 1.1% | 2.0% | 5.0% | 3.1% |
| recency (last k turns, i.e. a chat window) | 0.9% | 1.6% | 3.4% | 1.9% |
| BM25 (strong lexical baseline) | 50.2% | 57.9% | 67.1% | 64.1% |
| agentmem v0, dense-only *(before this round)* | 30.3% | 37.2% | 48.8% | 41.5% |
| **agentmem (shipped: hybrid, lexical = 0.8)** | **48.8%** | **56.2%** | **66.3%** | **63.0%** |
| agentmem, relevance only (recency/importance off) | 49.9% | 57.6% | 67.7% | 64.0% |

Sweep of the share of relevance that comes from BM25 (`lexical`), for agentmem recall@10:

| 0.0 | 0.5 | 0.8 | 1.0 |
|---|---|---|---|
| 37.2% | 54.1% | 56.2% | 56.8% |

**Findings:**
1. **The benchmark exposed a real weakness.** Dense-only retrieval with the built-in hashing embedder reached 37% recall@10. Common words, such as speaker names on every turn, swamped it. Hybrid BM25 scoring raised it to **56%**, about BM25's level, and **35× a chat-history window**.
2. **The shipped default of 0.8 was chosen from this sweep, on this dataset.** These LoCoMo numbers are therefore tuned, not held out.
3. **Recency and importance cost about 1.4 points on LoCoMo.** Its questions ask about arbitrary past moments, and there is no "what matters now" signal for those layers to exploit. They earn their keep in the support-desk test: the current-preference and profile rows.
4. **Not measured: end-to-end QA accuracy.** That needs an LLM to read the retrieved context and answer. It is free with Ollama locally, but Ollama could not be downloaded in this environment.
5. **Expected next gain:** a real semantic embedder, such as `OllamaEmbedder("nomic-embed-text")` with the default hybrid of 0.5. Not measured here, because the model download was blocked.

## 3. System tests

| suite | result |
|---|---|
| `pytest`: 24 unit tests, including the Ollama adapter against a mock Ollama HTTP server | 24/24 pass |
| `tests/e2e/run_e2e.py`: 14 scenarios against the built wheel in a fresh venv | 14/14 pass |

The end-to-end scenarios cover:
- persistence across processes
- `kill -9` recovery
- 6 concurrent writer processes and 8 threads
- the full CLI
- a 600-turn conversation
- decay and forgetting
- 5,000 memories (about 300 ms per recall)
- Claude through the real `anthropic` SDK with a mocked HTTP transport
- the examples
- hostile input
- the support-desk business run
- the Ollama adapter

## 4. Bugs found by testing and fixed

| found by | problem | fix |
|---|---|---|
| E2E | Important but irrelevant memories outranked relevant ones | Joint ranking across layers; normalised relevance gates recency and importance |
| E2E | Non-ASCII text got zero relevance | Unicode tokenizer, CJK bigrams, light stemming |
| E2E | Recall took about 930 ms at 5,000 memories | Shallow copies, float32 BLOB embeddings, numpy path: about 300 ms |
| support desk | "Who am I?" (all stopwords) was ranked as if it were a real query | Content-free queries fall back to importance and recency |
| support desk | The customer's name dropped out of the prompt once they had more than 6 facts | Core profile (keyed facts) is always in `build_context` |
| support desk | "I changed my mind, medium roast now" left the stale "dark roast" fact winning | Slot-based supersession on explicit update words ("now", "anymore", "changed my mind"…) |
| LoCoMo | Dense-only retrieval at 37% recall@10 | Hybrid BM25 + embeddings: 56% |
