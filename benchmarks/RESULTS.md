# agentmem: final results

Everything here is free to reproduce: no API keys and no paid services. Run on 2026-11-21 with Python 3.13 on CPU.

## Headline numbers

| Test | Chat history (last N messages) | Keyword search (BM25) | **agentmem** |
|---|---|---|---|
| **LoCoMo** public benchmark, held-out conversations 2–10, evidence recall@10 | 1.6% | 57.8% | **63.3%** |
| LoCoMo hit@10: at least one answer turn retrieved | 1.9% | 64.1% | **70.6%** |
| **Coffee-shop support desk**: answer in the model's context, week 8 | 27% | — | **100%** |
| Support desk, *reworded* questions: correct memory in the top 3 | 27% (in window) | — | **56%** |

**Recommended configuration:** `AgentMemory(..., embedder="minilm")`. It uses local all-MiniLM-L6-v2 embeddings (`pip install agentmem[onnx]`) with the shipped hybrid defaults.

## 1. LoCoMo: long-term conversational memory

The benchmark is [LoCoMo](https://github.com/snap-research/locomo) (Maharana et al., ACL 2024): 10 conversations of about 27 sessions each, spanning months, with questions annotated with the exact turns that answer them. Each conversation is replayed with its real dates through an injected clock. The metric is the share of answer turns that retrieval returns. No LLM is used.

**Method:**
- **Tuning:** settings were tuned on **conversation 1 only** (150 questions, 128 configurations; `benchmarks/tune.py`).
- **Final numbers:** reported on **conversations 2–10** (1,386 questions), which played no part in tuning.
- **Command:** `python benchmarks/locomo.py --conversations 2-10 --embedders hashing minilm`

| system (held-out) | recall@5 | recall@10 | recall@25 | hit@10 |
|---|---|---|---|---|
| random | 0.8% | 1.7% | 4.8% | 2.7% |
| recency (chat window) | 0.9% | 1.6% | 3.3% | 1.9% |
| BM25 | 50.1% | 57.8% | 67.1% | 64.1% |
| agentmem v0 (first release: dense hashing only) | 30.8% | 37.6% | 48.7% | 42.3% |
| agentmem, MiniLM embeddings only | 33.0% | 43.6% | 57.6% | 49.6% |
| agentmem, MiniLM hybrid with RRF fusion | 47.0% | 57.0% | 69.5% | 64.1% |
| agentmem, built-in embedder, shipped hybrid | 49.7% | 57.5% | 67.3% | 64.4% |
| agentmem, built-in embedder, shipped + neighbours (`expand=1`) | 52.4% | 61.0% | 69.9% | 67.7% |
| **agentmem, MiniLM, shipped hybrid** | **54.8%** | **63.3%** | **73.2%** | **70.6%** |

The table below shows recall@10 by question type on the held-out set:

| | multi-hop | temporal | open-domain | single-hop |
|---|---|---|---|---|
| BM25 | 30.5% | 66.0% | 25.9% | 66.9% |
| **agentmem, MiniLM, shipped** | **37.0%** | **68.9%** | **35.2%** | **72.7%** |

**Findings:**
1. **Hybrid search is the single biggest win.** Embeddings alone lose to BM25 on LoCoMo (43.6% vs 57.8%), because the questions are full of names, which keyword matching handles well. Combining them gives 63.3%, ahead of either alone. This matches common industry practice.
2. **Weighted score fusion beat RRF** on both the dev and test sets (63.3% vs 57.0%). RRF ignores how large the score gaps are, and here the gaps carry signal.
3. **Neighbour expansion** (sentence-window retrieval) helps the built-in embedder (57.5% → 61.0%) but not MiniLM. It stays opt-in.
4. **Recency and importance are weak signals on LoCoMo,** whose questions ask about arbitrary past moments. The tuned defaults keep them as moderate tie-breakers, gated by relevance. They matter in real applications (section 2).
5. **Not measured: end-to-end answer accuracy.** That needs an LLM to read the retrieved context, which is free with Ollama locally but couldn't be installed here.

## 2. Business use case: coffee-shop support desk

The simulation has 20 customers, each with 4 conversations over 8 weeks (812 turns), covering preferences, a food allergy, a damaged order and a change of mind. The command is `python examples/support_desk.py --embedder minilm`.

Results with MiniLM, the recommended configuration:

| category | stateless bot | last-20 window | agentmem top-3 (direct / reworded) | agentmem context (direct / reworded) |
|---|---|---|---|---|
| identity (name, city) | 0% | 2% | 100% / 50% | 100% / 100% |
| safety (allergy) | 0% | 0% | 100% / **100%** | 100% / 100% |
| preference (grind) | 0% | 0% | 100% / 40% | 100% / 100% |
| current preference, after a change of mind | 0% | 50% | 100% / 100% | 100% / 100% |
| open issue (damaged order) | 0% | 100% | 100% / 5%* | 100% / 100% |
| **ALL** | **0%** | **27%** | **100% / 56%** | **100% / 100%** |

\* The top hit for "Any update on the broken package?" is "Yes, the replacement is on its way.". That is the right memory, but it doesn't contain the order number, which is what the check looks for.

With the built-in embedder, reworded allergy questions scored **0%** in the top-3 search, against 100% with MiniLM. That is the clearest case for semantic embeddings. This test is synthetic and was written by us, so it shows how each layer behaves rather than providing independent evidence.

## 3. Changes made, and why

| Practice | Change | Measured effect |
|---|---|---|
| Real semantic embeddings | `MiniLMEmbedder`: all-MiniLM-L6-v2 on ONNX Runtime, CPU, SHA-256-verified download, about 1 ms per sentence batched | Reworded allergy questions in the top 3: 0% → 100% |
| Hybrid lexical + vector search | BM25 over the candidate set, fused with embedding similarity | LoCoMo recall@10: 37.6% → 63.3% |
| Rank fusion options | `fusion="weighted"` (default) or `"rrf"` | Weighted wins by 6.3 points (held out) |
| Score normalisation | True min-max, so embedding and RRF scores aren't squashed | Required for fusion to behave |
| Relevance-gated recency and importance | `relevance_gate=1.0`, weights 0.25 / 0.35, tuned on the dev split | +3.2 points vs the untuned hybrid (held out) |
| Dev/test discipline | `tune.py` on conversation 1; report conversations 2–10 | No test-set tuning in the final numbers |
| Semantic dedup that respects values | Merge only when vectors **and** content words agree | Fixed: MiniLM merged "dark roast" into "light roast" (current preference 70% → 100%) |
| Sentence-window retrieval | `recall(..., expand=n)` | +3.5 points with the built-in embedder |
| Core profile memory | Keyed facts always go into the context | Identity 100% in context |
| Explicit change-of-mind handling | Slot-based supersession, with history kept | Current preference 100% |

## 4. System tests

- `pytest`: **24/24** pass.
- `tests/e2e/run_e2e.py`: **15/15** pass, against a freshly built wheel in a clean venv with all extras. The scenarios cover:
  - persistence across processes
  - `kill -9` recovery
  - concurrent processes and threads
  - the CLI
  - 600-turn conversations
  - decay and forgetting
  - scale (5,000 memories)
  - Claude through the real SDK with a mocked transport
  - Ollama against a mock server
  - MiniLM semantic recall
  - the business scorecard
  - hostile input

## 5. Known limits

- **Answer accuracy:** not measured, because no LLM was available here. Run `examples/support_desk.py --ollama llama3.2` locally.
- **Fact extraction:** the rule-based extractor is pattern-based. For open-ended conversations, use the LLM extractor (`use_llm_memory=True` with Claude or Ollama).
- **Search scale:** search is exhaustive, about 300 ms at 5,000 memories per namespace. Beyond about 10,000, put a vector index behind `StorageBackend`.
- **Rerankers:** a cross-encoder reranker (standard in production RAG) was not added, because no reranker model was downloadable in this environment.
