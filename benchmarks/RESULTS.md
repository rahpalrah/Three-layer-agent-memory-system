# agentmem: final results

Everything here is free to reproduce: no API keys and no paid services. Run on CPU with Python 3.13. Retrieval settings were tuned on LoCoMo **conversation 1 only**, and every LoCoMo number below is from the **held-out conversations 2–10**.

## Headline numbers

| Test (held-out) | Chat history | BM25 | agentmem (MiniLM hybrid) | **agentmem + reranker** |
|---|---|---|---|---|
| LoCoMo evidence recall@5 | 0.9% | 50.1% | 54.8% | **63.1%** |
| LoCoMo evidence recall@10 | 1.6% | 57.8% | 63.3% | **68.9%** |
| LoCoMo hit@10 (at least one answer turn found) | 1.9% | 64.1% | 70.6% | **76.6%** |
| LoCoMo answer coverage@10 (gold-answer words in the retrieved context) | 17.4% | 60.0% | 63.2% | **66.9%** |
| Recall latency at 100,000 memories (p50) | — | — | **36 ms** (8.7 s before the index) | + about 30 pairs through the cross-encoder |
| Coffee-shop support desk: answer in the model's context, week 8 | 27% | — | **100%** | — |

**Recommended configuration:**

```python
from agentmem import AgentMemory
from agentmem.rerank import CrossEncoderReranker

mem = AgentMemory("memory.db", embedder="minilm", reranker=CrossEncoderReranker())   # pip install "agentmem[onnx]"
```

## 1. Retrieval quality: LoCoMo

[LoCoMo](https://github.com/snap-research/locomo) (Maharana et al., ACL 2024) has 10 long conversations spanning months. Questions are annotated with the turns that answer them. The held-out split has 1,386 questions and 5,463 turns. The command is `python benchmarks/locomo.py --conversations 2-10 --embedders hashing minilm --rerank`.

| system | recall@5 | recall@10 | recall@25 | hit@10 |
|---|---|---|---|---|
| random | 0.8% | 1.7% | 4.8% | 2.7% |
| recency (chat window) | 0.9% | 1.6% | 3.3% | 1.9% |
| BM25 | 50.1% | 57.8% | 67.1% | 64.1% |
| agentmem v0 (first release) | 30.8% | 37.6% | 48.7% | 42.3% |
| agentmem, built-in embedder, hybrid | 49.6% | 57.5% | 67.2% | 64.4% |
| agentmem, built-in embedder, hybrid + reranker | 59.8% | 64.0% | 68.4% | 71.1% |
| agentmem, MiniLM embeddings only | 33.0% | 43.6% | 57.6% | 49.5% |
| agentmem, MiniLM hybrid with RRF fusion | 46.6% | 57.1% | 69.8% | 64.1% |
| agentmem, MiniLM hybrid | 54.8% | 63.3% | 73.2% | 70.6% |
| **agentmem, MiniLM hybrid + cross-encoder reranker** | **63.1%** | **68.9%** | **74.7%** | **76.6%** |

**Reranker settings:** depth 30 and weight 0.7 were picked on the dev conversation (`benchmarks/tune.py --rerank`). There, recall@10 went from 67.1% to 74.7% (`results/tune_rerank.json`). On held-out data the gain is +5.6 points at 10 results and **+8.3 at 5**: the reranker matters most when only a few memories fit in the prompt.

**Index validation:** the MiniLM hybrid scores exactly the same with the stage-1 index as with the earlier full scan (63.3% recall@10). Two-stage retrieval costs no accuracy on this data.

## 2. Answer accuracy: LoCoMo QA

`benchmarks/locomo_qa.py` retrieves 10 memories with their session dates, asks an LLM to answer, and scores the answer two ways:
- **F1:** token-level F1, the official LoCoMo metric
- **J:** LLM-as-judge correctness, the Mem0-paper metric, with `--judge`

The LLM path is tested end to end against a mock Ollama server (E2E scenario 18). No real LLM can run in this environment, so the **F1 and J scores are not measured yet**.

The model-free part was measured: **answer coverage**, the share of gold-answer words present in the retrieved context. A reader that only sees this context can't do better than this ceiling.

| system | answer coverage | multi-hop | temporal | open-domain | single-hop |
|---|---|---|---|---|---|
| recency (chat window) | 17.4% | 11.4% | 26.6% | 15.4% | 16.2% |
| BM25 | 60.0% | 38.3% | 61.1% | 23.3% | 70.6% |
| agentmem (MiniLM hybrid) | 63.2% | 43.4% | 60.0% | 24.2% | 74.9% |
| **agentmem + reranker** | **66.9%** | **47.8%** | **61.1%** | **26.2%** | **79.5%** |

Coverage counts word overlap, so it isn't zero by chance: the chat window's 17.4% is mostly common words and dates. To measure real answer accuracy for free on your own machine:

```bash
ollama pull llama3.2
python benchmarks/locomo_qa.py --llm ollama:llama3.2 --judge
```

## 3. Speed: two-stage retrieval

Stage 1 takes vector top-200 ∪ BM25 top-200 from an inverted index, using exact numpy or FAISS HNSW. The full ranker then runs only on those candidates. The command is `python benchmarks/scale.py`. Each memory has a 512-dimensional embedding and is stored in SQLite.

| memories | full scan p50 | **indexed, numpy (p50 / p95)** | indexed, FAISS HNSW (p50 / p95) | top-10 agreement with full scan (numpy / HNSW) | index build, once per process |
|---|---|---|---|---|---|
| 1,000 | 64 ms | **7.3 / 9.2 ms** | 8.6 / 10.8 ms | 100.0% / 100.0% | 0.1 s |
| 10,000 | 727 ms | **12.3 / 16.3 ms** | 11.2 / 12.8 ms | 100.0% / 100.0% | 0.8 s |
| 50,000 | 3,747 ms | **20.3 / 24.1 ms** | 17.0 / 27.4 ms | 100.0% / 100.0% | 4.0 s |
| 100,000 | 8,734 ms | **35.5 / 44.9 ms** | 35.7 / 50.7 ms | 94.0% / 94.0% | 9.8 s |

- **Speed-up:** 245× at 100,000 memories.
- **Agreement:** 100% up to 50,000 memories. At 100,000 it is 94%, because the synthetic notes are near-duplicates: hundreds of candidates tie on relevance, and importance breaks the tie outside the 200-candidate pool. Raise `candidates=` for more coverage at higher latency.
- **Freshness:** the index stays in sync with its own writes and deletes. It detects writes from other processes or connections (SQLite `data_version`) and rebuilds; this is tested in E2E scenario 16.
- **FAISS vs numpy:** FAISS HNSW matches numpy here. At this size, exact matrix search is already fast, so FAISS is for larger collections.

## 4. Business use case: coffee-shop support desk

The simulation has 20 customers over 8 weeks (812 turns) with direct and reworded questions. With MiniLM:
- **Answer in the model's context: 100%** on both question sets, against 27% for a last-20-messages window and 0% for a stateless bot.
- **Top-3 search on reworded questions: 56%** (46% with the built-in embedder). Reworded allergy questions score 0% with the built-in embedder and **100% with MiniLM**.
- **Caveat:** the data is synthetic and was written by us. See `results/support_desk_*.json`.

## 5. Bugs found by this round of testing

| problem | fix |
|---|---|
| `AgentMemory(SQLiteStorage(db))` treated the storage object as a file path. It silently created a new database named after the object's string representation, and those writes never reached the real database. | A storage backend is accepted as the first argument; a `path` that isn't a string or path raises `TypeError` |
| With MiniLM, de-duplication merged "likes dark roast" into "prefers light roast" (0.87 similarity) | Facts merge only when vectors **and** content words agree |
| Recency and importance outweighed relevance on LoCoMo | Weights tuned on the dev split, with relevance gating |

## 6. System tests

- `pytest`: **31/31** pass.
- `tests/e2e/run_e2e.py`: **18/18** pass, against a freshly built wheel in a clean venv with all extras.

## 7. Model provenance

- **Embeddings:** all-MiniLM-L6-v2 (Apache-2.0), from Chroma's public ONNX mirror, verified by SHA-256 `913d7300…16ec3`.
- **Reranker:** cross-encoder/ms-marco-MiniLM-L-6-v2 (Apache-2.0), Xenova's ONNX export.
  - **Default download:** Hugging Face, with the files verified by pinned SHA-256 (`c623d0bc…` for `model.onnx`).
  - **These tests:** Hugging Face is blocked here, so the same files were taken from an npm package that bundles them. Its npm integrity hash was verified and its code was never run. The weights matched the pinned hash and reproduced the published reference scores (+8.8 vs −11.2).

## 8. Known limits

- **Answer accuracy:** F1 and J are still unmeasured until the harness is run with an LLM.
- **Index lifetime:** the index lives in the process's memory and is rebuilt once per process (about 1 s per 10,000 memories). For millions of memories, or a shared index across servers, implement `StorageBackend` on a vector database (pgvector, Qdrant).
- **Reranker cost:** it adds about 30 cross-encoder pairs per query on CPU.
- **Fact extraction:** the rule-based extractor is pattern-based. Use `use_llm_memory=True` for open-ended conversations.
