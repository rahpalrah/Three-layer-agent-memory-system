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

## 3. LongMemEval: long-term memory for chat assistants

[LongMemEval](https://github.com/xiaowu0162/LongMemEval) (Wu et al., ICLR 2025), the cleaned S variant, downloaded from Hugging Face (SHA-256 `d6f21ea9…3a442`). It has 500 questions. Each question has its **own** chat history of about 48 sessions and 494 turns (roughly 115k tokens). The question types are single-session (user, assistant, preference), **multi-session**, **knowledge-update** and **temporal-reasoning**.

**Method:** every turn of a haystack is written into a fresh `AgentMemory` with its real timestamp, and the question is asked at its `question_date`. We measure how well the evidence is retrieved from the top K turns. The 30 abstention questions have no evidence and are skipped, as in the official retrieval evaluation, which leaves 470 questions. No settings were tuned on this data (they were tuned on LoCoMo conversation 1 only), so this is an out-of-distribution test.
- **session-any@K:** at least one answer session appears among the sessions of the top-K turns.
- **session-all@K:** every answer session appears. This is the hard one: multi-session questions need 2 to 6 sessions.
- **turn-any@K:** an evidence turn itself is in the top K.

The command is `python benchmarks/longmemeval.py --rerank`.

| system | any@5 | all@5 | turn@5 | any@10 | all@10 | turn@10 | any@30 | all@30 | turn@30 |
|---|---|---|---|---|---|---|---|---|---|
| recency (chat window) | 8.1% | 1.7% | 2.3% | 11.5% | 2.6% | 4.9% | 16.4% | 4.9% | 12.1% |
| BM25 | 94.3% | 75.5% | 81.9% | 96.6% | 84.7% | 88.5% | 98.9% | 94.3% | 93.0% |
| agentmem dense only | 95.5% | 68.5% | 75.5% | 97.2% | 83.0% | 86.8% | 99.4% | 95.1% | 96.2% |
| agentmem hybrid (shipped) | 97.2% | 77.7% | 83.8% | 98.3% | 87.9% | 90.9% | 100.0% | 97.0% | 97.0% |
| agentmem hybrid, relevance only | 97.2% | 77.7% | 84.0% | 98.5% | 88.3% | 91.1% | 100.0% | 96.8% | 96.8% |
| **agentmem hybrid + reranker** | 97.4% | 81.7% | 87.0% | 98.7% | 88.7% | 92.6% | 100.0% | 97.0% | 97.0% |

session-all@10 by question type:

| system | single-session-user (n=64) | single-session-assistant (n=56) | single-session-preference (n=30) | multi-session (n=121) | knowledge-update (n=72) | temporal-reasoning (n=127) |
|---|---|---|---|---|---|---|
| recency (chat window) | 4.7% | 8.9% | 0.0% | 0.0% | 1.4% | 2.4% |
| BM25 | 100.0% | 100.0% | 76.7% | 71.9% | 97.2% | 77.2% |
| agentmem dense only | 95.3% | 100.0% | 93.3% | 71.1% | 97.2% | 70.1% |
| agentmem hybrid (shipped) | 100.0% | 100.0% | 93.3% | 77.7% | 100.0% | 78.0% |
| agentmem hybrid, relevance only | 100.0% | 100.0% | 93.3% | 78.5% | 98.6% | 79.5% |
| agentmem hybrid + reranker | 100.0% | 100.0% | 93.3% | 79.3% | 100.0% | 79.5% |

**Is agentmem better than BM25, or is it noise?** The paired bootstrap difference over questions (percentage points, 95% CI, 2,000 resamples; `n.s.` = not significant):

| system vs BM25 | session-any@10 | session-all@10 | turn-any@10 | session-all@5 |
|---|---|---|---|---|
| embeddings only | +0.6 [-1.5, +3.0] (n.s.) | -1.7 [-5.1, +1.5] (n.s.) | -1.7 [-5.1, +1.7] (n.s.) | -7.0 [-11.3, -3.0] |
| hybrid (shipped) | +1.7 [+0.4, +3.2] | +3.2 [+1.1, +5.3] | +2.3 [+0.2, +4.7] | +2.1 [-0.9, +5.1] (n.s.) |
| **hybrid + reranker** | +2.1 [+0.9, +3.6] | +4.0 [+1.9, +6.2] | +4.0 [+1.9, +6.2] | +6.2 [+3.4, +8.9] |

**Findings:**
1. **A chat window finds almost nothing** (11.5% session-any@10). The answer is in an older session, and at 494 turns per history, recency can't reach it.
2. **Embeddings alone don't beat BM25.** The difference is not significant at 10 results, and it is significantly *worse* at 5 (−7.0 points on session-all@5). The hybrid is what wins.
3. **The shipped hybrid beats BM25 significantly** at 10 results (+1.7 session-any, +3.2 session-all). It does not at 5.
4. **The reranker matters when the prompt budget is small.** Against plain hybrid it adds +4.0 points on session-all@5 (CI [+2.1, +6.2]) and +1.7 on turn-any@10. At 10 results on session-level metrics, the gain over plain hybrid is not significant.
5. **Recency and importance weighting make no measurable difference here** (shipped weights vs relevance only: −0.4 on session-all@10, CI [−1.5, +0.6]). LongMemEval's temporal questions need reasoning over dates, which these two scores don't capture. They are neutral on this benchmark, not a source of the gains.
6. **Where it still fails:** 53 of 470 questions miss at least one answer session at K=10 (multi-session 25, temporal-reasoning 26, preference 2). 47 of those 53 found at least one. At K=30 only 14 questions still miss. The remaining gap is multi-session coverage, which a larger budget or per-session retrieval would address.
7. **Granularity:** the index is per turn, with sessions inferred from the turns retrieved. Published LongMemEval retrieval numbers mostly index whole sessions or rounds, so compare relative gaps, not absolute numbers across papers.

### LongMemEval answer accuracy (reader: Qwen2.5-3B, local)

`benchmarks/longmemeval_qa.py` retrieves 10 turns, groups them by session with their dates, and has a reader LLM answer. The reader here is **Qwen2.5-3B-Instruct (Q4_K_M)** running on CPU through llama.cpp. The file is a Hugging Face download whose SHA-256 was verified. It uses a short-answer prompt, 60 questions (10 per type, abstention excluded), and a fixed random sample. All four systems get the same reader and the same prompt.

**Read these numbers with two caveats.** The reader is small: even the oracle, which is handed exactly the evidence sessions, answers only about a third of the questions correctly. And with 60 questions, each number carries a margin of roughly ±12 points (95%), so only large gaps are meaningful.

| system | judge (LLM-graded correct) | answer contains gold | token F1 | answered "unknown" |
|---|---|---|---|---|
| oracle (evidence sessions) | 36.7% | 28.3% | 15.7% | 11.7% |
| recency (chat window) | 3.3% | 0.0% | 3.2% | 81.7% |
| BM25 | 26.7% | 18.3% | 11.7% | 35.0% |
| **agentmem hybrid + reranker** | 31.7% | 25.0% | 14.4% | 20.0% |

The judge is the same 3B model, a noisy grader. The strict string-containment column agrees on the ordering.

Judge score by question type (10 questions each, so each cell moves in 10-point steps):

| system | single-session-user | single-session-assistant | single-session-preference | multi-session | knowledge-update | temporal-reasoning |
|---|---|---|---|---|---|---|
| oracle (evidence sessions) | 70.0% | 50.0% | 40.0% | 20.0% | 40.0% | 0.0% |
| recency (chat window) | 0.0% | 0.0% | 0.0% | 0.0% | 20.0% | 0.0% |
| BM25 | 30.0% | 50.0% | 20.0% | 30.0% | 30.0% | 0.0% |
| agentmem hybrid + reranker | 40.0% | 40.0% | 30.0% | 20.0% | 50.0% | 10.0% |

**What this does and doesn't show:**
- **Solid:** a chat window leaves the reader without the answer. It scores 3.3% and says "unknown" on 82% of questions. Every retrieval system lifts that by an order of magnitude.
- **Solid:** retrieval quality carries through to answers. agentmem (31.7%) sits close to the oracle (36.7%) and ahead of BM25 (26.7%).
- **Not established:** that agentmem beats BM25 at answering. The 5-point gap is 3 questions out of 60 and is well inside the noise. The retrieval benchmark above is where the significant gap lives.
- **The oracle is not a strict ceiling for this reader.** It receives whole sessions (often 20+ turns, each truncated to 500 characters), which can confuse a 3B model. agentmem beat it on knowledge-update and temporal-reasoning questions (50% vs 40% and 10% vs 0%).
- **Temporal reasoning is near zero for everyone,** including the oracle. A 3B model can't do the date arithmetic. This tests the reader, not memory.
- **For a publishable accuracy number,** run the same command with a stronger reader (`--llm claude:claude-sonnet-5-5 --judge --cot`, which needs an API key) and more questions per type.

## 4. Speed: two-stage retrieval

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

## 5. Business use case: coffee-shop support desk

The simulation has 20 customers over 8 weeks (812 turns) with direct and reworded questions. With MiniLM:
- **Answer in the model's context: 100%** on both question sets, against 27% for a last-20-messages window and 0% for a stateless bot.
- **Top-3 search on reworded questions: 56%** (46% with the built-in embedder). Reworded allergy questions score 0% with the built-in embedder and **100% with MiniLM**.
- **Caveat:** the data is synthetic and was written by us. See `results/support_desk_*.json`.

## 6. Bugs found by this round of testing

| problem | fix |
|---|---|
| `AgentMemory(SQLiteStorage(db))` treated the storage object as a file path. It silently created a new database named after the object's string representation, and those writes never reached the real database. | A storage backend is accepted as the first argument; a `path` that isn't a string or path raises `TypeError` |
| With MiniLM, de-duplication merged "likes dark roast" into "prefers light roast" (0.87 similarity) | Facts merge only when vectors **and** content words agree |
| Recency and importance outweighed relevance on LoCoMo | Weights tuned on the dev split, with relevance gating |

## 7. System tests

- `pytest`: **31/31** pass.
- `tests/e2e/run_e2e.py`: **18/18** pass, against a freshly built wheel in a clean venv with all extras.

## 8. Model provenance

- **Embeddings:** all-MiniLM-L6-v2 (Apache-2.0), from Chroma's public ONNX mirror, verified by SHA-256 `913d7300…16ec3`.
- **Reranker:** cross-encoder/ms-marco-MiniLM-L-6-v2 (Apache-2.0), Xenova's ONNX export.
  - **Default download:** Hugging Face, with the files verified by pinned SHA-256 (`c623d0bc…` for `model.onnx`).
  - **These tests:** Hugging Face is blocked here, so the same files were taken from an npm package that bundles them. Its npm integrity hash was verified and its code was never run. The weights matched the pinned hash and reproduced the published reference scores (+8.8 vs −11.2).

## 9. Known limits

- **Answer accuracy:** LoCoMo F1 and J are unmeasured. LongMemEval accuracy was measured only with a small 3B reader on 60 questions, so it supports ordering and large gaps but not absolute claims. Rerun with a stronger reader for headline numbers.
- **Index lifetime:** the index lives in the process's memory and is rebuilt once per process (about 1 s per 10,000 memories). For millions of memories, or a shared index across servers, implement `StorageBackend` on a vector database (pgvector, Qdrant).
- **Reranker cost:** it adds about 30 cross-encoder pairs per query on CPU.
- **Fact extraction:** the rule-based extractor is pattern-based. Use `use_llm_memory=True` for open-ended conversations.

## 10. Customer Support on Twitter (Kaggle): returning customers

Dataset: Kaggle `thoughtvector/customer-support-on-twitter` (CC BY-NC-SA 4.0, non-commercial use only). 1.54M customer messages from 702,669 authors, and 1.17M company replies linked to them. Harness: `benchmarks/twcs.py`. Full-corpus run: `benchmarks/twcs_scale.py`.

**Task.** A customer returns on a later day. Their message is the query, their earlier messages and the company's replies are the memory. Evidence is the earlier customer messages that share a distinctive word with the query (a word in at most 150 customer messages in the corpus). There are no human labels, so evidence is derived from word overlap. That favours keyword methods, which is the main caveat below.

**Two tracks.** *Full query* uses the customer's own words. *Distinctive words masked* removes those keywords from the query, so retrieval must work from the rest of the message.

### 10a. Per-customer memory (1,500 returning customers, 5,845 eligible)

| system | recall@10, full | recall@10, masked | recall@5, masked |
|---|---|---|---|
| recency window (last N messages) | 92.3% | 92.3% | 79.6% |
| BM25 | 99.1% | 96.9% | 89.7% |
| agentmem hybrid | 99.1% | 97.1% | **90.5%** (+0.8, CI [+0.1, +1.6]) |
| agentmem hybrid + reranker | 98.9% | 97.3% | **90.8%** (+1.2, CI [+0.4, +2.0]) |

Each customer's history is small (median 4 messages), so top-10 retrieval is close to saturated. Only the chat window is clearly worse. The reranker's gain is significant at 5 results only.

End-to-end lifecycle on the masked track (sessions, consolidation, SQLite, close and reopen):
- 16,954 history messages became 16,954 episodes, 145 facts, 4,795 sessions.
- Evidence recall@10 97.3%. The evidence text appeared in the prompt context for 92.8% of queries.
- Results were identical after closing and reopening the database (100%).
- Query latency p50 46 ms, p95 193 ms (includes the reranker). Database 43.5 MB.

### 10b. One brand's inbox in one shared memory (AmazonHelp, 309,698 messages)

Every message to one brand goes into a single namespace. Each customer is tagged with their own id. Each query uses the tag filter (this customer only) or no filter (whole inbox). This tests customer isolation and retrieval among thousands of other customers' messages.

| track | tag-isolated recall@10 | untagged recall@10 | untagged top-10 hits from other customers |
|---|---|---|---|
| full query (214 customers) | **99.5%** | 35.2% | 93.9% |
| distinctive words masked (223 customers) | **96.5%** | 12.2% | 97.3% |

Latency per query (two queries per customer, with reranker): p50 340 ms / 295 ms, p95 715 ms / 614 ms (full / masked).

**Bug found and fixed by this run.** The tag filter was applied after the index had already chosen its top 200 candidates. If other customers' messages outranked the target, the target was dropped. Isolated recall was 72% (full) and 31% (masked) before the fix. The index now applies the filter before the cut, and a regression test covers it.

**Second fix.** Adding a message scanned and copied the whole namespace to check novelty, so ingestion slowed quadratically. Writes now use a bounded cache of recent episodes. 20,000 writes take about 27 seconds.

### 10c. Full corpus: 2.70M tweets, 702,669 tenants (one SQLite file)

`python benchmarks/twcs_scale.py`, HashingEmbedder (so the run measures storage and index, not the model).

| measure | result |
|---|---|
| Tweets ingested (customer messages and linked replies) | 2,702,208 |
| Ingest throughput | 4,090 tweets/s (661 s total) |
| Database size | 11.3 GB |
| Evidence recall@10, 3,000 returning customers | 98.4% (any@10 99.5%) |
| Hits that came from the customer's own namespace | 100% |
| Query latency across tenants, p50 / p95 | 1.3 ms / 3.1 ms |

### 10d. Limits
- **Evidence is lexical.** Word overlap defines what should be found, so keyword methods start with an advantage. The masked track reduces this but does not remove it. Real answer labels would need annotation or an LLM judge.
- **Per-customer tests saturate.** Typical histories are small, so the 10a differences are small. The brand inbox (10b) is the harder test.
- **The scale run used the HashingEmbedder.** Model quality at scale was measured in 10a and 10b only.
- **Memory.** A 310k-message namespace stored as Python objects exceeded 15 GB and was killed. Brand runs now use SQLite, which stores embeddings as compact float32 blobs.
- **Licence.** CC BY-NC-SA 4.0 is non-commercial. Results built on this dataset should carry that restriction.
