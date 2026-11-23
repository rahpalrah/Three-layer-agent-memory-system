Customer Support on Twitter - 5,845 eligible returning customers, embedder=minilm, reranker=True

### track: full query (1,500 customers)

| system | evidence recall@5 | any@5 | evidence recall@10 | any@10 |
|---|---|---|---|---|
| recency window | 79.6% | 87.4% | 92.3% | 96.4% |
| BM25 | 96.4% | 99.2% | 99.1% | 100.0% |
| agentmem hybrid | 96.4% | 99.2% | 99.1% | 100.0% |
| agentmem hybrid + reranker | 96.1% | 98.9% | 98.9% | 99.9% |

paired bootstrap vs BM25 (evidence recall, percentage points, 95% CI):

| system | @5 difference | @10 difference |
|---|---|---|
| agentmem hybrid | +0.0 [-0.4, +0.5] (n.s.) | -0.0 [-0.2, +0.1] (n.s.) |
| agentmem hybrid + reranker | -0.3 [-0.8, +0.3] (n.s.) | -0.2 [-0.5, +0.0] (n.s.) |

### track: distinctive words masked (1,500 customers)

| system | evidence recall@5 | any@5 | evidence recall@10 | any@10 |
|---|---|---|---|---|
| recency window | 79.6% | 87.4% | 92.3% | 96.4% |
| BM25 | 89.7% | 94.8% | 96.9% | 99.0% |
| agentmem hybrid | 90.5% | 95.5% | 97.1% | 99.0% |
| agentmem hybrid + reranker | 90.8% | 95.7% | 97.3% | 99.0% |

paired bootstrap vs BM25 (evidence recall, percentage points, 95% CI):

| system | @5 difference | @10 difference |
|---|---|---|
| agentmem hybrid | +0.8 [+0.1, +1.6] | +0.2 [-0.4, +0.7] (n.s.) |
| agentmem hybrid + reranker | +1.2 [+0.4, +2.0] | +0.4 [-0.2, +1.0] (n.s.) |

end-to-end lifecycle on the distinctive-words-masked track (sessions, consolidation, SQLite, reopen):

- 1500 customers, 16,954 history messages -> 16,954 episodes kept, 145 facts, 4,795 sessions
- evidence recall@10 97.3%, any@10 99.0%
- evidence text present in the prompt context: 92.8% (average context 620 chars)
- identical results after closing and reopening the database: 100.0%
- query latency p50 46.0 ms, p95 192.7 ms; database 43.5 MB; ingest 119.2s
