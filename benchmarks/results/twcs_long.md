Customer Support on Twitter - 463 eligible returning customers, embedder=minilm, reranker=True

### track: full query (463 customers)

| system | evidence recall@5 | any@5 | evidence recall@10 | any@10 |
|---|---|---|---|---|
| recency window | 39.0% | 65.2% | 53.2% | 77.5% |
| BM25 | 74.1% | 95.2% | 87.4% | 98.7% |
| agentmem hybrid | 73.9% | 94.0% | 88.1% | 98.7% |
| agentmem hybrid + reranker | 74.8% | 93.5% | 86.7% | 97.6% |

paired bootstrap vs BM25 (evidence recall, percentage points, 95% CI):

| system | @5 difference | @10 difference |
|---|---|---|
| agentmem hybrid | -0.2 [-2.0, +1.6] (n.s.) | +0.7 [-0.5, +1.9] (n.s.) |
| agentmem hybrid + reranker | +0.7 [-1.4, +2.7] (n.s.) | -0.7 [-2.5, +1.0] (n.s.) |

### track: distinctive words masked (463 customers)

| system | evidence recall@5 | any@5 | evidence recall@10 | any@10 |
|---|---|---|---|---|
| recency window | 39.0% | 65.2% | 53.2% | 77.5% |
| BM25 | 50.4% | 76.9% | 69.3% | 89.4% |
| agentmem hybrid | 51.0% | 77.3% | 70.4% | 89.0% |
| agentmem hybrid + reranker | 51.4% | 77.1% | 69.9% | 88.6% |

paired bootstrap vs BM25 (evidence recall, percentage points, 95% CI):

| system | @5 difference | @10 difference |
|---|---|---|
| agentmem hybrid | +0.7 [-1.5, +3.0] (n.s.) | +1.1 [-1.1, +3.2] (n.s.) |
| agentmem hybrid + reranker | +1.0 [-1.5, +3.6] (n.s.) | +0.6 [-2.1, +3.1] (n.s.) |


peak memory of the harness process (corpus held in Python): 3.6 GB
