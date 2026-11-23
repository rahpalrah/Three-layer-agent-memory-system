Judged relevance, 28 queries with at least one relevant message (keywords removed from queries; pool = union of each system's top 10; single LLM judge):

| system | precision@10 | judged recall@10 |
|---|---|---|
| recency window | 83.2% | 76.2% |
| BM25 | 82.9% | 78.2% |
| agentmem hybrid | 82.5% | 77.3% |
| agentmem hybrid + reranker | 82.1% | 77.1% |
