LongMemEval-S retrieval - 470 questions (abstention excluded), turn-level index, embedder=minilm

| system | sess-any@5 | sess-all@5 | turn-any@5 | sess-any@10 | sess-all@10 | turn-any@10 | sess-any@30 | sess-all@30 | turn-any@30 |
|---|---|---|---|---|---|---|---|---|---|
| recency (chat window) | 8.1% | 1.7% | 2.3% | 11.5% | 2.6% | 4.9% | 16.4% | 4.9% | 12.1% |
| BM25 | 94.3% | 75.5% | 81.9% | 96.6% | 84.7% | 88.5% | 98.9% | 94.3% | 93.0% |
| agentmem dense only | 95.5% | 68.5% | 75.5% | 97.2% | 83.0% | 86.8% | 99.4% | 95.1% | 96.2% |
| agentmem hybrid (shipped) | 97.2% | 77.7% | 83.8% | 98.3% | 87.9% | 90.9% | 100.0% | 97.0% | 97.0% |
| agentmem hybrid, relevance only | 97.2% | 77.7% | 84.0% | 98.5% | 88.3% | 91.1% | 100.0% | 96.8% | 96.8% |
| agentmem hybrid + reranker | 97.4% | 81.7% | 87.0% | 98.7% | 88.7% | 92.6% | 100.0% | 97.0% | 97.0% |

session-any@10 by question type:

| system | single-session-user (n=64) | single-session-assistant (n=56) | single-session-preference (n=30) | multi-session (n=121) | knowledge-update (n=72) | temporal-reasoning (n=127) |
|---|---|---|---|---|---|---|
| recency (chat window) | 4.7% | 8.9% | 0.0% | 11.6% | 13.9% | 17.3% |
| BM25 | 100.0% | 100.0% | 76.7% | 98.3% | 100.0% | 94.5% |
| agentmem dense only | 95.3% | 100.0% | 93.3% | 99.2% | 98.6% | 95.3% |
| agentmem hybrid (shipped) | 100.0% | 100.0% | 93.3% | 98.3% | 100.0% | 96.9% |
| agentmem hybrid, relevance only | 100.0% | 100.0% | 93.3% | 98.3% | 100.0% | 97.6% |
| agentmem hybrid + reranker | 100.0% | 100.0% | 93.3% | 99.2% | 100.0% | 97.6% |

session-all@10 by question type:

| system | single-session-user (n=64) | single-session-assistant (n=56) | single-session-preference (n=30) | multi-session (n=121) | knowledge-update (n=72) | temporal-reasoning (n=127) |
|---|---|---|---|---|---|---|
| recency (chat window) | 4.7% | 8.9% | 0.0% | 0.0% | 1.4% | 2.4% |
| BM25 | 100.0% | 100.0% | 76.7% | 71.9% | 97.2% | 77.2% |
| agentmem dense only | 95.3% | 100.0% | 93.3% | 71.1% | 97.2% | 70.1% |
| agentmem hybrid (shipped) | 100.0% | 100.0% | 93.3% | 77.7% | 100.0% | 78.0% |
| agentmem hybrid, relevance only | 100.0% | 100.0% | 93.3% | 78.5% | 98.6% | 79.5% |
| agentmem hybrid + reranker | 100.0% | 100.0% | 93.3% | 79.3% | 100.0% | 79.5% |

paired bootstrap vs BM25 (95% CI of the difference, percentage points; 2,000 resamples):

| system | metric | difference | 95% CI | significant |
|---|---|---|---|---|
| agentmem dense only | session-any@10 | +0.6 | [-1.5, +3.0] | no |
| agentmem dense only | session-all@10 | -1.7 | [-5.1, +1.5] | no |
| agentmem dense only | turn-any@10 | -1.7 | [-5.1, +1.7] | no |
| agentmem dense only | session-all@5 | -7.0 | [-11.3, -3.0] | yes |
| agentmem hybrid (shipped) | session-any@10 | +1.7 | [+0.4, +3.2] | yes |
| agentmem hybrid (shipped) | session-all@10 | +3.2 | [+1.1, +5.3] | yes |
| agentmem hybrid (shipped) | turn-any@10 | +2.3 | [+0.2, +4.7] | yes |
| agentmem hybrid (shipped) | session-all@5 | +2.1 | [-0.9, +5.1] | no |
| agentmem hybrid, relevance only | session-any@10 | +1.9 | [+0.4, +3.4] | yes |
| agentmem hybrid, relevance only | session-all@10 | +3.6 | [+1.5, +5.7] | yes |
| agentmem hybrid, relevance only | turn-any@10 | +2.6 | [+0.2, +4.9] | yes |
| agentmem hybrid, relevance only | session-all@5 | +2.1 | [-0.6, +4.9] | no |
| agentmem hybrid + reranker | session-any@10 | +2.1 | [+0.9, +3.6] | yes |
| agentmem hybrid + reranker | session-all@10 | +4.0 | [+1.9, +6.2] | yes |
| agentmem hybrid + reranker | turn-any@10 | +4.0 | [+1.9, +6.2] | yes |
| agentmem hybrid + reranker | session-all@5 | +6.2 | [+3.4, +8.9] | yes |

ingest 101s, retrieval 407s for 470 questions (144 ms per system per question)
