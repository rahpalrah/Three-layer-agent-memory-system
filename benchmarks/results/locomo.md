LoCoMo retrieval - 1536 questions, 5882 dialogue turns, 10 conversations
agentmem kept 5879 episodes and 337 facts; ran in 114s

| system | recall@5 | recall@10 | recall@25 | hit@5 | hit@10 | hit@25 |
|---|---|---|---|---|---|---|
| random | 1.1% | 2.0% | 5.0% | 1.7% | 3.1% | 7.3% |
| recency | 0.9% | 1.6% | 3.4% | 1.2% | 1.9% | 4.1% |
| bm25 | 50.2% | 57.9% | 67.1% | 55.9% | 64.1% | 74.2% |
| agentmem (relevance only) | 49.9% | 57.6% | 67.7% | 55.9% | 64.0% | 75.1% |
| agentmem | 48.8% | 56.2% | 66.3% | 54.2% | 63.0% | 73.6% |

recall@10 by question type:

| system | multi-hop (n=282) | temporal (n=321) | open-domain (n=92) | single-hop (n=841) |
|---|---|---|---|---|
| random | 2.7% | 2.4% | 2.0% | 1.7% |
| recency | 0.9% | 1.9% | 0.3% | 1.8% |
| bm25 | 31.5% | 68.0% | 26.6% | 66.4% |
| agentmem (relevance only) | 31.1% | 67.9% | 26.8% | 65.9% |
| agentmem | 29.2% | 66.2% | 28.2% | 64.5% |
