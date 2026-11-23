LoCoMo retrieval - 1386 questions, 5463 dialogue turns
agentmem kept 5459 episodes and 253 facts; ran in 39s

| system | recall@5 | recall@10 | recall@25 | hit@5 | hit@10 | hit@25 |
|---|---|---|---|---|---|---|
| random | 0.8% | 1.7% | 4.8% | 1.2% | 2.7% | 6.6% |
| recency (chat window) | 0.9% | 1.6% | 3.3% | 1.2% | 1.9% | 4.1% |
| BM25 | 50.1% | 57.8% | 67.1% | 55.9% | 64.1% | 74.5% |
| agentmem [self] v0 defaults (dense only) | 36.7% | 45.5% | 57.3% | 41.4% | 51.5% | 64.8% |
| agentmem [self] v1 defaults (hybrid, untuned weights) | 47.8% | 56.1% | 66.0% | 53.5% | 63.2% | 74.0% |
| agentmem [self] dense only | 45.2% | 52.0% | 61.0% | 50.9% | 58.6% | 68.7% |
| agentmem [self] hybrid RRF | 47.0% | 56.8% | 67.7% | 52.7% | 63.8% | 75.3% |
| agentmem [self] SHIPPED: hybrid (tuned) | 50.5% | 58.3% | 68.0% | 56.6% | 65.4% | 75.7% |
| agentmem [self] SHIPPED + neighbours | 51.8% | 61.0% | 70.6% | 57.9% | 67.8% | 77.8% |

recall@10 by question type:

| system | multi-hop (n=250) | temporal (n=284) | open-domain (n=81) | single-hop (n=771) |
|---|---|---|---|---|
| random | 1.5% | 3.1% | 1.3% | 1.3% |
| recency (chat window) | 0.6% | 2.1% | 0.3% | 1.8% |
| BM25 | 30.5% | 66.0% | 25.9% | 66.9% |
| agentmem [self] v0 defaults (dense only) | 21.1% | 55.1% | 23.6% | 52.2% |
| agentmem [self] v1 defaults (hybrid, untuned weights) | 29.7% | 65.2% | 27.8% | 64.4% |
| agentmem [self] dense only | 27.0% | 60.3% | 23.8% | 60.0% |
| agentmem [self] hybrid RRF | 30.3% | 66.6% | 26.0% | 65.0% |
| agentmem [self] SHIPPED: hybrid (tuned) | 32.0% | 67.3% | 28.4% | 66.6% |
| agentmem [self] SHIPPED + neighbours | 29.1% | 65.8% | 26.7% | 73.2% |
