LoCoMo retrieval - 1386 questions, 5463 dialogue turns
agentmem kept 5460 episodes and 314 facts; ran in 1922s

| system | recall@5 | recall@10 | recall@25 | hit@5 | hit@10 | hit@25 |
|---|---|---|---|---|---|---|
| random | 0.8% | 1.7% | 4.8% | 1.2% | 2.7% | 6.6% |
| recency (chat window) | 0.9% | 1.6% | 3.3% | 1.2% | 1.9% | 4.1% |
| BM25 | 50.1% | 57.8% | 67.1% | 55.9% | 64.1% | 74.5% |
| agentmem [hashing] v0 defaults (dense only) | 30.8% | 37.6% | 48.7% | 34.6% | 42.3% | 54.8% |
| agentmem [hashing] v1 defaults (hybrid, untuned weights) | 48.9% | 55.7% | 66.5% | 54.6% | 62.6% | 74.2% |
| agentmem [hashing] dense only | 32.5% | 39.8% | 49.3% | 36.3% | 44.7% | 55.5% |
| agentmem [hashing] hybrid RRF | 43.0% | 51.2% | 62.3% | 47.8% | 57.4% | 69.9% |
| agentmem [hashing] SHIPPED: hybrid (tuned) | 49.6% | 57.5% | 67.2% | 55.3% | 64.4% | 74.9% |
| agentmem [hashing] SHIPPED + neighbours | 52.5% | 61.0% | 69.9% | 58.1% | 67.7% | 77.4% |
| agentmem [hashing] SHIPPED + cross-encoder rerank | 59.8% | 64.0% | 68.4% | 67.0% | 71.1% | 76.1% |
| agentmem [minilm] v0 defaults (dense only) | 23.9% | 35.6% | 52.8% | 28.2% | 40.7% | 60.2% |
| agentmem [minilm] v1 defaults (hybrid, untuned weights) | 50.8% | 60.1% | 70.5% | 56.8% | 67.3% | 77.9% |
| agentmem [minilm] dense only | 33.0% | 43.6% | 57.6% | 37.1% | 49.5% | 65.2% |
| agentmem [minilm] hybrid RRF | 46.6% | 57.1% | 69.8% | 52.8% | 64.1% | 77.3% |
| agentmem [minilm] SHIPPED: hybrid (tuned) | 54.8% | 63.3% | 73.2% | 61.2% | 70.6% | 81.0% |
| agentmem [minilm] SHIPPED + neighbours | 53.3% | 63.1% | 72.4% | 59.7% | 70.1% | 79.6% |
| agentmem [minilm] SHIPPED + cross-encoder rerank | 63.1% | 68.9% | 74.7% | 70.0% | 76.6% | 82.0% |

recall@10 by question type:

| system | multi-hop (n=250) | temporal (n=284) | open-domain (n=81) | single-hop (n=771) |
|---|---|---|---|---|
| random | 1.5% | 3.1% | 1.3% | 1.3% |
| recency (chat window) | 0.6% | 2.1% | 0.3% | 1.8% |
| BM25 | 30.5% | 66.0% | 25.9% | 66.9% |
| agentmem [hashing] v0 defaults (dense only) | 13.9% | 42.1% | 18.2% | 45.6% |
| agentmem [hashing] v1 defaults (hybrid, untuned weights) | 29.1% | 63.8% | 27.1% | 64.3% |
| agentmem [hashing] dense only | 14.5% | 43.7% | 16.9% | 49.0% |
| agentmem [hashing] hybrid RRF | 23.7% | 57.4% | 24.6% | 60.6% |
| agentmem [hashing] SHIPPED: hybrid (tuned) | 30.9% | 65.6% | 27.7% | 66.3% |
| agentmem [hashing] SHIPPED + neighbours | 28.8% | 66.0% | 26.4% | 73.2% |
| agentmem [hashing] SHIPPED + cross-encoder rerank | 40.0% | 70.2% | 29.6% | 73.2% |
| agentmem [minilm] v0 defaults (dense only) | 19.2% | 39.9% | 21.1% | 40.8% |
| agentmem [minilm] v1 defaults (hybrid, untuned weights) | 34.7% | 66.5% | 33.7% | 68.8% |
| agentmem [minilm] dense only | 24.4% | 48.7% | 24.1% | 49.9% |
| agentmem [minilm] hybrid RRF | 33.8% | 63.6% | 28.9% | 65.3% |
| agentmem [minilm] SHIPPED: hybrid (tuned) | 36.9% | 68.9% | 35.2% | 72.7% |
| agentmem [minilm] SHIPPED + neighbours | 32.1% | 67.6% | 27.7% | 75.2% |
| agentmem [minilm] SHIPPED + cross-encoder rerank | 44.9% | 71.5% | 37.9% | 78.9% |
