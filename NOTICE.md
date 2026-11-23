# Notices and third-party licences

## agentmem (this repository)
The library, tests and benchmark code are released under the MIT licence (see `LICENSE`).

## Datasets used in benchmarks (not redistributed)
Datasets are downloaded by the user and are not stored in this repository (see `.gitignore`).

| Dataset | Source | Licence | Use restriction |
|---|---|---|---|
| Customer Support on Twitter | Kaggle `thoughtvector/customer-support-on-twitter` | **CC BY-NC-SA 4.0** | **Non-commercial only.** Derivatives must carry the same licence. Attribution required. |
| LoCoMo | github.com/snap-research/locomo | Not verified here: check the upstream repository | Follow the upstream terms |
| LongMemEval | github.com/xiaowu0162/LongMemEval | Not verified here: check the upstream repository | Follow the upstream terms |

**Results derived from the Customer Support on Twitter dataset** (`benchmarks/results/twcs*.json`, `twcs_judged.*`, and section 10 of `benchmarks/RESULTS.md`) are statistics computed from that data. They contain no tweet text. Any report, product or publication built on them inherits the CC BY-NC-SA 4.0 non-commercial restriction and must credit the dataset.

## Models
- all-MiniLM-L6-v2 (Apache-2.0), downloaded from Chroma's public mirror, SHA-256 verified.
- cross-encoder/ms-marco-MiniLM-L-6-v2 (Apache-2.0), Xenova ONNX export, SHA-256 verified.
- Qwen2.5-3B-Instruct (Qwen licence), used only for the LongMemEval reader experiment.
