"""Grid-search retrieval settings on a LoCoMo *dev* split (default: conversation 1).

Report final numbers on the held-out conversations with
``python benchmarks/locomo.py --conversations 2-10``.
"""
from __future__ import annotations

import argparse
import itertools
import json
import re
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import locomo as L  # noqa: E402
from agentmem import RetrievalWeights  # noqa: E402


def evidence(q):
    return {x.strip() for y in (q.get("evidence") or []) for x in re.split(r"[;,]", y) if x.strip()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--embedder", default="minilm", choices=["hashing", "minilm"])
    ap.add_argument("--dev", default="1")
    ap.add_argument("--out", default=str(L.HERE / "results" / "tune.json"))
    a = ap.parse_args()
    data = [L.load(L.HERE / "data" / "locomo10.json")[i - 1] for i in L.parse_range(a.dev)]
    emb = L.make_embedder(a.embedder)
    mems = [(s, L.ingest_agentmem(s, emb)) for s in data]
    qas = [(m, q) for s, m in mems for q in s["qa"] if q["category"] != 5 and evidence(q)]
    grid = itertools.product([0.3, 0.5, 0.7, 0.8], ["weighted", "rrf"], [0.75, 1.0], [1.0, 0.5, 0.2, 0.0], [0, 1])
    rows = []
    t0 = time.perf_counter()
    for lam, fusion, gate, ri, expand in grid:
        w = RetrievalWeights(lexical=lam, fusion=fusion, relevance_gate=gate, recency=0.5 * ri, importance=0.7 * ri)
        rec = {5: [], 10: []}
        for m, q in qas:
            got = L.agentmem_retriever(m, w, expand)(q["question"], 10)
            ev = evidence(q)
            for k in rec:
                rec[k].append(len(set(got[:k]) & ev) / len(ev))
        rows.append({"lexical": lam, "fusion": fusion, "gate": gate, "recency_importance_scale": ri,
                     "expand": expand, "recall@5": 100 * statistics.mean(rec[5]),
                     "recall@10": 100 * statistics.mean(rec[10])})
    rows.sort(key=lambda r: (r["recall@10"], r["recall@5"]), reverse=True)
    print(f"{len(rows)} configs on dev conversations {a.dev} ({len(qas)} questions, {time.perf_counter()-t0:.0f}s)")
    for r in rows[:8]:
        print(json.dumps(r))
    print("... worst:", json.dumps(rows[-1]))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"embedder": a.embedder, "dev": a.dev, "rows": rows}, indent=2))


if __name__ == "__main__":
    main()
