"""Recall latency vs. memory size: full scan vs. stage-1 index (numpy exact, FAISS HNSW).

    python benchmarks/scale.py --sizes 1000 10000 50000 100000
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import tempfile
import time
from pathlib import Path

from agentmem import AgentMemory

WORDS = ("billing invoice refund kubernetes cluster deploy hiring interview candidate marketing "
         "campaign launch security audit password travel flight hotel design mockup legal contract "
         "review budget forecast roadmap incident outage customer churn pricing onboarding").split()


def make_texts(n, rng):
    return [f"{' '.join(rng.sample(WORDS, 4))} note {i} ref{rng.randint(0, n // 10)}" for i in range(n)]


def timed(fn, reps):
    lat = []
    for _ in range(reps):
        t = time.perf_counter(); fn(); lat.append((time.perf_counter() - t) * 1000)
    return lat


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[1000, 10000, 50000, 100000])
    ap.add_argument("--queries", type=int, default=30)
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "results" / "scale.json"))
    a = ap.parse_args()
    rng = random.Random(0)
    queries = [" ".join(rng.sample(WORDS, 2)) for _ in range(a.queries)]
    rows = []
    for n in a.sizes:
        with tempfile.TemporaryDirectory() as d:
            db = Path(d) / "m.db"
            w = AgentMemory(db, index=None, auto_consolidate=False)
            t = time.perf_counter()
            texts = make_texts(n, rng)
            for i in range(0, n, 5000):
                w.remember_many(texts[i:i + 5000], importance=[rng.random() for _ in texts[i:i + 5000]])
            ingest = time.perf_counter() - t
            w.close()
            exact = AgentMemory(db, index=None, auto_consolidate=False)
            truth = {q: [h.item.id for h in exact.recall(q, k=10, reinforce=False)] for q in queries[:5]}
            row = {"memories": n, "ingest_s": round(ingest, 2), "db_mb": round(db.stat().st_size / 1e6, 1)}
            reps_full = 3 if n >= 50000 else 10
            row["full_scan_ms_p50"] = statistics.median(timed(lambda: exact.recall(queries[0], k=10, reinforce=False), reps_full))
            for backend in ("numpy", "faiss"):
                try:
                    m = AgentMemory(db, index=backend, auto_consolidate=False)
                except ImportError:
                    continue
                t = time.perf_counter(); m.recall(queries[0], k=10, reinforce=False); cold = (time.perf_counter() - t) * 1000
                lat = []
                for q in queries:
                    lat += timed(lambda: m.recall(q, k=10, reinforce=False), 1)
                agree = statistics.mean(len(set(truth[q]) & {h.item.id for h in m.recall(q, k=10, reinforce=False)}) / 10
                                        for q in truth)
                row[f"{backend}_cold_build_ms"] = round(cold)
                row[f"{backend}_ms_p50"] = round(statistics.median(lat), 1)
                row[f"{backend}_ms_p95"] = round(sorted(lat)[int(0.95 * (len(lat) - 1))], 1)
                row[f"{backend}_top10_agreement"] = round(100 * agree, 1)
                m.close()
            row["full_scan_ms_p50"] = round(row["full_scan_ms_p50"], 1)
            exact.close()
        rows.append(row)
        print(json.dumps(row), flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(rows, indent=2))
    print("\n| memories | full scan p50 | index (numpy) p50 / p95 | index (FAISS HNSW) p50 / p95 | HNSW top-10 agreement | index build (cold) |")
    print("|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['memories']:,} | {r['full_scan_ms_p50']:,.0f} ms | {r.get('numpy_ms_p50','-')} / {r.get('numpy_ms_p95','-')} ms | "
              f"{r.get('faiss_ms_p50','-')} / {r.get('faiss_ms_p95','-')} ms | {r.get('faiss_top10_agreement','-')}% | {r.get('numpy_cold_build_ms','-'):,} ms |")


if __name__ == "__main__":
    sys.exit(main())
