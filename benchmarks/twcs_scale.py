"""Customer Support on Twitter at full scale: every customer message and company reply
(about 2.8M tweets) ingested into one SQLite database, one namespace per customer.

Measures what matters for a deployed multi-tenant memory: ingest throughput, database size,
and per-query latency across tenants, plus whether each customer's earlier messages are
retrieved. Embeddings are the dependency-free HashingEmbedder so the run measures the storage
and index path rather than model inference (see twcs.py for retrieval quality with MiniLM).

    python benchmarks/twcs_scale.py
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import twcs as T  # noqa: E402
from agentmem import (AgentMemory, HashingEmbedder, HeuristicImportanceScorer, Layer,  # noqa: E402
                      MemoryItem, SQLiteStorage)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(T.DATA))
    ap.add_argument("--queries", type=int, default=3000)
    ap.add_argument("--db", default=str(HERE / "data" / "twcs" / "scale.db"))
    ap.add_argument("--out", default=str(HERE / "results" / "twcs_scale.json"))
    a = ap.parse_args(argv)

    t0 = time.perf_counter()
    inbound, reply_of = T.load(Path(a.data))
    df = T.token_df(inbound)
    probes = T.build_probes(inbound, df)
    load_s = time.perf_counter() - t0
    print(f"loaded in {load_s:.0f}s: {sum(map(len, inbound.values())):,} customer messages, "
          f"{len(reply_of):,} company replies, {len(probes):,} returning customers", file=sys.stderr, flush=True)

    db = Path(a.db)
    db.parent.mkdir(parents=True, exist_ok=True)
    if db.exists():
        db.unlink()
    st = SQLiteStorage(db)
    emb = HashingEmbedder()
    scorer = HeuristicImportanceScorer()
    t = time.perf_counter()
    stored = 0
    for n, (cid, items) in enumerate(inbound.items(), 1):
        eps = []
        for h in items:
            eps.append((h["ts"], h["text"], "user", h["tid"], None))
            rp = reply_of.get(h["tid"])
            if rp:
                eps.append((rp["ts"], rp["text"], "support", rp["tid"], h["tid"]))
        if not eps:
            continue
        eps.sort()
        # written directly so each item keeps its real tweet timestamp (remember_many would stamp
        # the whole batch with one time)
        st.put_many([MemoryItem(content=text, layer=Layer.EPISODIC, namespace=cid,
                                importance=scorer.score(text, role=role), embedding=emb.embed(text),
                                created_at=ts, last_accessed=ts,
                                metadata={"tid": tid, "parent": parent, "role": role})
                     for ts, text, role, tid, parent in eps])
        stored += len(eps)
        if n % 100000 == 0:
            print(f"  {n:,} customers, {stored:,} messages ({stored / (time.perf_counter() - t):,.0f}/s)",
                  file=sys.stderr, flush=True)
    ingest_s = time.perf_counter() - t
    st.close()
    db_mb = db.stat().st_size / 1e6
    print(f"ingested {stored:,} tweets in {ingest_s:.0f}s; database {db_mb:.0f} MB", file=sys.stderr, flush=True)

    # --- queries: fresh object per request (a new process would see the same), index on
    st = SQLiteStorage(db)
    rng = random.Random(0)
    sample = rng.sample(probes, min(a.queries, len(probes)))
    lat, rec, any_, same_tenant, n_hits = [], [], [], 0, 0
    for p in sample:
        mem = AgentMemory(st, namespace=p["cid"], embedder=emb, auto_consolidate=False, clock=lambda: p["query_ts"])
        t = time.perf_counter()
        hits = mem.recall(p["query"], k=10, reinforce=False)
        lat.append((time.perf_counter() - t) * 1000)
        tuples = T.hit_tuples(mem, hits)
        n_hits += len(hits)
        c = T.covered(tuples, p["evidence"])
        rec.append(len(c) / len(p["evidence"]))
        any_.append(1.0 if c else 0.0)
        same_tenant += sum(1 for h in hits if h.item.namespace == p["cid"])
    lat.sort()
    result = {
        "tweets_ingested": stored, "customers": len(inbound), "returning_customers": len(probes),
        "load_seconds": round(load_s, 1), "ingest_seconds": round(ingest_s, 1),
        "ingest_per_second": round(stored / ingest_s), "database_mb": round(db_mb, 1),
        "queries": len(sample), "evidence_recall_at_10": 100 * statistics.mean(rec),
        "any_at_10": 100 * statistics.mean(any_),
        "hits_from_own_namespace": 100 * same_tenant / max(1, n_hits),
        "query_ms_p50": round(lat[len(lat) // 2], 2), "query_ms_p95": round(lat[int(0.95 * (len(lat) - 1))], 2),
    }
    st.close()
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    print(f"wrote {out}")
    return result


if __name__ == "__main__":
    main()
