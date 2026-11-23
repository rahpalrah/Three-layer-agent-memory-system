"""Judged relevance for the returning-customer test: removes the word-overlap bias.

The lexical evidence in twcs.py is defined by shared words, so keyword search is favoured.
This script instead uses *pooled judgments* (standard in IR evaluation):

  export  for long-history returning customers, with the distinctive words removed from the
          query (so the keyword shortcut is gone), run every system, pool each system's top-10
          customer messages, and write the pool to judged/pool.jsonl for judging.
  judge   read judged/judgments.json ({cid: {tid: 0|1}}) and score every system on the pool:
          precision@10 and judged recall@10 (relevant retrieved / relevant in the pool).

Judging: relevance is decided by a language model reading the query, the earlier message and the
company reply, not by word overlap. It is a single judge, not a human panel, so treat it as an
independent check on the lexical benchmark rather than as ground truth.

    python benchmarks/twcs_judged.py export --customers 30 --min-messages 15
    python benchmarks/twcs_judged.py judge
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import twcs as T  # noqa: E402
import locomo as L  # noqa: E402
from longmemeval import CachedEmbedder  # noqa: E402
from agentmem import AgentMemory, InMemoryStorage  # noqa: E402

JUDGED = HERE / "data" / "twcs" / "judged"


def systems_hits(p, inbound, reply_of, embedder, reranker):
    """Top-10 customer tids per system, deduplicated, using the same code paths as twcs.py."""
    eps = T.history_episodes(p, inbound, reply_of)
    if not eps:
        return {}
    clock = {"t": 0.0}
    mem = AgentMemory(storage=InMemoryStorage(), namespace=p["cid"], embedder=embedder,
                      clock=lambda: clock["t"], auto_consolidate=False)
    T.ingest(mem, clock, eps)
    clock["t"] = p["query_ts"]
    docs = [e["text"] for e in eps]
    out = {}
    out["recency window"] = [(e["tid"], e["parent"]) for e in reversed(eps)][:10]
    out["BM25"] = [(eps[i]["tid"], eps[i]["parent"]) for i in L.BM25(docs).top(p["query"], 10)]
    mem.reranker = None
    out["agentmem hybrid"] = T.hit_tuples(mem, mem.recall(p["query"], k=10, reinforce=False))
    mem.reranker = reranker
    out["agentmem hybrid + reranker"] = T.hit_tuples(mem, mem.recall(p["query"], k=10, reinforce=False))
    # every system is scored on customer messages: a company reply counts for its customer message
    return {name: [tid_customer(t, par) for t, par in hits if tid_customer(t, par)][:10]
            for name, hits in out.items()}


def tid_customer(tid, parent):
    return parent if parent else tid


def export(a):
    inbound, reply_of = T.load(Path(a.data))
    df = T.token_df(inbound)
    probes = [p for p in T.build_probes(inbound, df)
              if len(inbound[p["cid"]]) >= a.min_messages]
    rng = random.Random(a.seed)
    probes = rng.sample(probes, min(a.customers, len(probes)))
    # the keyword shortcut is removed: the query no longer contains distinctive words
    probes = [dict(p, query_original=p["query"], query=T.mask_query(p["query"], df)) for p in probes]
    probes = [p for p in probes if T.tokenize(p["query"])]
    emb = CachedEmbedder(L.make_embedder("minilm"), T.HERE / "data" / "twcs" / "minilm_cache")
    from agentmem.rerank import CrossEncoderReranker
    rr = CrossEncoderReranker()
    text_of = {}
    for cid in {p["cid"] for p in probes}:
        for h in inbound[cid]:
            text_of[h["tid"]] = (h["text"], reply_of.get(h["tid"], {}).get("text", ""))
    JUDGED.mkdir(parents=True, exist_ok=True)
    rows = []
    for p in probes:
        hits = systems_hits(p, inbound, reply_of, emb, rr)
        pool = []
        for sysname, tids in hits.items():
            for tid in tids:
                if tid not in [x["tid"] for x in pool]:
                    msg, reply = text_of.get(tid, ("", ""))
                    pool.append({"tid": tid, "message": msg, "company_reply": reply})
        rows.append({"cid": p["cid"], "query": p["query"], "query_original": p["query_original"] if "query_original" in p else None,
                     "pool": pool, "systems": hits})
    emb.save()
    (JUDGED / "pool.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    print(f"{len(rows)} queries, {sum(len(r['pool']) for r in rows)} pooled judgments needed -> {JUDGED / 'pool.jsonl'}")


def judge(a):
    rows = [json.loads(l) for l in (JUDGED / "pool.jsonl").read_text().splitlines()]
    judgments = json.loads((JUDGED / "judgments.json").read_text())
    res = {}
    for r in rows:
        rel = {tid for tid, v in judgments.get(r["cid"], {}).items() if v == 1}
        pooled_rel = rel & {x["tid"] for x in r["pool"]}
        if not pooled_rel:
            continue
        for sysname, tids in r["systems"].items():
            top = tids[:10]
            res.setdefault(sysname, {"p10": [], "r10": []})
            res[sysname]["p10"].append(len([t for t in top if t in rel]) / 10)
            res[sysname]["r10"].append(len([t for t in top if t in pooled_rel]) / len(pooled_rel))
    n = sum(1 for r in rows if any(v == 1 for v in judgments.get(r["cid"], {}).values()))
    lines = [f"Judged relevance, {n} queries with at least one relevant message (keywords removed from queries; "
             f"pool = union of each system's top 10; single LLM judge):\n",
             "| system | precision@10 | judged recall@10 |", "|---|---|---|"]
    out = {}
    for name, v in res.items():
        p, r = 100 * statistics.mean(v["p10"]), 100 * statistics.mean(v["r10"])
        out[name] = {"precision@10": p, "judged_recall@10": r, "queries": len(v["p10"])}
        lines.append(f"| {name} | {p:.1f}% | {r:.1f}% |")
    text = "\n".join(lines)
    print(text)
    (HERE / "results" / "twcs_judged.json").write_text(json.dumps({"queries": n, "systems": out}, indent=2))
    (HERE / "results" / "twcs_judged.md").write_text(text + "\n")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["export", "judge"])
    ap.add_argument("--data", default=str(T.DATA))
    ap.add_argument("--customers", type=int, default=30)
    ap.add_argument("--min-messages", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    export(a) if a.command == "export" else judge(a)


if __name__ == "__main__":
    main()
