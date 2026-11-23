"""Customer Support on Twitter: returning-customer memory benchmark.

Dataset: Kaggle ``thoughtvector/customer-support-on-twitter`` (Stuart Axelbrooke, CC BY-NC-SA 4.0).
2.8M tweets from 702k authors to 109 brands, April-September 2016. Customer messages are
``inbound == True`` and carry an anonymised ``author_id``; company replies are linked back to
the customer tweet through ``in_response_to_tweet_id``.

Task: a customer who has written on an earlier day writes again. Can agentmem surface the
earlier messages (and the company's replies to them) that the new message refers to?

    1. Customers: authors with messages on >= 2 different days.
    2. Probe: the customer's last message is the query. Everything before it is the history
       (customer messages, each with the company reply that preceded the query date).
    3. Evidence: earlier customer messages sharing a *distinctive* token with the query. A token
       is distinctive when it appears in at most RARE_DF customer messages in the whole corpus.
       Evidence is derived from word overlap, not from human labels, so it favours keyword
       methods (see the caveat in RESULTS.md). It is the only label source the dataset offers.
    4. Metrics: evidence recall@k (share of evidence tweets covered by the top k hits, where a
       company reply covers the customer tweet it answers) and any@k (at least one covered).

Two runs:
    python benchmarks/twcs.py                       # retrieval quality, all systems
    python benchmarks/twcs.py --e2e                 # full lifecycle: sessions, consolidation,
                                                    #   SQLite, close + reopen, context, latency
"""
from __future__ import annotations

import argparse
import collections
import csv
import datetime as dt
import json
import random
import statistics
import sys
import tempfile
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import locomo as L  # noqa: E402  (BM25 baseline)
from longmemeval import CachedEmbedder  # noqa: E402
from agentmem import (AgentMemory, InMemoryStorage, Layer, RetrievalWeights,  # noqa: E402
                      SQLiteStorage)
from agentmem.embeddings import tokenize  # noqa: E402

DATA = HERE / "data" / "twcs" / "twcs.csv"
RARE_DF = 150          # a token in <= this many customer messages is "distinctive"
MIN_TOKEN_LEN = 4
KS = (5, 10)
csv.field_size_limit(10**9)


def parse_ts(s: str) -> float:
    return dt.datetime.strptime(s, "%a %b %d %H:%M:%S %z %Y").timestamp()


def day_of(ts: float) -> str:
    return dt.datetime.fromtimestamp(ts, dt.timezone.utc).strftime("%Y-%m-%d")


def load(path: Path = DATA):
    """-> (inbound: author -> [customer messages], reply_of: customer tweet id -> company reply)."""
    inbound = collections.defaultdict(list)
    reply_of = {}
    with open(path, encoding="utf-8", newline="") as f:
        for r in csv.DictReader(f):
            ts = parse_ts(r["created_at"])
            if r["inbound"] == "True":
                inbound[r["author_id"]].append({"tid": r["tweet_id"], "ts": ts, "text": r["text"]})
            else:
                parent = (r["in_response_to_tweet_id"] or "").split(",")[0].strip()
                if parent and parent not in reply_of:
                    reply_of[parent] = {"tid": r["tweet_id"], "ts": ts, "text": r["text"],
                                        "author": r["author_id"]}
    return inbound, reply_of


def token_df(inbound) -> collections.Counter:
    df = collections.Counter()
    for items in inbound.values():
        for it in items:
            df.update(set(tokenize(it["text"])))
    return df


def rare_tokens(text: str, df) -> set:
    return {t for t in tokenize(text)
            if len(t) >= MIN_TOKEN_LEN and not t.isdigit() and df.get(t, 0) <= RARE_DF}


def build_probe(cid: str, items, df):
    items = sorted(items, key=lambda i: i["ts"])
    if len(items) < 3 or len({day_of(i["ts"]) for i in items}) < 2:
        return None
    q = items[-1]
    if day_of(q["ts"]) == day_of(items[0]["ts"]):
        return None
    qr = rare_tokens(q["text"], df)
    if not qr:
        return None
    hist = items[:-1]
    ev = [h for h in hist if qr & set(tokenize(h["text"]))]
    if not ev:
        return None
    return {"cid": cid, "query": q["text"], "query_tid": q["tid"], "query_ts": q["ts"],
            "evidence": [h["tid"] for h in ev], "evidence_text": [h["text"] for h in ev],
            "rare": sorted(qr)[:6]}


def mask_query(text: str, df) -> str:
    """The 'masked' track: drop every word that carries a distinctive token. The customer
    no longer repeats the exact keyword ("cracked screen" -> "any update on it?"), so retrieval
    has to rely on the rest of the message and on the surrounding history."""
    return " ".join(w for w in text.split() if not rare_tokens(w, df))


def build_probes(inbound, df):
    out = []
    for cid, items in inbound.items():
        p = build_probe(cid, items, df)
        if p:
            out.append(p)
    out.sort(key=lambda p: p["cid"])
    return out


def history_episodes(probe, inbound, reply_of):
    """Everything the customer and the brand said before the query, in time order."""
    out = []
    for h in inbound[probe["cid"]]:
        if h["ts"] >= probe["query_ts"] or h["tid"] == probe["query_tid"]:
            continue
        out.append({"text": h["text"], "role": "user", "tid": h["tid"], "parent": None, "ts": h["ts"]})
        rp = reply_of.get(h["tid"])
        if rp and rp["ts"] < probe["query_ts"]:
            out.append({"text": rp["text"], "role": "support", "tid": rp["tid"],
                        "parent": h["tid"], "ts": rp["ts"]})
    out.sort(key=lambda e: e["ts"])
    return out


def brand_episodes(brand, inbound, reply_of):
    """All messages in one brand's inbox: each customer message the brand answered, plus the
    answer. Episodes carry their customer's id ``cid`` for tag isolation."""
    out = []
    for cid, items in inbound.items():
        for h in items:
            rp = reply_of.get(h["tid"])
            if not rp or rp["author"] != brand:
                continue
            out.append({"text": h["text"], "role": "user", "tid": h["tid"], "parent": None,
                        "ts": h["ts"], "cid": cid})
            out.append({"text": rp["text"], "role": "support", "tid": rp["tid"], "parent": h["tid"],
                        "ts": rp["ts"], "cid": cid})
    out.sort(key=lambda e: e["ts"])
    return out


def ingest(mem: AgentMemory, clock: dict, episodes) -> None:
    """Replay history one calendar day per session, using the tweets' own timestamps."""
    day = None
    for e in episodes:
        d = day_of(e["ts"])
        if d != day:
            if day is not None:
                mem.end_session()
            mem.start_session(f"{mem.namespace}-{d}")
            day = d
        clock["t"] = e["ts"]
        mem.add(e["text"], role=e["role"], metadata={"tid": e["tid"], "parent": e["parent"]},
                tags=[e["cid"]] if e.get("cid") else None)
    if day is not None:
        mem.end_session()


def hit_tuples(mem: AgentMemory, hits):
    """(tid, parent) for each hit; semantic facts are traced back to their source tweets."""
    out, src = [], []
    for h in hits:
        if h.item.layer == Layer.SEMANTIC:
            src += h.item.metadata.get("sources", [])
        else:
            out.append((h.item.metadata.get("tid"), h.item.metadata.get("parent")))
    if src:
        for ep in mem.storage.get_many(list(dict.fromkeys(src))):
            out.append((ep.metadata.get("tid"), ep.metadata.get("parent")))
    return out


def covered(hits, evidence) -> set:
    ev = set(evidence)
    cov = set()
    for tid, parent in hits:
        for x in (tid, parent):
            if x in ev:
                cov.add(x)
    return cov


def paired_ci(a, b, rng, n_boot=2000):
    d = [x - y for x, y in zip(a, b)]
    n = len(d)
    boots = sorted(100 * sum(d[i] for i in [rng.randrange(n) for _ in range(n)]) / n for _ in range(n_boot))
    return 100 * sum(d) / n, boots[int(0.025 * n_boot)], boots[int(0.975 * n_boot)]


def make_embedder(name):
    if name == "hashing":
        from agentmem import HashingEmbedder
        return HashingEmbedder()
    return L.make_embedder(name)


# --------------------------------------------------------------- quality run
def run_quality(probes, inbound, reply_of, embedder, reranker):
    """Per probe: the hit list of each system (tids, most relevant first)."""
    per = collections.defaultdict(list)
    for p in probes:
        eps = history_episodes(p, inbound, reply_of)
        if not eps:
            continue
        clock = {"t": 0.0}
        mem = AgentMemory(storage=InMemoryStorage(), namespace=p["cid"], embedder=embedder,
                          clock=lambda: clock["t"], auto_consolidate=False)
        ingest(mem, clock, eps)
        clock["t"] = p["query_ts"]
        docs = [e["text"] for e in eps]
        per["recency window"].append([(e["tid"], e["parent"]) for e in reversed(eps)][:max(KS)])
        per["BM25"].append([(eps[i]["tid"], eps[i]["parent"]) for i in L.BM25(docs).top(p["query"], max(KS))])
        mem.reranker = None
        per["agentmem hybrid"].append(hit_tuples(mem, mem.recall(p["query"], k=max(KS), reinforce=False)))
        if reranker is not None:
            mem.reranker = reranker
            per["agentmem hybrid + reranker"].append(
                hit_tuples(mem, mem.recall(p["query"], k=max(KS), reinforce=False)))
        mem.reranker = None
        per["_ids"].append(p["cid"])
    return per


def score_quality(per, probes, rng):
    by_cid = {p["cid"]: p for p in probes}
    ids = per.pop("_ids")
    probes = [by_cid[c] for c in ids]
    summary, vs = {}, {}
    for name, hits in per.items():
        summary[name] = {}
        for k in KS:
            rec, any_ = [], []
            for p, h in zip(probes, hits):
                c = covered(h[:k], p["evidence"])
                rec.append(len(c) / len(p["evidence"]))
                any_.append(1.0 if c else 0.0)
            summary[name][k] = {"evidence_recall": 100 * statistics.mean(rec),
                                "any": 100 * statistics.mean(any_), "_rec": rec, "_any": any_}
    base = summary["BM25"]
    for name in summary:
        if name in ("BM25", "recency window"):
            continue
        vs[name] = {}
        for k in KS:
            d, lo, hi = paired_ci(summary[name][k]["_rec"], base[k]["_rec"], rng)
            vs[name][k] = {"diff": d, "ci": [lo, hi], "significant": lo > 0 or hi < 0}
    for name in summary:
        for k in KS:
            summary[name][k].pop("_rec"), summary[name][k].pop("_any")
    return summary, vs, len(probes)


# --------------------------------------------------------------- end to end
def run_end_to_end(probes, inbound, reply_of, embedder, reranker, db_path: Path):
    """The full system on real data: ingest through sessions with consolidation into SQLite,
    close the database, reopen it in a new object, then answer every probe and build the prompt
    context. Also checks the reopened answers match the ones from before the close."""
    st = SQLiteStorage(db_path)
    clock = {"t": 0.0}
    ingest_t0 = time.perf_counter()
    pre, stats = {}, collections.Counter()
    for p in probes:
        eps = history_episodes(p, inbound, reply_of)
        if not eps:
            continue
        mem = AgentMemory(st, namespace=p["cid"], embedder=embedder, clock=lambda: clock["t"],
                          auto_consolidate=True)
        ingest(mem, clock, eps)
        clock["t"] = p["query_ts"]
        mem.reranker = reranker
        pre[p["cid"]] = [h.item.id for h in mem.recall(p["query"], k=max(KS), reinforce=False)]
        s = mem.stats()
        stats["customers"] += 1
        stats["episodes"] += s["episodic"]
        stats["facts"] += s["semantic"]
        stats["sessions"] += s["sessions"]
        stats["history_messages"] += len(eps)
    ingest_s = time.perf_counter() - ingest_t0
    st.close()

    # close + reopen: a new storage object, as in a new process
    st = SQLiteStorage(db_path)
    rec, any_, ctx_hit, lat, match = [], [], [], [], 0
    ctx_chars = []
    for p in probes:
        if p["cid"] not in pre:
            continue
        clock["t"] = p["query_ts"]
        mem = AgentMemory(st, namespace=p["cid"], embedder=embedder, clock=lambda: clock["t"],
                          reranker=reranker, auto_consolidate=False)
        t = time.perf_counter()
        hits = mem.recall(p["query"], k=max(KS), reinforce=False)
        lat.append((time.perf_counter() - t) * 1000)
        match += [h.item.id for h in hits] == pre[p["cid"]]
        tuples = hit_tuples(mem, hits)
        c = covered(tuples[:max(KS)], p["evidence"])
        rec.append(len(c) / len(p["evidence"]))
        any_.append(1.0 if c else 0.0)
        ctx = mem.build_context(p["query"])
        ctx_chars.append(len(ctx))
        ctx_hit.append(1.0 if any(t_[:60] in ctx for t_ in p["evidence_text"]) else 0.0)
    st.close()
    n = len(rec)
    lat.sort()
    return {
        "customers": n, "ingest_seconds": round(ingest_s, 1),
        "history_messages": stats["history_messages"], "episodes_kept": stats["episodes"],
        "facts_learned": stats["facts"], "sessions": stats["sessions"],
        "evidence_recall_at_10": 100 * statistics.mean(rec),
        "any_at_10": 100 * statistics.mean(any_),
        "evidence_in_prompt_context": 100 * statistics.mean(ctx_hit),
        "avg_context_chars": round(statistics.mean(ctx_chars)),
        "reopen_identical_results": 100 * match / n,
        "query_ms_p50": round(lat[n // 2], 1), "query_ms_p95": round(lat[int(0.95 * (n - 1))], 1),
        "db_mb": round(db_path.stat().st_size / 1e6, 1),
    }


# --------------------------------------------------------------- main
def run_brand(brand, probes_all, inbound, reply_of, embedder, reranker, n_customers, seed, masked_mode, df):
    """One shared memory for a whole brand inbox; each customer's last message is the query and
    is removed from the store. Compares tag-isolated recall (one customer's history) against
    untagged recall (everyone's history), and reports how often untagged hits belong to others."""
    eps = brand_episodes(brand, inbound, reply_of)
    brand_cids = {e["cid"] for e in eps}
    probes = [p for p in probes_all if p["cid"] in brand_cids]
    if masked_mode:
        probes = [dict(p, query=mask_query(p["query"], df)) for p in probes]
        probes = [p for p in probes if tokenize(p["query"])]
    rng = random.Random(seed)
    probes = rng.sample(probes, min(n_customers, len(probes)))
    query_tids = {p["query_tid"] for p in probes}
    store = [e for e in eps if e["tid"] not in query_tids]
    end_ts = max(e["ts"] for e in eps)
    clock = {"t": end_ts}
    # SQLite, not an in-memory dict: 300k+ embeddings as Python lists exhaust RAM
    # (float32 BLOBs on disk keep the footprint to the index matrix alone).
    with tempfile.TemporaryDirectory() as d:
        st = SQLiteStorage(Path(d) / "brand.db")
        mem = AgentMemory(storage=st, namespace=brand, embedder=embedder,
                          clock=lambda: clock["t"], auto_consolidate=False)
        t0 = time.perf_counter()
        ingest(mem, clock, store)
        ingest_s = time.perf_counter() - t0
        out = _brand_queries(mem, brand, probes, store, end_ts, ingest_s, masked_mode, eps, reranker)
        st.close()
    return out


def _brand_queries(mem, brand, probes, store, end_ts, ingest_s, masked_mode, eps, reranker=None):
    clock = {"t": end_ts}
    mem.reranker = reranker
    mem.clock = lambda: clock["t"]
    print(f"  brand {brand}: {len(eps):,} inbox messages ({len(store):,} stored, {len(probes):,} queries), "
          f"ingest {ingest_s:.0f}s", file=sys.stderr, flush=True)

    rows = collections.defaultdict(list)
    lat = []
    for p in probes:
        ev = set(p["evidence"]) & {e["tid"] for e in store}
        if not ev:
            continue
        p = dict(p, evidence=sorted(ev))
        for label, tags in (("tag-isolated (this customer only)", [p["cid"]]), ("untagged (whole inbox)", None)):
            t = time.perf_counter()
            hits = mem.recall(p["query"], k=max(KS), reinforce=False, tags=tags)
            lat.append((time.perf_counter() - t) * 1000)
            tuples = hit_tuples(mem, hits)
            cov = covered(tuples[:max(KS)], p["evidence"])
            rows[label].append({"rec": {k: len(covered(tuples[:k], p["evidence"])) / len(p["evidence"]) for k in KS},
                                "any": {k: float(bool(covered(tuples[:k], p["evidence"]))) for k in KS}})
            if tags is None:
                owners = [h.item.tags[0] if h.item.tags else None for h in hits[:max(KS)]]
                rows["_foreign"].append(sum(1 for o in owners if o and o != p["cid"]) / max(1, len(owners)))
    out = {"brand": brand, "customers": len(rows["_foreign"]), "stored_messages": len(store),
           "ingest_seconds": round(ingest_s, 1), "masked": masked_mode, "systems": {}}
    for label in ("tag-isolated (this customer only)", "untagged (whole inbox)"):
        r = rows[label]
        out["systems"][label] = {f"evidence_recall@{k}": 100 * statistics.mean(x["rec"][k] for x in r) for k in KS}
        out["systems"][label].update({f"any@{k}": 100 * statistics.mean(x["any"][k] for x in r) for k in KS})
    out["foreign_hits_untagged_at10"] = 100 * statistics.mean(rows["_foreign"])
    lat.sort()
    out["query_ms_p50"] = round(lat[len(lat) // 2], 1)
    out["query_ms_p95"] = round(lat[int(0.95 * (len(lat) - 1))], 1)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(DATA))
    ap.add_argument("--customers", type=int, default=1500, help="sample of probe-eligible customers")
    ap.add_argument("--min-messages", type=int, default=0,
                    help="only customers with at least this many messages (long-history track)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--embedder", default="minilm", choices=["minilm", "hashing"])
    ap.add_argument("--rerank", action="store_true")
    ap.add_argument("--e2e", action="store_true", help="also run the end-to-end lifecycle test")
    ap.add_argument("--brand", help="run the shared-inbox test for one brand, e.g. AmazonHelp")
    ap.add_argument("--out", default=str(HERE / "results" / "twcs.json"))
    a = ap.parse_args(argv)

    t0 = time.perf_counter()
    inbound, reply_of = load(Path(a.data))
    df = token_df(inbound)
    probes_all = [p for p in build_probes(inbound, df) if len(inbound[p["cid"]]) >= a.min_messages]
    rng = random.Random(a.seed)
    probes = rng.sample(probes_all, min(a.customers, len(probes_all)))
    print(f"loaded {sum(map(len, inbound.values())):,} customer messages from {len(inbound):,} authors; "
          f"{len(probes_all):,} probe-eligible customers; using {len(probes):,} "
          f"({time.perf_counter() - t0:.0f}s)", file=sys.stderr)

    embedder = make_embedder(a.embedder)
    if a.embedder == "minilm":
        embedder = CachedEmbedder(embedder, HERE / "data" / "twcs" / "minilm_cache")
    reranker = None
    if a.rerank:
        from agentmem.rerank import CrossEncoderReranker
        reranker = CrossEncoderReranker()

    if a.brand:
        result = {"embedder": a.embedder, "rerank": a.rerank, "runs": {}}
        for masked in (False, True):
            key = "distinctive words masked" if masked else "full query"
            r = run_brand(a.brand, probes_all, inbound, reply_of, embedder, reranker, a.customers,
                          a.seed, masked, df)
            result["runs"][key] = r
            print(f"\n### {a.brand} - {key} ({r['customers']} customers, {r['stored_messages']:,} stored messages)\n")
            print("| system | evidence recall@5 | any@5 | evidence recall@10 | any@10 |\n|---|---|---|---|---|")
            for label, s in r["systems"].items():
                print(f"| {label} | {s['evidence_recall@5']:.1f}% | {s['any@5']:.1f}% | "
                      f"{s['evidence_recall@10']:.1f}% | {s['any@10']:.1f}% |")
            print(f"\nuntagged top-10 hits belonging to other customers: {r['foreign_hits_untagged_at10']:.1f}%")
            print(f"query latency in a {r['stored_messages']:,}-message namespace: p50 {r['query_ms_p50']} ms, "
                  f"p95 {r['query_ms_p95']} ms")
        if a.embedder == "minilm":
            embedder.save()
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(result, indent=2))
        print(f"\nwrote {a.out}")
        return result

    masked = []
    for p in probes:
        q = mask_query(p["query"], df)
        if tokenize(q):
            masked.append(dict(p, query=q))
    tracks = {"full query": probes, "distinctive words masked": masked}
    result = {"embedder": a.embedder, "rerank": a.rerank, "min_messages": a.min_messages,
              "probes_available": len(probes_all), "tracks": {}}
    summaries = {}
    for track, tp in tracks.items():
        per = run_quality(tp, inbound, reply_of, embedder, reranker)
        summary, vs, n = score_quality(per, tp, random.Random(1))
        result["tracks"][track] = {"customers": n, "summary": summary, "vs_bm25": vs}
        summaries[track] = (n, summary, vs)
    if a.e2e:
        with tempfile.TemporaryDirectory() as d:
            result["end_to_end"] = run_end_to_end(masked, inbound, reply_of, embedder, reranker,
                                                  Path(d) / "twcs.db")
    if a.embedder == "minilm":
        embedder.save()

    lines = [f"Customer Support on Twitter - {len(probes_all):,} eligible returning customers, "
             f"embedder={a.embedder}, reranker={a.rerank}\n"]
    for track, (n, summary, vs) in summaries.items():
        lines += [f"### track: {track} ({n:,} customers)\n",
                  "| system | evidence recall@5 | any@5 | evidence recall@10 | any@10 |", "|---|---|---|---|---|"]
        for name, s in summary.items():
            lines.append(f"| {name} | {s[5]['evidence_recall']:.1f}% | {s[5]['any']:.1f}% | "
                         f"{s[10]['evidence_recall']:.1f}% | {s[10]['any']:.1f}% |")
        lines += ["\npaired bootstrap vs BM25 (evidence recall, percentage points, 95% CI):\n",
                  "| system | @5 difference | @10 difference |", "|---|---|---|"]
        for name, v in vs.items():
            cell = lambda k: f"{v[k]['diff']:+.1f} [{v[k]['ci'][0]:+.1f}, {v[k]['ci'][1]:+.1f}]" + \
                ("" if v[k]["significant"] else " (n.s.)")
            lines.append(f"| {name} | {cell(5)} | {cell(10)} |")
        lines.append("")
    if a.e2e:
        e = result["end_to_end"]
        lines += ["end-to-end lifecycle on the distinctive-words-masked track (sessions, consolidation, SQLite, reopen):\n",
                  f"- {e['customers']} customers, {e['history_messages']:,} history messages -> "
                  f"{e['episodes_kept']:,} episodes kept, {e['facts_learned']:,} facts, {e['sessions']:,} sessions",
                  f"- evidence recall@10 {e['evidence_recall_at_10']:.1f}%, any@10 {e['any_at_10']:.1f}%",
                  f"- evidence text present in the prompt context: {e['evidence_in_prompt_context']:.1f}% "
                  f"(average context {e['avg_context_chars']} chars)",
                  f"- identical results after closing and reopening the database: {e['reopen_identical_results']:.1f}%",
                  f"- query latency p50 {e['query_ms_p50']} ms, p95 {e['query_ms_p95']} ms; "
                  f"database {e['db_mb']} MB; ingest {e['ingest_seconds']}s"]
    import resource
    peak_gb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6  # Linux: kilobytes
    result["peak_harness_rss_gb"] = round(peak_gb, 2)
    lines.append(f"\npeak memory of the harness process (corpus held in Python): {peak_gb:.1f} GB")
    text = "\n".join(lines)
    print(text)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    out.with_suffix(".md").write_text(text + "\n")
    print(f"\nwrote {out} ({time.perf_counter() - t0:.0f}s total)")
    return result


if __name__ == "__main__":
    main()
