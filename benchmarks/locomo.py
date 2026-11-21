"""LoCoMo long-term conversational memory benchmark - retrieval evaluation.

LoCoMo (Maharana et al., ACL 2024, https://github.com/snap-research/locomo):
10 very long two-person conversations (~27 sessions each, spanning months) with
~2,000 questions annotated with the exact dialogue turns ("evidence") that
answer them.

We replay every conversation into memory session by session (with its real
dates via an injected clock), then for each question measure

    evidence recall@k = |retrieved turns ∩ evidence turns| / |evidence turns|

No LLM is needed, so it runs for free. (End-to-end QA accuracy additionally
needs an LLM to read the retrieved context and answer; see --help.)

    python benchmarks/locomo.py                 # downloads the dataset (~2.8 MB)
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import statistics
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

from agentmem import AgentMemory, InMemoryStorage, Layer, RetrievalWeights
from agentmem.embeddings import tokenize

URL = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
HERE = Path(__file__).resolve().parent
# category ids as used by the LoCoMo evaluation code
CATEGORIES = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop", 5: "adversarial"}


def load(path: Path):
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print(f"downloading {URL} ...", file=sys.stderr)
        urllib.request.urlretrieve(URL, path)
    return json.loads(path.read_text())


def parse_date(s: str, fallback: float) -> float:
    for fmt in ("%I:%M %p on %d %B, %Y", "%I:%M %p on %d %b, %Y"):
        try:
            return time.mktime(time.strptime(s.strip(), fmt))
        except (ValueError, AttributeError):
            pass
    return fallback


def sessions(conv):
    n = 1
    while f"session_{n}" in conv:
        yield n, conv.get(f"session_{n}_date_time", ""), conv[f"session_{n}"]
        n += 1


def turn_text(t) -> str:
    text = f"{t['speaker']}: {t['text']}"
    if t.get("blip_caption"):
        text += f" [shares a photo: {t['blip_caption']}]"
    return text


# ------------------------------------------------------------------ baselines
class BM25:
    def __init__(self, docs, k1=1.5, b=0.75):
        self.toks = [tokenize(d) for d in docs]
        self.k1, self.b = k1, b
        self.avg = statistics.mean(len(t) for t in self.toks) or 1.0
        df = Counter(w for t in self.toks for w in set(t))
        n = len(docs)
        self.idf = {w: math.log(1 + (n - c + 0.5) / (c + 0.5)) for w, c in df.items()}
        self.tf = [Counter(t) for t in self.toks]

    def top(self, query, k):
        q = tokenize(query)
        scores = []
        for i, (tf, toks) in enumerate(zip(self.tf, self.toks)):
            s = 0.0
            for w in q:
                f = tf.get(w)
                if f:
                    s += self.idf[w] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * len(toks) / self.avg))
            scores.append((s, i))
        scores.sort(reverse=True)
        return [i for _, i in scores[:k]]


# ------------------------------------------------------------------ systems
def ingest_agentmem(sample, weights=None, discard_below=0.1):
    conv = sample["conversation"]
    clock = {"t": 0.0}
    mem = AgentMemory(storage=InMemoryStorage(), namespace=sample["sample_id"], clock=lambda: clock["t"],
                      weights=weights, discard_below=discard_below)
    last = 0.0
    for n, date, turns in sessions(conv):
        start = parse_date(date, last + 86400)
        clock["t"] = start
        with mem.session(f"{sample['sample_id']}-s{n}"):
            for t in turns:
                mem.add(turn_text(t), role=t["speaker"], metadata={"dia_id": t["dia_id"]})
                clock["t"] += 30
        last = clock["t"]
    clock["t"] = last + 86400  # questions are asked the day after the last session
    return mem


def agentmem_retriever(mem):
    ep_dia = {e.id: e.metadata.get("dia_id") for e in mem.storage.list(mem.namespace, Layer.EPISODIC)}

    def retrieve(question, k):
        out = []
        for h in mem.recall(question, k=k, reinforce=False):
            if h.item.layer == Layer.SEMANTIC:  # facts point back to their source turns
                out += [ep_dia.get(s) for s in h.item.metadata.get("sources", [])]
            else:
                out.append(h.item.metadata.get("dia_id"))
        return [d for d in dict.fromkeys(out) if d][:k]
    return retrieve


def all_turns(sample):
    return [t for _, _, turns in sessions(sample["conversation"]) for t in turns]


def run(data, ks, include_adversarial=False, seed=0, lexical=0.5):
    rng = random.Random(seed)
    systems = ["random", "recency", "bm25", "agentmem (relevance only)", "agentmem"]
    # results[system][k] -> list of (category, recall, hit)
    results = {s: {k: [] for k in ks} for s in systems}
    kept = {"turns": 0, "episodes": 0, "facts": 0}
    t0 = time.perf_counter()
    for sample in data:
        turns = all_turns(sample)
        ids = [t["dia_id"] for t in turns]
        bm25 = BM25([turn_text(t) for t in turns])
        full = ingest_agentmem(sample, weights=RetrievalWeights(lexical=lexical))
        rel_only = ingest_agentmem(sample, weights=RetrievalWeights(recency=0.0, importance=0.0, lexical=lexical))
        kept["turns"] += len(turns)
        kept["episodes"] += full.episodic.count()
        kept["facts"] += full.semantic.count()
        retrievers = {
            "random": lambda q, k: rng.sample(ids, min(k, len(ids))),
            "recency": lambda q, k: ids[-k:],
            "bm25": lambda q, k: [ids[i] for i in bm25.top(q, k)],
            "agentmem (relevance only)": agentmem_retriever(rel_only),
            "agentmem": agentmem_retriever(full),
        }
        for qa in sample["qa"]:
            cat = qa["category"]
            if cat == 5 and not include_adversarial:
                continue
            ev = set(qa.get("evidence") or [])
            # evidence ids occasionally carry formatting noise like "D8:6; D9:17"
            ev = {e.strip() for x in ev for e in re.split(r"[;,]", x) if e.strip()}
            if not ev:
                continue
            for name, fn in retrievers.items():
                got_all = fn(qa["question"], max(ks))
                for k in ks:
                    got = set(got_all[:k])
                    inter = len(got & ev)
                    results[name][k].append((cat, inter / len(ev), inter > 0))
    return results, kept, time.perf_counter() - t0


def report(results, kept, seconds, ks):
    lines = []
    n = len(next(iter(results.values()))[ks[0]])
    lines.append(f"LoCoMo retrieval - {n} questions, {kept['turns']} dialogue turns, 10 conversations")
    lines.append(f"agentmem kept {kept['episodes']} episodes and {kept['facts']} facts; ran in {seconds:.0f}s\n")
    hdr = "| system | " + " | ".join(f"recall@{k}" for k in ks) + " | " + " | ".join(f"hit@{k}" for k in ks) + " |"
    lines += [hdr, "|" + "---|" * (1 + 2 * len(ks))]
    summary = {}
    for name, byk in results.items():
        r = [100 * statistics.mean(x[1] for x in byk[k]) for k in ks]
        h = [100 * statistics.mean(x[2] for x in byk[k]) for k in ks]
        summary[name] = {"recall": dict(zip(ks, r)), "hit": dict(zip(ks, h))}
        lines.append(f"| {name} | " + " | ".join(f"{v:.1f}%" for v in r) + " | " + " | ".join(f"{v:.1f}%" for v in h) + " |")
    k = ks[1] if len(ks) > 1 else ks[0]
    lines.append(f"\nrecall@{k} by question type:\n")
    cats = sorted({c for c, _, _ in results["agentmem"][k]})
    lines.append("| system | " + " | ".join(f"{CATEGORIES.get(c, c)} (n={sum(1 for x in results['agentmem'][k] if x[0]==c)})" for c in cats) + " |")
    lines.append("|" + "---|" * (1 + len(cats)))
    by_cat = {}
    for name, byk in results.items():
        g = defaultdict(list)
        for c, r, _ in byk[k]:
            g[c].append(r)
        by_cat[name] = {CATEGORIES.get(c, c): 100 * statistics.mean(g[c]) for c in cats}
        lines.append(f"| {name} | " + " | ".join(f"{100*statistics.mean(g[c]):.1f}%" for c in cats) + " |")
    return "\n".join(lines), {"summary": summary, "by_category": by_cat, "kept": kept, "questions": n}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(HERE / "data" / "locomo10.json"))
    ap.add_argument("--k", type=int, nargs="+", default=[5, 10, 25])
    ap.add_argument("--include-adversarial", action="store_true")
    ap.add_argument("--limit", type=int, help="only the first N conversations")
    ap.add_argument("--lexical", type=float, default=0.8, help="BM25 share of relevance (hybrid search)")
    ap.add_argument("--out", default=str(HERE / "results" / "locomo.json"))
    a = ap.parse_args(argv)
    data = load(Path(a.data))[: a.limit]
    results, kept, secs = run(data, a.k, a.include_adversarial, lexical=a.lexical)
    text, js = report(results, kept, secs, a.k)
    print(text)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({**js, "k": a.k, "seconds": round(secs, 1)}, indent=2))
    Path(a.out).with_suffix(".md").write_text(text + "\n")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
