"""LongMemEval retrieval benchmark (Wu et al., ICLR 2025).

500 questions, each with its own chat history ("haystack") of ~48 sessions / ~490
turns (the S variant), over six question types:

  single-session-user / -assistant / -preference   one session holds the answer
  multi-session                                    answer spread over 2-6 sessions
  knowledge-update                                 the user's situation changed; latest wins
  temporal-reasoning                               needs dates / ordering

Every turn of a question's haystack is written into a fresh AgentMemory (with its
real timestamp via an injected clock), the question is asked at ``question_date``,
and we measure how well the evidence is retrieved:

  session-any@K   some answer session appears among the sessions of the top-K turns
  session-all@K   *every* answer session appears (matters for multi-session)
  turn-any@K      some evidence turn (``has_answer``) is itself in the top-K turns

Abstention questions (``*_abs``) have no evidence and are skipped, as in the official
retrieval evaluation. No LLM is needed.

    python benchmarks/longmemeval.py --limit 30                 # quick smoke run
    python benchmarks/longmemeval.py --rerank                   # full run (~hour on 4 CPUs)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import locomo as L  # noqa: E402  (BM25 baseline, shared helpers)
from agentmem import AgentMemory, HeuristicImportanceScorer, InMemoryStorage, Layer, RetrievalWeights  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = HERE / "data" / "longmemeval" / "longmemeval_s_cleaned.json"
CACHE = HERE / "data" / "longmemeval" / "emb_cache"
KS = (5, 10, 30)
TYPES = ["single-session-user", "single-session-assistant", "single-session-preference",
         "multi-session", "knowledge-update", "temporal-reasoning"]


def parse_date(s: str) -> float:
    return time.mktime(time.strptime(s.strip(), "%Y/%m/%d (%a) %H:%M"))


class CachedEmbedder:
    """Memoises an embedder across questions (haystacks share many turns) and persists
    the cache so reruns of other retrieval settings skip embedding entirely."""

    def __init__(self, inner, cache_path: Path | None = None) -> None:
        import numpy as np
        self.np, self.inner, self.dim = np, inner, inner.dim
        self.cache: dict = {}
        self.cache_path = cache_path
        self.hits = self.misses = 0
        if cache_path and cache_path.with_suffix(".npy").exists():
            keys = json.loads(cache_path.with_suffix(".json").read_text())
            mat = np.load(cache_path.with_suffix(".npy"))
            self.cache = dict(zip(keys, mat))

    @staticmethod
    def key(text: str) -> str:
        return hashlib.blake2b(text.encode(), digest_size=12).hexdigest()

    def embed_many(self, texts):
        keys = [self.key(t) for t in texts]
        todo = [(k, t) for k, t in dict(zip(keys, texts)).items() if k not in self.cache]
        self.hits += len(texts) - len(todo)
        self.misses += len(todo)
        if todo:
            many = getattr(self.inner, "embed_many", None)
            vecs = many([t for _, t in todo]) if many else [self.inner.embed(t) for _, t in todo]
            for (k, _), v in zip(todo, vecs):
                self.cache[k] = self.np.asarray(v, dtype=self.np.float32)
        return [self.cache[k].tolist() for k in keys]

    def embed(self, text: str):
        return self.embed_many([text])[0]

    def save(self) -> None:
        if not self.cache_path or not self.cache:
            return
        self.cache_path.parent.mkdir(parents=True, exist_ok=True)
        keys = list(self.cache)
        self.np.save(self.cache_path.with_suffix(".npy"), self.np.stack([self.cache[k] for k in keys]))
        self.cache_path.with_suffix(".json").write_text(json.dumps(keys))


def build_memory(q, embedder, scorer):
    """Replay one question's haystack into a fresh memory, oldest session first."""
    clock = {"t": 0.0}
    mem = AgentMemory(storage=InMemoryStorage(), namespace=q["question_id"], embedder=embedder,
                      clock=lambda: clock["t"], auto_consolidate=False)
    order = sorted(range(len(q["haystack_sessions"])), key=lambda i: parse_date(q["haystack_dates"][i]))
    turns = []  # (text, session_id, has_answer)
    for i in order:
        sid, date, session = q["haystack_session_ids"][i], q["haystack_dates"][i], q["haystack_sessions"][i]
        texts = [f"{t['role']}: {t['content']}" for t in session]
        clock["t"] = parse_date(date)
        mem.remember_many(
            texts,
            importance=[scorer.score(t["content"], role=t["role"]) for t in session],
            metadata=[{"sid": sid, "ti": n, "role": t["role"], "has_answer": bool(t.get("has_answer"))}
                      for n, t in enumerate(session)])
        turns += [(tx, sid, bool(t.get("has_answer"))) for tx, t in zip(texts, session)]
        clock["t"] += 3600
    clock["t"] = parse_date(q["question_date"])
    return mem, clock, turns


def agentmem_system(mem, weights, reranker=None, depth=30, weight=0.7):
    def retrieve(query, k):
        mem.episodic.weights = weights
        mem.reranker, mem.rerank_depth, mem.rerank_weight = reranker, depth, weight
        hits = mem.recall(query, k=k, layers=[Layer.EPISODIC], reinforce=False)
        return [(h.item.metadata["sid"], h.item.metadata["has_answer"]) for h in hits]
    return retrieve


def configs(rerank):
    W = RetrievalWeights
    out = [
        ("agentmem dense only", W(lexical=0.0), None),
        ("agentmem hybrid (shipped)", W(lexical=0.5), None),
        ("agentmem hybrid, relevance only", W(lexical=0.5, recency=0.0, importance=0.0), None),
    ]
    if rerank:
        out.append(("agentmem hybrid + reranker", W(lexical=0.5), "rerank"))
    return out


def score(got, answer_sids, k):
    top = got[:k]
    sids = {s for s, _ in top}
    ans = set(answer_sids)
    return {"session_any": bool(sids & ans), "session_all": ans <= sids,
            "turn_any": any(h for _, h in top)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=str(DATA))
    ap.add_argument("--limit", type=int, help="questions per type (quick run)")
    ap.add_argument("--types", nargs="+", choices=TYPES)
    ap.add_argument("--rerank", action="store_true")
    ap.add_argument("--embedder", default="minilm", choices=["minilm", "hashing"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(HERE / "results" / "longmemeval.json"))
    a = ap.parse_args(argv)

    data = json.loads(Path(a.data).read_text())
    data = [q for q in data if not q["question_id"].endswith("_abs")]
    if a.types:
        data = [q for q in data if q["question_type"] in a.types]
    if a.limit:
        rng = random.Random(a.seed)
        keep = []
        for t in TYPES:
            pool = [q for q in data if q["question_type"] == t]
            keep += rng.sample(pool, min(a.limit, len(pool)))
        data = keep

    emb = L.make_embedder(a.embedder)
    cached = CachedEmbedder(emb, CACHE.with_name(f"{a.embedder}_cache")) if a.embedder == "minilm" else emb
    reranker = None
    if a.rerank:
        from agentmem.rerank import CrossEncoderReranker
        reranker = CrossEncoderReranker()
    scorer = HeuristicImportanceScorer()  # no novelty term: cheap, deterministic

    results = defaultdict(lambda: {k: [] for k in KS})  # system -> k -> [(type, metrics)]
    qids = []
    ingest_s = recall_s = 0.0
    t0 = time.perf_counter()
    for n, q in enumerate(data, 1):
        t = time.perf_counter()
        mem, clock, turns = build_memory(q, cached, scorer)
        ingest_s += time.perf_counter() - t
        sids = [s for _, s, _ in turns]
        bm25 = L.BM25([tx for tx, _, _ in turns])
        systems = {
            "recency (chat window)": lambda query, k: [(sids[i], turns[i][2]) for i in range(len(turns) - k, len(turns))],
            "BM25": lambda query, k: [(sids[i], turns[i][2]) for i in bm25.top(query, k)],
        }
        for label, w, kind in configs(a.rerank):
            systems[label] = agentmem_system(mem, w, reranker if kind == "rerank" else None)
        qids.append(q["question_id"])
        t = time.perf_counter()
        for name, fn in systems.items():
            got = fn(q["question"], max(KS))
            for k in KS:
                results[name][k].append((q["question_type"], score(got, q["answer_session_ids"], k)))
        recall_s += time.perf_counter() - t
        if n % 10 == 0 or n == len(data):
            el = time.perf_counter() - t0
            print(f"  {n}/{len(data)} questions  ({el:.0f}s elapsed, ~{el / n * (len(data) - n):.0f}s left; "
                  f"embed cache {cached.hits} hits / {cached.misses} misses)" if a.embedder == "minilm"
                  else f"  {n}/{len(data)} questions ({el:.0f}s)", file=sys.stderr, flush=True)
    if a.embedder == "minilm":
        cached.save()

    # ---------------------------------------------------------------- report
    def pct(rows, key):
        return 100 * statistics.mean(m[key] for _, m in rows) if rows else float("nan")

    lines = [f"LongMemEval-S retrieval - {len(data)} questions (abstention excluded), turn-level index, "
             f"embedder={a.embedder}\n",
             "| system | " + " | ".join(f"sess-any@{k} | sess-all@{k} | turn-any@{k}" for k in KS) + " |",
             "|---|" + "---|" * (3 * len(KS))]
    summary = {}
    for name, byk in results.items():
        summary[name] = {k: {m: pct(byk[k], m) for m in ("session_any", "session_all", "turn_any")} for k in KS}
        lines.append(f"| {name} | " + " | ".join(
            f"{summary[name][k]['session_any']:.1f}% | {summary[name][k]['session_all']:.1f}% | "
            f"{summary[name][k]['turn_any']:.1f}%" for k in KS) + " |")
    kk = 10
    types = [t for t in TYPES if any(ty == t for ty, _ in next(iter(results.values()))[kk])]
    for metric, title in (("session_any", "session-any"), ("session_all", "session-all")):
        lines += [f"\n{title}@{kk} by question type:\n",
                  "| system | " + " | ".join(f"{t} (n={sum(1 for ty, _ in next(iter(results.values()))[kk] if ty == t)})"
                                            for t in types) + " |", "|---|" + "---|" * len(types)]
        for name, byk in results.items():
            lines.append(f"| {name} | " + " | ".join(
                f"{pct([r for r in byk[kk] if r[0] == t], metric):.1f}%" for t in types) + " |")
    # paired bootstrap over questions: is agentmem's gain over BM25 larger than noise?
    rng = random.Random(1)
    n = len(qids)
    boots = [[rng.randrange(n) for _ in range(n)] for _ in range(2000)]
    lines += ["\npaired bootstrap vs BM25 (95% CI of the difference, percentage points; 2,000 resamples):\n",
              "| system | metric | difference | 95% CI | significant |", "|---|---|---|---|---|"]
    significance = {}
    for name in results:
        if name in ("BM25", "recency (chat window)"):
            continue
        for k, metric in ((10, "session_any"), (10, "session_all"), (10, "turn_any"), (5, "session_all")):
            a_ = [float(m[metric]) for _, m in results[name][k]]
            b_ = [float(m[metric]) for _, m in results["BM25"][k]]
            d = [x - y for x, y in zip(a_, b_)]
            diffs = sorted(100 * sum(d[i] for i in idx) / n for idx in boots)
            lo, hi = diffs[int(0.025 * len(diffs))], diffs[int(0.975 * len(diffs))]
            point = 100 * sum(d) / n
            sig = lo > 0 or hi < 0
            significance[f"{name}|{metric}@{k}"] = {"diff": point, "ci": [lo, hi], "significant": sig}
            lines.append(f"| {name} | {metric.replace('_', '-')}@{k} | {point:+.1f} | [{lo:+.1f}, {hi:+.1f}] | "
                         f"{'yes' if sig else 'no'} |")
    lines.append(f"\ningest {ingest_s:.0f}s, retrieval {recall_s:.0f}s for {len(data)} questions "
                 f"({1000 * recall_s / len(data) / len(results):.0f} ms per system per question)")
    text = "\n".join(lines)
    print(text)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"questions": len(data), "embedder": a.embedder, "summary": summary,
                               "by_type": {name: {t: {m: pct([r for r in byk[kk] if r[0] == t], m)
                                                      for m in ("session_any", "session_all", "turn_any")}
                                                  for t in types} for name, byk in results.items()},
                               "vs_bm25": significance,
                               "per_question": [{"id": qid, "type": results["BM25"][10][i][0],
                                                 **{name: {str(k): results[name][k][i][1] for k in KS}
                                                    for name in results}} for i, qid in enumerate(qids)],
                               "seconds": round(time.perf_counter() - t0, 1)}, indent=2))
    out.with_suffix(".md").write_text(text + "\n")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
