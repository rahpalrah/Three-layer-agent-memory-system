"""LoCoMo question answering: does the agent answer correctly from memory?

For each question: retrieve k memories (with their session dates), ask an LLM to
answer from them, score against the gold answer.

Metrics
  F1          token-level F1 after SQuAD/LoCoMo answer normalisation (official metric)
  J           LLM-as-judge correctness, as in the Mem0 paper (--judge)
  coverage    share of gold-answer tokens present in the retrieved context. Needs no
              LLM: an upper bound for any reader that only sees this context.

    python benchmarks/locomo_qa.py --llm none                       # free, model-free coverage
    python benchmarks/locomo_qa.py --llm ollama:llama3.2 --judge    # free, local LLM
    python benchmarks/locomo_qa.py --llm claude:claude-sonnet-5-5   # needs ANTHROPIC_API_KEY
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
import string
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import locomo as L  # noqa: E402
from agentmem import Layer, RetrievalWeights  # noqa: E402

ANSWER_PROMPT = (
    "You answer questions about a long conversation between two people, using ONLY the memories "
    "below. Each memory is prefixed with the date of the conversation it comes from; resolve relative "
    "dates (\"yesterday\", \"last week\") against that date. Answer in as few words as possible "
    "(a name, a date, a short phrase). If the memories do not contain the answer, reply \"unknown\".")
JUDGE_PROMPT = (
    "You grade answers. Given a question, a gold answer and a generated answer, reply CORRECT if the "
    "generated answer conveys the same information as the gold answer (be generous with wording, "
    "date formats and extra detail), otherwise WRONG. Reply with one word.")

_ARTICLES = re.compile(r"\b(a|an|the|and)\b")


def normalize(s) -> str:
    s = str(s).lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    return " ".join(_ARTICLES.sub(" ", s).split())


def f1(pred, gold) -> float:
    p, g = normalize(pred).split(), normalize(gold).split()
    if not p or not g:
        return float(p == g)
    common = Counter(p) & Counter(g)
    same = sum(common.values())
    if same == 0:
        return 0.0
    prec, rec = same / len(p), same / len(g)
    return 2 * prec * rec / (prec + rec)


def coverage(context: str, gold) -> float:
    g = set(normalize(gold).split())
    c = set(normalize(context).split())
    return len(g & c) / len(g) if g else 0.0


def make_llm(spec: str):
    if spec == "none":
        return None
    provider, _, model = spec.partition(":")
    if provider == "ollama":
        from agentmem.integrations.ollama import ollama_complete
        return ollama_complete(model or "llama3.2", options={"temperature": 0})
    if provider == "claude":
        from anthropic import Anthropic
        from agentmem.integrations.anthropic import claude_complete
        return claude_complete(Anthropic(), model or "claude-sonnet-5-5", max_tokens=64)
    if provider == "openai":
        from agentmem.integrations.openai import openai_complete
        return openai_complete(model or "gpt-4.1-mini", max_tokens=64)
    raise ValueError(spec)


def systems(mem, reranker, turns_by_id, bm25, ids):
    def fmt(dia_ids):
        return [f"[{turns_by_id[d]['date']}] {L.turn_text(turns_by_id[d]['turn'])}" for d in dia_ids if d in turns_by_id]

    def agentmem(rr):
        retr = L.agentmem_retriever(mem, RetrievalWeights(lexical=0.5), 0, rr, 30, 0.7)
        return lambda q, k: fmt(retr(q, k))
    out = {
        "recency (chat window)": lambda q, k: fmt(ids[-k:]),
        "BM25": lambda q, k: fmt([ids[i] for i in bm25.top(q, k)]),
        "agentmem (hybrid)": agentmem(None),
    }
    if reranker is not None:
        out["agentmem (+ cross-encoder rerank)"] = agentmem(reranker)
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--llm", default="none",
                    help="none | extractive (no model) | ollama:MODEL | claude:MODEL | openai:MODEL")
    ap.add_argument("--embedder", default="self", choices=["self", "hashing", "minilm"],
                    help="self = learned from each conversation, no downloads (default)")
    ap.add_argument("--judge", action="store_true", help="also grade with LLM-as-judge (J score)")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--conversations", default="2-10")
    ap.add_argument("--limit-questions", type=int, help="per conversation (for quick LLM runs)")
    ap.add_argument("--rerank", action="store_true", help="add the cross-encoder reranker (needs model files)")
    ap.add_argument("--out", default=str(L.HERE / "results" / "locomo_qa.json"))
    a = ap.parse_args(argv)
    extractive = a.llm == "extractive"
    llm = None if extractive else make_llm(a.llm)
    reader = None
    if extractive:
        from agentmem.reader import ExtractiveReader
        reader = ExtractiveReader()
    data = [L.load(L.HERE / "data" / "locomo10.json")[i - 1] for i in L.parse_range(a.conversations)]
    fixed_emb = None if a.embedder == "self" else L.make_embedder(a.embedder)
    reranker = None
    if a.rerank:
        from agentmem.rerank import CrossEncoderReranker
        reranker = CrossEncoderReranker()
    res = defaultdict(lambda: defaultdict(list))  # system -> metric -> values
    by_cat = defaultdict(lambda: defaultdict(list))
    examples = []
    t0 = time.perf_counter()
    for sample in data:
        turns_by_id = {}
        for n, date, turns in L.sessions(sample["conversation"]):
            for t in turns:
                turns_by_id[t["dia_id"]] = {"date": date, "turn": t}
        ids = list(turns_by_id)
        bm25 = L.BM25([L.turn_text(v["turn"]) for v in turns_by_id.values()])
        if a.embedder == "self":
            from agentmem.selfembed import SelfEmbedder
            emb = SelfEmbedder(dim=256).fit(L.turn_text(v["turn"]) for v in turns_by_id.values())
        else:
            emb = fixed_emb
        mem = L.ingest_agentmem(sample, emb)
        sys_fns = systems(mem, reranker, turns_by_id, bm25, ids)
        qas = [q for q in sample["qa"] if q["category"] != 5 and "answer" in q][: a.limit_questions]
        for q in qas:
            gold = q["answer"]
            for name, fn in sys_fns.items():
                ctx = fn(q["question"], a.k)
                text = "\n".join(ctx)
                cov = coverage(text, gold)
                res[name]["coverage"].append(cov)
                by_cat[name][L.CATEGORIES[q["category"]]].append(cov)
                if reader is not None:
                    pred = reader.answer(q["question"], ctx)
                    res[name]["f1"].append(f1(pred, gold))
                    if len(examples) < 30:
                        examples.append({"system": name, "question": q["question"], "gold": gold, "pred": pred})
                elif llm is not None:
                    pred = llm(ANSWER_PROMPT, [{"role": "user", "content": f"Memories:\n{text}\n\nQuestion: {q['question']}"}]).strip()
                    res[name]["f1"].append(f1(pred, gold))
                    if a.judge:
                        verdict = llm(JUDGE_PROMPT, [{"role": "user", "content":
                                      f"Question: {q['question']}\nGold answer: {gold}\nGenerated answer: {pred}"}])
                        res[name]["j"].append(float("CORRECT" in verdict.upper()))
                    if len(examples) < 30:
                        examples.append({"system": name, "question": q["question"], "gold": gold, "pred": pred})
        print(f"  {sample['sample_id']} done ({time.perf_counter() - t0:.0f}s)", file=sys.stderr)
    n = len(next(iter(res.values()))["coverage"])
    print(f"LoCoMo QA - {n} questions (conversations {a.conversations}), k={a.k}, llm={a.llm}\n")
    metrics = ["coverage"] + (["f1"] if (llm or reader) else []) + (["j"] if a.judge else [])
    print("| system | " + " | ".join({"coverage": "answer coverage", "f1": "F1", "j": "J (judge)"}[m] for m in metrics) + " |")
    print("|---|" + "---|" * len(metrics))
    summary = {}
    for name, ms in res.items():
        summary[name] = {m: 100 * statistics.mean(ms[m]) for m in metrics}
        print(f"| {name} | " + " | ".join(f"{summary[name][m]:.1f}%" for m in metrics) + " |")
    cats = sorted({c for d in by_cat.values() for c in d})
    print("\nanswer coverage by question type:\n")
    print("| system | " + " | ".join(cats) + " |")
    print("|---|" + "---|" * len(cats))
    for name, d in by_cat.items():
        print(f"| {name} | " + " | ".join(f"{100 * statistics.mean(d[c]):.1f}%" for c in cats) + " |")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps({"llm": a.llm, "k": a.k, "questions": n, "summary": summary,
                                       "by_category": {k: {c: 100 * statistics.mean(v) for c, v in d.items()} for k, d in by_cat.items()},
                                       "examples": examples}, indent=2))


if __name__ == "__main__":
    main()
