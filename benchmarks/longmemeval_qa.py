"""LongMemEval answer accuracy: does a reader LLM answer correctly from retrieved memory?

For each question the haystack is replayed into a fresh AgentMemory; each system retrieves
turns; the turns are grouped by session (with dates) into a prompt using LongMemEval's reader
prompt; an LLM answers; the answer is graded.

Systems
  oracle     only the evidence sessions: the ceiling for this reader (no retrieval error)
  recency    the last N turns of the history (a plain chat window)
  BM25       keyword retrieval
  agentmem   hybrid retrieval (+ cross-encoder reranker with --rerank)

Grading
  contains   normalised gold answer appears in the prediction (LLM-free, strict on wording)
  F1         token F1 against the gold answer
  judge      --judge: the same LLM grades CORRECT/WRONG (LongMemEval's official judge is GPT-4o;
             a small local judge is noisy, so read it as a relative number only)

    python benchmarks/longmemeval_qa.py --llm llamacpp:/path/model.gguf --per-type 10
    python benchmarks/longmemeval_qa.py --llm ollama:llama3.2 --judge
    python benchmarks/longmemeval_qa.py --llm claude:claude-sonnet-5-5 --judge
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import locomo as L  # noqa: E402
import locomo_qa as Q  # noqa: E402
import longmemeval as LME  # noqa: E402
from agentmem import HeuristicImportanceScorer, Layer, RetrievalWeights  # noqa: E402

HERE = Path(__file__).resolve().parent

READER_PROMPT = (
    "I will give you several history chats between you and a user. Please answer the question based on "
    "the relevant chat history. Answer the question step by step: first extract all the relevant "
    "information, and then reason over the information to get the answer. Finish with a line of the form "
    "\"Final answer: <short answer>\".")
DIRECT_PROMPT = (
    "I will give you several history chats between you and a user. Answer the question using only the "
    "chat history. Reply with exactly one line of the form \"Final answer: <short answer>\". If the history "
    "does not contain the answer, reply \"Final answer: unknown\".")
JUDGE_PROMPT = (
    "I will give you a question, a correct answer, and a response from a model. Please answer yes if the "
    "response contains the correct answer, and no otherwise. If the response is equivalent to the correct "
    "answer or contains all the intermediate steps to get the correct answer, answer yes. If the response "
    "only contains a subset of the information required by the answer, answer no. For time questions, do "
    "not penalise off-by-one errors in days/weeks/months. Reply with one word: yes or no.")


def make_llm(spec: str, max_tokens: int = 400):
    if spec.startswith("llamacpp:"):
        from llama_cpp import Llama
        llm = Llama(model_path=spec.split(":", 1)[1], n_ctx=8192, n_threads=4, verbose=False)

        def complete(system, messages):
            out = llm.create_chat_completion(
                messages=[{"role": "system", "content": system}] + messages, max_tokens=max_tokens, temperature=0.0)
            return out["choices"][0]["message"]["content"]
        return complete
    if spec.startswith("openai:"):
        from agentmem.integrations.openai import openai_complete
        return openai_complete(spec.split(":", 1)[1] or "gpt-4.1-mini", max_tokens=max_tokens)
    return Q.make_llm(spec)


def fmt_context(items, max_chars):
    """items: [(session_date_ts, session_date_str, role, content)] -> text grouped by session."""
    by_session = defaultdict(list)
    for ts, date, role, content, sid in items:
        by_session[(ts, date, sid)].append((role, content))
    parts = []
    for (ts, date, sid), turns in sorted(by_session.items()):
        body = "\n".join(f"{r}: {c[:max_chars]}" for r, c in turns)
        parts.append(f"### Session date: {date}\n{body}")
    return "\n\n".join(parts)


def final_answer(text: str) -> str:
    for line in reversed(text.strip().splitlines()):
        if line.lower().startswith("final answer"):
            return line.split(":", 1)[-1].strip()
    return text.strip().splitlines()[-1] if text.strip() else ""


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--llm", required=True, help="llamacpp:/path.gguf | ollama:MODEL | claude:MODEL")
    ap.add_argument("--judge", action="store_true")
    ap.add_argument("--per-type", type=int, default=10)
    ap.add_argument("--k", type=int, default=10, help="turns retrieved per system")
    ap.add_argument("--max-chars", type=int, default=700, help="truncate each retrieved turn")
    ap.add_argument("--rerank", action="store_true")
    ap.add_argument("--cot", action="store_true",
                    help="LongMemEval's step-by-step reader prompt (better for strong models, ~3x slower)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--data", default=str(LME.DATA))
    ap.add_argument("--out", default=str(HERE / "results" / "longmemeval_qa.json"))
    a = ap.parse_args(argv)

    data = [q for q in json.loads(Path(a.data).read_text()) if not q["question_id"].endswith("_abs")]
    rng = random.Random(a.seed)
    qs = []
    for t in LME.TYPES:
        pool = [q for q in data if q["question_type"] == t]
        qs += rng.sample(pool, min(a.per_type, len(pool)))
    llm = make_llm(a.llm, max_tokens=400 if a.cot else 60)
    reader_prompt = READER_PROMPT if a.cot else DIRECT_PROMPT
    emb = LME.CachedEmbedder(L.make_embedder("minilm"), LME.CACHE.with_name("minilm_cache"))
    reranker = None
    if a.rerank:
        from agentmem.rerank import CrossEncoderReranker
        reranker = CrossEncoderReranker()
    scorer = HeuristicImportanceScorer()
    res = defaultdict(lambda: defaultdict(list))
    bytype = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    examples = []
    t0 = time.perf_counter()
    for n, q in enumerate(qs, 1):
        mem, clock, turns = LME.build_memory(q, emb, scorer)
        order = {sid: i for i, sid in enumerate(q["haystack_session_ids"])}
        date_of = dict(zip(q["haystack_session_ids"], q["haystack_dates"]))
        sess = {sid: s for sid, s in zip(q["haystack_session_ids"], q["haystack_sessions"])}
        bm25 = L.BM25([tx for tx, _, _ in turns])

        def mk(sid, role, content):
            return (LME.parse_date(date_of[sid]), date_of[sid], role, content, sid)

        flat = [(sid, t["role"], t["content"]) for sid, s in zip(q["haystack_session_ids"], q["haystack_sessions"]) for t in s]
        sysctx = {}
        sysctx["oracle (evidence sessions)"] = [mk(sid, t["role"], t["content"])
                                                for sid in q["answer_session_ids"] for t in sess[sid]]
        sysctx["recency (chat window)"] = [mk(*f) for f in sorted(flat, key=lambda f: LME.parse_date(date_of[f[0]]))[-a.k:]]
        sysctx["BM25"] = [mk(*flat[i]) for i in bm25.top(q["question"], a.k)]
        w = RetrievalWeights(lexical=0.5)
        mem.episodic.weights = w
        mem.reranker = reranker
        hits = mem.recall(q["question"], k=a.k, layers=[Layer.EPISODIC], reinforce=False)
        name = "agentmem hybrid" + (" + reranker" if reranker else "")
        sysctx[name] = [mk(h.item.metadata["sid"], h.item.metadata["role"], h.item.content.split(": ", 1)[-1]) for h in hits]

        for sname, items in sysctx.items():
            ctx = fmt_context(items, a.max_chars)
            prompt = f"{ctx}\n\nCurrent Date: {q['question_date']}\nQuestion: {q['question']}\nAnswer:"
            raw = llm(reader_prompt, [{"role": "user", "content": prompt}])
            pred = final_answer(raw)
            gold = q["answer"]
            row = {"contains": float(Q.normalize(gold) in Q.normalize(raw)) if Q.normalize(gold) else 0.0,
                   "f1": Q.f1(pred, gold)}
            if a.judge:
                v = llm(JUDGE_PROMPT, [{"role": "user", "content":
                        f"Question: {q['question']}\nCorrect answer: {gold}\nModel response: {raw}\nAnswer yes or no:"}])
                row["judge"] = float(v.strip().lower().startswith("yes"))
            row["unknown"] = float("unknown" in pred.lower())
            for m, v in row.items():
                res[sname][m].append(v)
                bytype[sname][q["question_type"]][m].append(v)
            if len(examples) < 40 and sname.startswith("agentmem"):
                examples.append({"type": q["question_type"], "question": q["question"], "gold": gold,
                                 "pred": pred, "contains": row["contains"]})
        el = time.perf_counter() - t0
        print(f"  {n}/{len(qs)} ({el:.0f}s, ~{el / n * (len(qs) - n):.0f}s left)", file=sys.stderr, flush=True)

    metrics = ["contains", "f1"] + (["judge"] if a.judge else []) + ["unknown"]
    print(f"\nLongMemEval-S QA - {len(qs)} questions ({a.per_type}/type, abstention excluded), k={a.k}, llm={a.llm}\n")
    print("| system | " + " | ".join(metrics) + " |")
    print("|---|" + "---|" * len(metrics))
    summary = {s: {m: 100 * statistics.mean(v[m]) for m in metrics} for s, v in res.items()}
    for s, v in summary.items():
        print(f"| {s} | " + " | ".join(f"{v[m]:.1f}%" for m in metrics) + " |")
    primary = "judge" if a.judge else "contains"
    types = [t for t in LME.TYPES if any(t in bytype[s] for s in bytype)]
    print(f"\n{primary} by question type:\n")
    print("| system | " + " | ".join(types) + " |")
    print("|---|" + "---|" * len(types))
    for s in res:
        print(f"| {s} | " + " | ".join(f"{100 * statistics.mean(bytype[s][t][primary]):.1f}%" for t in types) + " |")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"llm": a.llm, "questions": len(qs), "k": a.k, "summary": summary,
                               "by_type": {s: {t: {m: 100 * statistics.mean(v) for m, v in d.items()}
                                               for t, d in tt.items()} for s, tt in bytype.items()},
                               "examples": examples, "seconds": round(time.perf_counter() - t0)}, indent=2))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
