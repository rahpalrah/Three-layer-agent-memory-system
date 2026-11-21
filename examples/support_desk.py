"""Business use case: repeat-customer support for an online coffee shop.

20 synthetic customers, 4 conversations each over 8 simulated weeks
(introductions, preferences, an allergy, a damaged order, a change of mind),
then a week-8 visit asking questions only memory can answer.

    python examples/support_desk.py                 # free: no LLM needed
    python examples/support_desk.py --ollama llama3.2   # also let a local model answer

It compares three bots:
  stateless   - forgets everything between conversations (most chatbots)
  window-20   - keeps the last 20 messages of history (the usual "chat history" approach)
  agentmem    - three-layer memory
and writes a scorecard to benchmarks/results/support_desk.json.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import tempfile
import time
from pathlib import Path

from agentmem import AgentMemory

DAY = 86400
T0 = time.mktime((2026, 1, 5, 10, 0, 0, 0, 0, -1))

NAMES = ["Priya", "Marco", "Aisha", "Kenji", "Sofia", "Liam", "Fatima", "Noah", "Elena", "Ravi",
         "Chloe", "Omar", "Hannah", "Diego", "Mei", "Lucas", "Zara", "Ethan", "Ines", "Tomas"]
CITIES = ["Pune", "Milan", "Lagos", "Osaka", "Lisbon", "Dublin", "Cairo", "Toronto", "Athens", "Chennai",
          "Lyon", "Amman", "Hamburg", "Bogota", "Taipei", "Porto", "Nairobi", "Denver", "Seville", "Prague"]
GRINDS = ["whole beans", "espresso grind", "French press grind", "pour-over grind"]
ROASTS = ["light", "medium", "dark"]
ALLERGIES = ["hazelnut", "dairy", "almond", "soy", None]
FILLER_USER = ["ok thanks", "What are your opening hours?", "Do you ship on weekends?", "cool",
               "How long does delivery usually take?", "thanks!", "Is the website down today?"]
FILLER_BOT = ["Happy to help!", "We ship Monday to Saturday.", "Delivery takes 2-4 business days.",
              "Anything else I can do for you?", "Noted, thanks!"]


def make_customers(seed: int = 7):
    rng = random.Random(seed)
    out = []
    for i, name in enumerate(NAMES):
        roast = rng.choice(ROASTS)
        changes = i % 2 == 0
        out.append({
            "id": f"cust-{i:02d}", "name": name, "city": CITIES[i], "grind": rng.choice(GRINDS),
            "roast": roast, "new_roast": rng.choice([r for r in ROASTS if r != roast]) if changes else None,
            "allergy": ALLERGIES[i % len(ALLERGIES)], "order": 4000 + rng.randint(100, 999),
        })
    return out


def conversations(c, rng):
    """Yield (day_offset, [(role, text), ...]) for one customer's history."""
    def filler(n):
        turns = []
        for _ in range(n):
            turns += [("user", rng.choice(FILLER_USER)), ("assistant", rng.choice(FILLER_BOT))]
        return turns

    s1 = [("user", f"Hi, my name is {c['name']} and I live in {c['city']}."), ("assistant", "Welcome!"),
          ("user", f"I prefer {c['grind']} for my coffee."), ("assistant", "Noted, thanks!")] + filler(3)
    s2 = filler(2) + [("user", f"I really like {c['roast']} roast coffee."), ("assistant", "Great choice.")]
    if c["allergy"]:
        s2 += [("user", f"I'm allergic to {c['allergy']}, so please no flavoured blends with it."),
               ("assistant", "Understood, we'll keep that in mind.")]
    s2 += filler(3)
    s3 = filler(1) + [
        ("user", f"My order #{c['order']} arrived damaged, the bag was torn."),
        ("assistant", "I'm sorry! We'll send a replacement."),
        ("user", f"Remember that you promised me a free replacement for order #{c['order']}."),
        ("assistant", "Yes, the replacement is on its way.")]
    if c["new_roast"]:
        s3 += [("user", f"Also, I changed my mind: I prefer {c['new_roast']} roast coffee now."),
               ("assistant", "Updated!")]
    s3 += filler(3)
    s4 = filler(2)
    return [(0, s1), (14, s2), (35, s3), (56, s4)]


def probes(c):
    """(question, expected substring, kind). 'current' probes require the newest value to win."""
    ps = [("What is my name?", c["name"], "identity"),
          ("Which city do I live in?", c["city"], "identity"),
          ("What grind do I prefer for my coffee?", c["grind"], "preference"),
          ("What happened with my damaged order?", f"#{c['order']}", "open issue")]
    if c["allergy"]:
        ps.append(("Am I allergic to anything?", c["allergy"], "safety"))
    ps.append(("Which roast coffee do I prefer?", f"{c['new_roast'] or c['roast']} roast", "current pref"))
    return ps


def paraphrased_probes(c):
    """Same needs, different words - no keyword overlap with what the customer said."""
    ps = [("Who am I?", c["name"], "identity"),
          ("Where should you ship my parcel?", c["city"], "identity"),
          ("How should my beans be prepared before shipping?", c["grind"], "preference"),
          ("Any update on the broken package?", f"#{c['order']}", "open issue")]
    if c["allergy"]:
        ps.append(("Is there any ingredient I can't have?", c["allergy"], "safety"))
    ps.append(("Pick a coffee for me based on my taste", f"{c['new_roast'] or c['roast']} roast", "current pref"))
    return ps


def run(workdir: Path, ollama_model=None, paraphrase=False):
    customers = make_customers()
    rng = random.Random(42)
    now = {"t": T0}
    db = workdir / "support.db"
    rows = []
    stats = {"turns": 0, "kept_episodes": 0, "facts": 0, "ctx_chars": [], "history_chars": []}

    for c in customers:
        mem = AgentMemory(db, namespace=c["id"], clock=lambda: now["t"])
        history = []  # everything ever said - what a "window" bot draws from
        for day, turns in conversations(c, rng):
            now["t"] = T0 + day * DAY
            with mem.session(week=day // 7):
                for role, text in turns:
                    mem.add(text, role=role)
                    history.append(text)
                    now["t"] += 30
        now["t"] = T0 + 56 * DAY + 3600
        stats["turns"] += len(history)
        stats["kept_episodes"] += mem.episodic.count()
        stats["facts"] += mem.semantic.count()

        with mem.session(week=8):
            for q, expected, kind in (paraphrased_probes(c) if paraphrase else probes(c)):
                hits = mem.recall(q, k=3, reinforce=False)
                ctx = mem.build_context(q)
                stats["ctx_chars"].append(len(ctx))
                stats["history_chars"].append(sum(len(h) for h in history))
                window = " ".join(history[-20:])
                ok_top3 = any(expected.lower() in h.item.content.lower() for h in hits)
                if kind == "current pref":
                    # the first roast-mentioning memory must carry the *current* value
                    first = next((h.item.content for h in hits if "roast" in h.item.content.lower()), "")
                    ok_top3 = expected.lower() in first.lower()
                    stale = f"{c['roast']} roast"
                    ok_ctx = expected in ctx and (not c["new_roast"] or ctx.find(expected) < max(ctx.find(stale), 0) or stale not in ctx)
                else:
                    ok_ctx = expected.lower() in ctx.lower()
                answer = None
                if ollama_model:
                    from agentmem.integrations.ollama import OllamaAgent
                    answer = OllamaAgent(mem, model=ollama_model, system=(
                        "You are the support assistant of an online coffee shop. Answer briefly.")).chat(q)
                rows.append({"customer": c["id"], "kind": kind, "question": q, "expected": expected,
                             "stateless": False, "window20": expected.lower() in window.lower(),
                             "agentmem_top3": ok_top3, "agentmem_context": ok_ctx,
                             "top_hit": hits[0].item.content if hits else None, "llm_answer": answer})
        mem.close()
    return rows, stats


def summarize(rows, stats):
    kinds = sorted({r["kind"] for r in rows})
    def pct(rs, key):
        return 100.0 * sum(r[key] for r in rs) / len(rs) if rs else 0.0
    table = []
    for k in kinds + ["ALL"]:
        rs = rows if k == "ALL" else [r for r in rows if r["kind"] == k]
        table.append({"category": k, "n": len(rs), "stateless": pct(rs, "stateless"),
                      "window20": pct(rs, "window20"), "agentmem_top3": pct(rs, "agentmem_top3"),
                      "agentmem_context": pct(rs, "agentmem_context")})
    avg_ctx = sum(stats["ctx_chars"]) / len(stats["ctx_chars"])
    avg_hist = sum(stats["history_chars"]) / len(stats["history_chars"])
    return {"customers": len({r['customer'] for r in rows}), "probes": len(rows), "table": table,
            "turns_observed": stats["turns"], "episodes_kept": stats["kept_episodes"],
            "facts_learned": stats["facts"], "avg_context_chars": round(avg_ctx),
            "avg_full_history_chars": round(avg_hist),
            "failures": [r for r in rows if not r["agentmem_context"]][:15]}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--ollama", metavar="MODEL", help="also have a local Ollama model answer each probe")
    ap.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "benchmarks/results/support_desk.json"))
    a = ap.parse_args(argv)
    t = time.perf_counter()
    results = {}
    for mode in ("direct", "paraphrased"):
        with tempfile.TemporaryDirectory() as d:
            rows, stats = run(Path(d), a.ollama, paraphrase=(mode == "paraphrased"))
        results[mode] = (summarize(rows, stats), rows)
        report(mode, results[mode][0])
    out = {m: {**s, "rows": rows} for m, (s, rows) in results.items()}
    out["seconds"] = round(time.perf_counter() - t, 2)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2))
    print(f"\nwrote {a.out} ({out['seconds']}s)")
    return out


def report(mode, s):
    print(f"\n== {mode} questions - {s['customers']} customers, {s['probes']} week-8 questions ==\n")
    print(f"{'category':<14}{'n':>4}{'stateless':>11}{'window-20':>11}{'agentmem@3':>12}{'agentmem ctx':>14}")
    for r in s["table"]:
        print(f"{r['category']:<14}{r['n']:>4}{r['stateless']:>10.0f}%{r['window20']:>10.0f}%"
              f"{r['agentmem_top3']:>11.0f}%{r['agentmem_context']:>13.0f}%")
    print(f"\nObserved {s['turns_observed']} turns -> kept {s['episodes_kept']} episodes, learned {s['facts_learned']} facts")
    print(f"Prompt memory: {s['avg_context_chars']} chars injected vs {s['avg_full_history_chars']} chars of full history")
    if s["failures"]:
        print("\nMisses (answer not in injected context):")
        for f in s["failures"][:8]:
            print(f"  [{f['customer']}] {f['question']!r} expected {f['expected']!r}; top hit: {f['top_hit']!r}")


if __name__ == "__main__":
    main(sys.argv[1:])
