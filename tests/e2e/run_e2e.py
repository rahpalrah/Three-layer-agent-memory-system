"""End-to-end system test for agentmem, run against the *installed* package.

    python tests/e2e/run_e2e.py [--workdir DIR]

Each scenario uses separate OS processes where persistence matters, so
nothing is shared through Python state.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import threading
import time
import traceback
from pathlib import Path

PY = sys.executable
RESULTS = []


def run_py(code: str, *args: str, check: bool = True, env=None) -> subprocess.CompletedProcess:
    p = subprocess.run([PY, "-c", textwrap.dedent(code), *args], capture_output=True, text=True,
                       env=env, timeout=300)
    if check and p.returncode != 0:
        raise AssertionError(f"subprocess failed ({p.returncode}):\n{p.stdout}\n{p.stderr}")
    return p


def cli(db, *args, ns="default", check=True) -> str:
    exe = Path(PY).parent / "agentmem"
    p = subprocess.run([str(exe), "--db", str(db), "--ns", ns, *args], capture_output=True, text=True)
    if check and p.returncode != 0:
        raise AssertionError(f"cli {args} failed:\n{p.stdout}\n{p.stderr}")
    return p.stdout


def scenario(fn):
    def wrapper(work: Path):
        d = work / fn.__name__
        d.mkdir()
        t = time.perf_counter()
        try:
            note = fn(d) or ""
            status = "SKIP" if note.startswith("SKIPPED") else "PASS"
            RESULTS.append((fn.__name__, status, time.perf_counter() - t, note))
        except Exception as e:  # noqa: BLE001
            RESULTS.append((fn.__name__, "FAIL", time.perf_counter() - t,
                            f"{type(e).__name__}: {e}\n{traceback.format_exc(limit=3)}"))
    wrapper.__name__ = fn.__name__
    return wrapper


# ---------------------------------------------------------------------------
@scenario
def s01_install_surface(d):
    p = run_py("""
        import agentmem, agentmem.integrations.anthropic as a
        assert agentmem.__version__ == "0.1.0"
        missing = [n for n in agentmem.__all__ if not hasattr(agentmem, n)]
        assert not missing, missing
        print(agentmem.__file__)
    """)
    assert "site-packages" in p.stdout, f"not testing installed package: {p.stdout}"
    return "imports from site-packages"


@scenario
def s02_cross_process_persistence(d):
    db = d / "m.db"
    run_py("""
        import sys; from agentmem import AgentMemory
        m = AgentMemory(sys.argv[1], namespace="alice")
        with m.session(channel="web"):
            m.add("Hi, my name is Alice Smith and I work at Acme Corp.", role="user")
            m.add("Great to meet you Alice!", role="assistant")
            m.add("I'm allergic to peanuts. Remember that my flight is on December 12.", role="user")
        m.close()
    """, str(db))
    p = run_py("""
        import sys, json; from agentmem import AgentMemory
        m = AgentMemory(sys.argv[1], namespace="alice")
        out = {
          "facts": sorted(f.content for f in m.semantic.facts()),
          "flight": m.recall("when is my flight?", k=1)[0].item.content,
          "work": m.recall("where does the user work?", k=1)[0].item.content,
          "sessions": [s.metadata for s in m.sessions()],
          "ctx": m.build_context("food allergies"),
          "other_ns": len(AgentMemory(sys.argv[1], namespace="bob").recall("Alice")),
        }
        print(json.dumps(out))
    """, str(db))
    o = json.loads(p.stdout)
    for f in ["The user's name is Alice Smith.", "The user works at Acme Corp.",
              "The user is allergic to peanuts.", "The user's flight is on December 12."]:
        assert f in o["facts"], (f, o["facts"])
    assert "December 12" in o["flight"], o["flight"]
    assert "Acme" in o["work"], o["work"]
    assert o["sessions"] == [{"channel": "web"}]
    assert "peanuts" in o["ctx"]
    assert o["other_ns"] == 0, "namespace leak"
    return f"{len(o['facts'])} facts recalled in a fresh process"


@scenario
def s03_crash_and_resume(d):
    db = d / "m.db"
    p = run_py("""
        import sys, os; from agentmem import AgentMemory
        m = AgentMemory(sys.argv[1])
        m.start_session("sess-42")
        m.add("My name is Bob and I prefer tea.", role="user")
        m.add("Remember that the launch is on Friday.", role="user")
        os._exit(9)   # hard crash: no end_session, no close
    """, str(db), check=False)
    assert p.returncode == 9
    p = run_py("""
        import sys, json; from agentmem import AgentMemory
        m = AgentMemory(sys.argv[1])
        m.start_session("sess-42")
        restored = [i.content for i in m.working]
        m.add("Also I like jazz.", role="user")
        rep = m.end_session()
        print(json.dumps({"restored": restored, "learned": rep.facts_learned,
                          "facts": sorted(f.content for f in m.semantic.facts()),
                          "sessions": len(m.sessions()), "working_left": m.stats()["working"]}))
    """, str(db))
    o = json.loads(p.stdout)
    assert o["restored"] == ["My name is Bob and I prefer tea.", "Remember that the launch is on Friday."], o
    assert o["sessions"] == 1 and o["working_left"] == 0
    for f in ["The user's name is Bob.", "The user prefers tea.", "The user likes jazz.",
              "The launch is on Friday."]:
        assert f in o["facts"], (f, o["facts"])
    return "working memory survived kill -9 and consolidated on resume"


@scenario
def s04_concurrent_processes(d):
    db = d / "m.db"
    code = """
        import sys; from agentmem import AgentMemory
        ns, n = sys.argv[2], int(sys.argv[3])
        m = AgentMemory(sys.argv[1], namespace=ns, working_capacity=5)
        with m.session():
            for i in range(n):
                m.add(f"[{ns}] status update number {i} about project deadline", role="user")
        m.close()
    """
    procs = [subprocess.Popen([PY, "-c", textwrap.dedent(code), str(db), ns, "60"],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
             for ns in ["shared", "shared", "shared", "w1", "w2", "w3"]]
    errs = [(p.wait(), p.stderr.read()) for p in procs]
    bad = [e for rc, e in errs if rc != 0]
    assert not bad, bad[0][-1500:]
    p = run_py("""
        import sys, json; from agentmem import AgentMemory
        print(json.dumps({ns: AgentMemory(sys.argv[1], namespace=ns).stats() for ns in ["shared","w1","w2","w3"]}))
    """, str(db))
    o = json.loads(p.stdout)
    assert o["shared"]["episodic"] == 180 and o["shared"]["sessions"] == 3, o["shared"]
    for ns in ["w1", "w2", "w3"]:
        assert o[ns]["episodic"] == 60, o[ns]
    return "6 writer processes, 360 episodes, no lost writes"


@scenario
def s05_threads_in_one_process(d):
    from agentmem import AgentMemory, SQLiteStorage
    st = SQLiteStorage(d / "m.db")
    errors = []

    def worker(i):
        try:
            m = AgentMemory(storage=st, namespace=f"t{i}")
            with m.session():
                for j in range(40):
                    m.add(f"thread {i} note {j} remember the budget", role="user")
                    m.recall("budget", k=3)
        except Exception as e:  # noqa: BLE001
            errors.append(repr(e))

    ts = [threading.Thread(target=worker, args=(i,)) for i in range(8)]
    [t.start() for t in ts]
    [t.join() for t in ts]
    assert not errors, errors[:3]
    counts = [st.count(f"t{i}") for i in range(8)]
    assert all(c >= 40 for c in counts), counts
    return "8 threads sharing one SQLiteStorage"


@scenario
def s06_cli_workflow(d):
    db, db2 = d / "m.db", d / "copy.db"
    assert "added" in cli(db, "add", "My name is Turing and I live in Manchester.", ns="u")
    cli(db, "learn", "The user's favourite language is Lisp.", "--key", "user.lang", ns="u")
    cli(db, "learn", "The user's favourite language is Python.", "--key", "user.lang", ns="u")
    out = cli(db, "list", "--layer", "semantic", ns="u")
    assert "Python" in out and "Lisp" not in out, out
    assert "Manchester" in cli(db, "search", "where does the user live", ns="u")
    stats = json.loads(cli(db, "stats", ns="u"))
    assert stats["semantic"] == 3 and stats["sessions"] == 1, stats
    assert json.loads(cli(db, "consolidate", ns="u"))["episodes_considered"] == 0
    assert len(cli(db, "sessions", ns="u").strip().splitlines()) == 1
    cli(db, "export", str(d / "dump.json"), ns="u")
    assert "imported 4" in cli(db2, "import", str(d / "dump.json"), ns="restored"), "import count"
    assert "Turing" in cli(db2, "search", "name", ns="restored")
    fid = cli(db2, "list", "--layer", "semantic", ns="restored").split()[0]
    assert "forgot" in cli(db2, "forget", fid, ns="restored")
    assert json.loads(cli(db2, "stats", ns="restored"))["semantic"] == 2
    r = subprocess.run([str(Path(PY).parent / "agentmem"), "--db", str(db2), "forget", "zzz"],
                       capture_output=True, text=True)
    assert r.returncode == 1
    return "add/learn/list/search/stats/consolidate/sessions/export/import/forget"


@scenario
def s07_long_conversation(d):
    from agentmem import AgentMemory
    m = AgentMemory(d / "m.db", working_capacity=10, working_token_budget=600)
    peak = 0
    with m.session():
        m.add("My name is Dana and I live in Lisbon.", role="user")
        for i in range(200):
            m.add(f"Let's keep discussing item {i} of the backlog in some detail here.", role="user")
            m.add("ok", role="user")
            m.add(f"Sure, item {i} looks reasonable.", role="assistant")
            peak = max(peak, len(m.working))
            assert m.working.tokens <= 600
        mid = m.recall("where does the user live?", k=3)
        assert any("Lisbon" in h.item.content for h in mid), "early fact lost mid-session"
    s = m.stats()
    assert peak <= 10
    assert s["episodic"] < 601, "trivia not discarded"
    assert m.semantic.get_by_key("user.location").content == "The user lives in Lisbon."
    return f"601 turns, working peak {peak}, {s['episodic']} episodes kept, {s['semantic']} facts"


@scenario
def s08_decay_and_forgetting(d):
    from agentmem import AgentMemory
    m = AgentMemory(d / "m.db", auto_consolidate=False)
    year = 365 * 86400
    old_junk = [m.remember(f"random chatter {i}", importance=0.1) for i in range(20)]
    old_key = m.remember("We signed the Series A term sheet", importance=0.95)
    old_pin = m.remember("Account id is 7731", importance=0.05, pinned=True)
    recalled = m.remember("weekly sync notes", importance=0.3)
    for it in old_junk + [old_key, old_pin, recalled]:
        it.created_at = it.last_accessed = time.time() - year
        m.storage.put(it)
    for _ in range(15):
        m.episodic.retrieve("weekly sync notes", k=1)  # reinforcement
    fresh = m.remember("fresh low-importance note", importance=0.2)
    removed = m.episodic.forget(threshold=0.2)
    ids = {i.id for i in m.episodic.timeline()}
    assert removed == 20, removed
    assert {old_key.id, old_pin.id, recalled.id, fresh.id} <= ids
    m.episodic.forget(threshold=0.0, max_items=2)
    ids = {i.id for i in m.episodic.timeline()}
    assert old_pin.id in ids and len(ids) == 2, ("cap counts all items, never evicts pinned", len(ids))
    return "forgot 20 faded, kept important/pinned/reinforced/fresh; size cap honoured"


@scenario
def s09_scale(d):
    from agentmem import AgentMemory
    m = AgentMemory(d / "m.db", auto_consolidate=False)
    topics = ["billing", "kubernetes", "hiring", "marketing", "security", "travel", "design", "legal"]
    t = time.perf_counter()
    for i in range(5000):
        m.remember(f"Note {i}: discussion about {topics[i % 8]} item {i // 8}", importance=0.3 + (i % 7) / 10)
    write = time.perf_counter() - t
    m.learn("The user's top priority is security hardening.", importance=0.9)
    t = time.perf_counter()
    for q in topics:
        hits = m.recall(f"{q} discussion", k=5)
        assert all(q in h.item.content or h.item.layer.value == "semantic" for h in hits[:3]), (q, hits[:3])
    per_q = (time.perf_counter() - t) / len(topics)
    assert m.recall("what is my top priority?", k=1)[0].item.content.startswith("The user's top priority")
    size = (d / "m.db").stat().st_size / 1e6
    return f"5000 episodes: write {write:.1f}s, recall {per_q*1000:.0f} ms/query, db {size:.1f} MB"


@scenario
def s10_claude_integration_real_sdk(d):
    try:
        import anthropic
    except ImportError:
        return "SKIPPED (anthropic not installed)"
    try:  # newer SDKs are built on httpx2, older ones on httpx
        import httpx2 as httpx
    except ImportError:
        import httpx
    from agentmem import AgentMemory
    from agentmem.integrations.anthropic import MemoryAgent
    requests = []

    def handler(req: httpx.Request) -> httpx.Response:
        body = json.loads(req.content)
        requests.append(body)
        sysmsg = body.get("system", "")
        if "Rate how important" in sysmsg:
            text = "9" if "vegetarian" in body["messages"][0]["content"] else "2"
        elif "Extract durable facts" in sysmsg:
            ids = [l.split("]")[0][1:] for l in body["messages"][0]["content"].splitlines()]
            text = ('Here you go:\n[{"fact": "The user is vegetarian.", "importance": 0.9, '
                    '"key": "user.diet", "sources": ["%s"]}]' % ids[0])
        else:
            text = "Try a mushroom risotto." if "vegetarian" in sysmsg else "Hello!"
        return httpx.Response(200, json={
            "id": "msg_1", "type": "message", "role": "assistant", "model": body["model"],
            "content": [{"type": "text", "text": text}], "stop_reason": "end_turn",
            "stop_sequence": None, "usage": {"input_tokens": 10, "output_tokens": 5}})

    client = anthropic.Anthropic(api_key="test", http_client=httpx.Client(transport=httpx.MockTransport(handler)),
                                 max_retries=0)
    db = d / "m.db"
    m = AgentMemory(db, namespace="u")
    agent = MemoryAgent(client, m, use_llm_memory=True)
    with m.session():
        assert agent.chat("Hi! I'm vegetarian, by the way.") == "Hello!"
        agent.chat("Thanks")
    m.close()
    m2 = AgentMemory(db, namespace="u")
    agent2 = MemoryAgent(client, m2, use_llm_memory=True)
    with m2.session():
        reply = agent2.chat("Suggest a dinner recipe")
    assert reply == "Try a mushroom risotto.", reply
    chat_reqs = [r for r in requests if r["system"].startswith("You are")]
    last = chat_reqs[-1]
    assert "<memory>" in last["system"] and "The user is vegetarian." in last["system"]
    for r in chat_reqs:
        roles = [x["role"] for x in r["messages"]]
        assert roles[0] == "user" and all(a != b for a, b in zip(roles, roles[1:])), roles
    assert m2.semantic.get_by_key("user.diet") is not None
    return f"{len(requests)} API calls through real anthropic SDK; fact carried across sessions"


@scenario
def s11_examples(d):
    root = Path(__file__).resolve().parents[2] / "examples"
    for ex in ["quickstart.py", "custom_backend.py"]:
        p1 = subprocess.run([PY, str(root / ex)], cwd=d, capture_output=True, text=True)
        assert p1.returncode == 0, p1.stderr
    p2 = subprocess.run([PY, str(root / "quickstart.py")], cwd=d, capture_output=True, text=True)
    assert "'sessions': 1" in p2.stdout.splitlines()[0], "second run did not see first run"
    assert "deadline is June 3" in p2.stdout
    return "quickstart (x2) and custom_backend ran"


@scenario
def s12_edge_cases(d):
    from agentmem import AgentMemory
    m = AgentMemory(d / "m.db")
    nasty = ["", "   ", "'); DROP TABLE memories; --", "Emoji 🧠🚀 and ünïcödé — 日本語のテキスト",
             "x" * 200_000, "Remember that\nmulti\nline\ttext works", "null\x00byte"]
    with m.session():
        for t in nasty:
            m.add(t, role="user")
        m.add("tool output", role="tool", metadata={"nested": {"a": [1, 2, {"b": None}]}})
    assert m.storage.count("default") > 0
    assert m.recall("日本語", k=1)[0].item.content.startswith("Emoji")
    assert m.recall("") is not None and m.recall(None) is not None
    assert m.build_context(None) is not None
    assert m.forget("does-not-exist") is False
    m.end_session()  # no active session -> no-op
    m.reset()
    assert m.stats()["episodic"] == 0 and m.stats()["semantic"] == 0
    with AgentMemory() as mm:          # in-memory + context manager
        mm.add("hi there, I am a pilot", role="user")
    return "SQL-ish, unicode, 200 KB text, NUL byte, empty, nested metadata, reset"


@scenario
def s13_business_support_desk(d):
    root = Path(__file__).resolve().parents[2]
    out = d / "support.json"
    p = subprocess.run([PY, str(root / "examples" / "support_desk.py"), "--out", str(out)],
                       cwd=d, capture_output=True, text=True, timeout=300)
    assert p.returncode == 0, p.stderr[-2000:]
    r = json.loads(out.read_text())
    direct = {row["category"]: row for row in r["direct"]["table"]}
    para = {row["category"]: row for row in r["paraphrased"]["table"]}
    assert direct["ALL"]["stateless"] == 0
    assert direct["ALL"]["agentmem_context"] >= 95, direct["ALL"]
    assert para["ALL"]["agentmem_context"] >= 90, para["ALL"]
    assert direct["current pref"]["agentmem_top3"] >= 90, "change-of-mind not honoured"
    assert direct["ALL"]["agentmem_context"] > direct["ALL"]["window20"] + 50
    return (f"answer in prompt: {direct['ALL']['agentmem_context']:.0f}% direct / "
            f"{para['ALL']['agentmem_context']:.0f}% paraphrased vs window-20 {direct['ALL']['window20']:.0f}%")


@scenario
def s14_ollama_adapter_mock_server(d):
    root = Path(__file__).resolve().parents[2]
    p = subprocess.run([PY, "-m", "pytest", "-q", "-p", "no:cacheprovider", str(root / "tests" / "test_ollama.py")],
                       cwd=d, capture_output=True, text=True, timeout=120)
    assert p.returncode == 0, p.stdout[-2000:]
    return "OllamaAgent + OllamaEmbedder against a mock Ollama HTTP server"


@scenario
def s15_semantic_embeddings_minilm(d):
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        return "SKIPPED (pip install agentmem[onnx])"
    from agentmem import AgentMemory
    db = d / "m.db"
    m = AgentMemory(db, namespace="u", embedder="minilm")
    with m.session():
        m.add("My name is Rosa and I live in Valencia.", role="user")
        m.add("I'm allergic to shellfish.", role="user")
        m.add("I really like dark roast coffee.", role="user")
    with m.session():
        m.add("Actually I changed my mind, I prefer light roast coffee now.", role="user")
    m.close()
    m2 = AgentMemory(db, namespace="u", embedder="minilm")   # fresh process-equivalent
    top = m2.recall("Is there any seafood I must avoid?", k=1)[0].item.content
    assert "shellfish" in top, top                               # no shared keywords: semantic match
    facts = {f.content for f in m2.semantic.facts()}
    assert "The user prefers light roast coffee now." in facts and "The user likes dark roast coffee." not in facts, facts
    assert m2.embedder.dim == 384
    return "paraphrase recall without shared keywords; change-of-mind with semantic dedup"


@scenario
def s16_index_at_scale(d):
    try:
        import numpy  # noqa: F401
    except ImportError:
        return "SKIPPED (pip install agentmem[fast])"
    from agentmem import AgentMemory, SQLiteStorage
    db = d / "m.db"
    m = AgentMemory(db, auto_consolidate=False)
    words = "billing kubernetes hiring marketing security travel design legal budget roadmap".split()
    texts = [f"{words[i % 10]} {words[(i * 7) % 10]} note {i}" for i in range(20000)]
    m.remember_many(texts)
    needle = m.remember("The vault passphrase hint is the name of the first office dog", importance=0.9)
    m.recall("warmup", k=1)
    t = time.perf_counter()
    for _ in range(20):
        hits = m.recall("office dog passphrase", k=5, reinforce=False)
    ms = (time.perf_counter() - t) / 20 * 1000
    assert hits[0].item.id == needle.id
    assert ms < 150, f"indexed recall too slow: {ms:.0f} ms"
    other = AgentMemory(SQLiteStorage(db), auto_consolidate=False)          # second connection
    late = other.remember("Quarterly zebra migration report")
    assert m.recall("zebra migration", k=1)[0].item.id == late.id           # index refreshed
    m.forget(needle.id)
    assert all(h.item.id != needle.id for h in m.recall("office dog passphrase", k=5))
    return f"20k memories: {ms:.0f} ms/recall indexed; cross-connection freshness; deletes"


@scenario
def s17_cross_encoder_rerank(d):
    try:
        from agentmem.rerank import CrossEncoderReranker
        rr = CrossEncoderReranker()
    except Exception as e:  # noqa: BLE001
        return f"SKIPPED (reranker model unavailable: {type(e).__name__})"
    from agentmem import AgentMemory
    m = AgentMemory(embedder="minilm", reranker=rr, auto_consolidate=False)
    m.remember("Berlin has a population of 3.5 million registered inhabitants.")
    m.remember("How many people visit the Berlin museum each year is unknown.")
    m.remember("People in Berlin like to live near parks.")
    top = m.recall("How many people live in Berlin?", k=1)[0].item.content
    assert top.startswith("Berlin has a population"), top
    return "cross-encoder puts the answering passage first"


@scenario
def s18_qa_harness_with_mock_llm(d):
    try:
        import onnxruntime  # noqa: F401
    except ImportError:
        return "SKIPPED (pip install agentmem[onnx])"
    import http.server
    root = Path(__file__).resolve().parents[2]
    data = root / "benchmarks" / "data" / "locomo10.json"
    if not data.exists():
        return "SKIPPED (LoCoMo not downloaded)"

    class H(http.server.BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            sysmsg = body["messages"][0]["content"]
            text = "CORRECT" if sysmsg.startswith("You grade") else "unknown"
            out = json.dumps({"message": {"role": "assistant", "content": text}}).encode()
            self.send_response(200); self.send_header("Content-Length", str(len(out))); self.end_headers()
            self.wfile.write(out)
    srv = http.server.HTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    env = dict(os.environ, OLLAMA_HOST=f"http://127.0.0.1:{srv.server_port}")
    out = d / "qa.json"
    p = subprocess.run([PY, str(root / "benchmarks" / "locomo_qa.py"), "--llm", "ollama:mock", "--judge",
                        "--conversations", "2", "--limit-questions", "5", "--no-rerank", "--out", str(out)],
                       capture_output=True, text=True, env=env, timeout=600)
    srv.shutdown()
    assert p.returncode == 0, p.stderr[-2000:]
    r = json.loads(out.read_text())
    s = r["summary"]["agentmem (minilm hybrid)"]
    assert s["f1"] == 0.0 and s["j"] == 100.0 and 0 < s["coverage"] <= 100, s  # mock says "unknown"/"CORRECT"
    return "LLM answer + judge path runs end to end (mock Ollama); metrics computed"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workdir")
    ap.add_argument("-k", help="only scenarios containing this substring")
    a = ap.parse_args()
    work = Path(a.workdir or tempfile.mkdtemp(prefix="agentmem-e2e-"))
    if work.exists():
        shutil.rmtree(work)
    work.mkdir(parents=True)
    scen = [v for k, v in globals().items() if k.startswith("s") and k[1:3].isdigit()]
    for s in scen:
        if not a.k or a.k in s.__name__:
            s(work)
    w = max(len(r[0]) for r in RESULTS)
    for name, status, dt, note in RESULTS:
        print(f"{status}  {name:<{w}}  {dt:6.2f}s  {note.splitlines()[0] if status != 'FAIL' else ''}")
        if status == "FAIL":
            print(textwrap.indent(note, "      "))
    failed = sum(r[1] == "FAIL" for r in RESULTS)
    skipped = sum(r[1] == "SKIP" for r in RESULTS)
    print(f"\n{len(RESULTS) - failed - skipped} passed, {skipped} skipped, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
