import time
from types import SimpleNamespace

import pytest

from agentmem import (AgentMemory, HeuristicImportanceScorer, Layer, MemoryItem,
                      RetrievalWeights, SQLiteStorage)
from agentmem.scoring import effective_importance


@pytest.fixture
def db(tmp_path):
    return tmp_path / "mem.db"


# ---------------------------------------------------------------- importance
def test_importance_orders_signal_over_chatter():
    s = HeuristicImportanceScorer()
    trivial = s.score("ok thanks", role="user")
    fact = s.score("Remember that my daughter is allergic to peanuts", role="user")
    plain = s.score("The weather looks cloudy right now", role="user")
    assert trivial < plain < fact
    assert 0 <= trivial and fact <= 1


def test_importance_decays_and_reinforces():
    w = RetrievalWeights(importance_half_life_days=1)
    now = time.time()
    old = MemoryItem("x", importance=0.8, created_at=now - 10 * 86400)
    fresh = MemoryItem("x", importance=0.8, created_at=now)
    assert effective_importance(old, now, w) < effective_importance(fresh, now, w)
    old.access_count = 10
    assert effective_importance(old, now, w) > 0.4
    assert effective_importance(MemoryItem("p", importance=0.1, pinned=True), now, w) == 1.0


# ------------------------------------------------------------------- working
def test_working_memory_evicts_to_episodic_and_discards_trivia():
    mem = AgentMemory(working_capacity=3, working_token_budget=None, auto_consolidate=False)
    with mem.session():
        mem.add("ok", role="user")  # trivial -> discarded on eviction
        for i in range(5):
            mem.add(f"Project milestone {i} is due on Friday", role="user")
        assert len(mem.working) == 3
        eps = [e.content for e in mem.episodic.timeline()]
        assert "ok" not in eps
        assert "Project milestone 0 is due on Friday" in eps
    # session end flushes everything
    assert len(mem.working) == 0
    assert mem.episodic.count() == 5


def test_token_budget_eviction():
    mem = AgentMemory(working_capacity=100, working_token_budget=50)
    with mem.session():
        for _ in range(5):
            mem.add("word " * 20, role="user")  # ~25 tokens each
        assert mem.working.tokens <= 50


# ------------------------------------------------------------------ episodic
def test_episodic_retrieval_relevance_and_reinforcement():
    mem = AgentMemory(auto_consolidate=False)
    a = mem.remember("We deployed the payments service to production", importance=0.6)
    mem.remember("Lunch was pasta at the Italian place", importance=0.6)
    hits = mem.episodic.retrieve("payments deploy production", k=1)
    assert hits[0].item.id == a.id
    assert mem.storage.get(a.id).access_count == 1


def test_forgetting_keeps_pinned_and_important():
    mem = AgentMemory(auto_consolidate=False)
    old = time.time() - 365 * 86400
    junk = mem.remember("meh", importance=0.05)
    pinned = mem.remember("core", importance=0.05, pinned=True)
    keep = mem.remember("big decision", importance=0.9)
    for it in (junk, pinned, keep):
        it.created_at = it.last_accessed = old
        mem.storage.put(it)
    mem.episodic.forget(threshold=0.15)
    ids = {i.id for i in mem.episodic.timeline()}
    assert junk.id not in ids and pinned.id in ids and keep.id in ids


# ------------------------------------------------------------------ semantic
def test_semantic_dedup_reinforces():
    mem = AgentMemory()
    f1 = mem.learn("The user prefers dark mode in every app", importance=0.5)
    f2 = mem.learn("The user prefers dark mode in every app.", importance=0.5)
    assert f1.id == f2.id
    assert mem.semantic.count() == 1
    assert f2.metadata["evidence"] == 2 and f2.importance > 0.5


def test_semantic_key_supersedes_with_history():
    mem = AgentMemory()
    mem.learn("The user lives in Paris.", key="user.location")
    f = mem.learn("The user lives in Berlin.", key="user.location")
    assert mem.semantic.count() == 1
    assert "Berlin" in mem.semantic.get_by_key("user.location").content
    assert f.metadata["history"][0]["content"] == "The user lives in Paris."


# ------------------------------------------------------------- consolidation
def test_consolidation_extracts_facts_from_session():
    mem = AgentMemory()
    with mem.session():
        mem.add("Hi! My name is Ada Lovelace and I live in London.", role="user")
        mem.add("Nice to meet you Ada.", role="assistant")
        mem.add("I prefer short answers. Remember that my deadline is March 3.", role="user")
    facts = [f.content for f in mem.semantic.facts()]
    assert "The user's name is Ada Lovelace." in facts
    assert "The user lives in London." in facts
    assert any("short answers" in f for f in facts)
    assert any("deadline is March 3" in f for f in facts)
    # re-consolidating is idempotent
    assert mem.consolidate().episodes_considered == 0


# --------------------------------------------------- cross-session persistence
def test_cross_session_persistence(db):
    m1 = AgentMemory(db, namespace="u1")
    with m1.session(topic="intro"):
        m1.add("My name is Grace and I work at Navy Labs.", role="user")
    m1.close()

    m2 = AgentMemory(db, namespace="u1")  # new process, same file
    hits = m2.recall("where does the user work?", k=3)
    assert any("Navy Labs" in h.item.content for h in hits)
    assert m2.sessions()[0].metadata == {"topic": "intro"}
    assert "Grace" in m2.build_context("user name")
    # namespaces are isolated
    assert AgentMemory(db, namespace="u2").recall("Grace") == []


def test_resume_interrupted_session(db):
    m1 = AgentMemory(db)
    s = m1.start_session("sess-1")
    m1.add("Draft the Q3 roadmap", role="user")
    m1.storage.close()  # simulate crash: no end_session

    m2 = AgentMemory(db)
    m2.start_session("sess-1")
    assert [i.content for i in m2.working] == ["Draft the Q3 roadmap"]
    assert m2.current_session.id == s.id


def test_recall_spans_layers_and_dedups():
    mem = AgentMemory(auto_consolidate=False)
    mem.learn("The user's favourite language is Python.")
    mem.remember("User asked about Python packaging")
    with mem.session():
        mem.add("Let's talk Python decorators", role="user")
        layers = {h.item.layer for h in mem.recall("python", k=10)}
    assert layers == {Layer.SEMANTIC, Layer.EPISODIC, Layer.WORKING}


def test_hooks_and_export_import(tmp_path):
    events = []
    mem = AgentMemory()
    mem.on(lambda e, p: events.append(e))
    with mem.session():
        mem.add("My name is Linus.", role="user")
    assert {"session_start", "add", "evict", "learn", "consolidate", "session_end"} <= set(events)
    path = tmp_path / "dump.json"
    mem.export_json(path)
    import json
    other = AgentMemory(namespace="copy")
    assert other.import_data(json.loads(path.read_text())) == mem.storage.count("default")
    assert other.recall("Linus")


def test_sqlite_roundtrip(db):
    st = SQLiteStorage(db)
    it = MemoryItem("hello", tags=["t"], metadata={"a": 1}, embedding=[0.1, 0.2], pinned=True)
    st.put(it)
    back = st.get(it.id)
    assert back.embedding == pytest.approx(it.embedding)
    assert back.to_dict() == it.to_dict()
    assert st.delete(it.id) and st.get(it.id) is None


# ------------------------------------------------------------- integration
class FakeClient:
    def __init__(self):
        self.calls = []
        self.messages = SimpleNamespace(create=self._create)

    def _create(self, **kw):
        self.calls.append(kw)
        return SimpleNamespace(content=[SimpleNamespace(type="text", text="Noted!")])


def test_memory_agent_injects_long_term_context(db):
    from agentmem.integrations.anthropic import MemoryAgent, _alternate
    client = FakeClient()
    mem = AgentMemory(db)
    with mem.session():
        MemoryAgent(client, mem).chat("My name is Ada and I'm allergic to shellfish.")
    with mem.session():
        MemoryAgent(client, mem).chat("Suggest a dinner recipe for me")
    system = client.calls[-1]["system"]
    assert "shellfish" in system and "<memory>" in system
    msgs = client.calls[-1]["messages"]
    assert msgs[0]["role"] == "user"
    assert _alternate([{"role": "assistant", "content": "a"}, {"role": "user", "content": "b"},
                       {"role": "user", "content": "c"}]) == [{"role": "user", "content": "b\n\nc"}]


def test_cli_smoke(db, capsys):
    from agentmem.cli import main
    assert main(["--db", str(db), "add", "My name is Turing."]) == 0
    assert main(["--db", str(db), "search", "name"]) == 0
    assert "Turing" in capsys.readouterr().out


def test_extraction_quality():
    from agentmem.consolidation import RuleBasedExtractor
    ep = MemoryItem("Hi, my name is Ada and I live in New York. Remember that my exam is on May 2.",
                    metadata={"role": "user"}, importance=0.7)
    facts = {f.content for f in RuleBasedExtractor().extract([ep])}
    assert "The user's name is Ada." in facts
    assert "The user lives in New York." in facts
    assert "The user's exam is on May 2." in facts
    assert HeuristicImportanceScorer().score("ok thanks!", role="user") < 0.1


def test_query_relevance_beats_importance():
    mem = AgentMemory()
    with mem.session():
        mem.add("Hi, my name is Alice Smith and I work at Acme Corp.", role="user")
        mem.add("I'm allergic to peanuts. Remember that my flight is on December 12.", role="user")
    assert "Acme" in mem.recall("where does the user work?", k=1)[0].item.content
    assert "December 12" in mem.recall("when is my flight?", k=1)[0].item.content


def test_unicode_recall():
    mem = AgentMemory()
    mem.remember("Réunion à Zürich über das Budget")
    mem.remember("東京で会議があります")
    mem.remember("Something else entirely")
    assert "Zürich" in mem.recall("zürich budget", k=1)[0].item.content
    assert "東京" in mem.recall("東京の会議", k=1)[0].item.content


def test_change_of_mind_supersedes_only_same_slot():
    mem = AgentMemory()
    with mem.session():
        mem.add("I prefer whole beans for my coffee.", role="user")
        mem.add("I really like dark roast coffee.", role="user")
    with mem.session():
        mem.add("Actually I changed my mind: I prefer medium roast coffee now.", role="user")
    active = {f.content for f in mem.semantic.facts()}
    assert "The user prefers medium roast coffee now." in active
    assert "The user likes dark roast coffee." not in active          # retired
    assert "The user prefers whole beans for my coffee." in active     # different slot: untouched
    old = [f for f in mem.semantic.facts(include_superseded=True) if "dark" in f.content][0]
    assert old.metadata["superseded_by"]                               # kept for audit


def test_contentless_query_and_profile_in_context():
    mem = AgentMemory()
    with mem.session():
        mem.add("My name is Zara and I live in Nairobi.", role="user")
        for i in range(12):
            mem.add(f"I like topic number {i} a lot.", role="user")
        mem.add("Do you ship on weekends?", role="user")
    ctx = mem.build_context("Who am I?")
    assert "Zara" in ctx and "Nairobi" in ctx and ctx.startswith("User profile:")
    assert any("Zara" in h.item.content for h in mem.recall("who am I?", k=3))


def test_hybrid_search_discounts_ubiquitous_words():
    # every turn starts with the speaker's name: dense-only similarity is dominated by it
    mem = AgentMemory(auto_consolidate=False)
    for i in range(30):
        mem.remember(f"Caroline: chatting about item {i} of my week")
    target = mem.remember("Caroline: I adopted a puppy named Biscuit")
    hits = mem.recall("Caroline puppy", k=1, reinforce=False)
    assert hits[0].item.id == target.id


def test_storage_backend_accepted_positionally(tmp_path):
    st = SQLiteStorage(tmp_path / "m.db")
    m = AgentMemory(st)
    m.remember("hello")
    assert st.count("default") == 1
    with pytest.raises(TypeError):
        AgentMemory(123)
