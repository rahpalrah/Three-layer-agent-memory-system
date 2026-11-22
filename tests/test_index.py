"""Stage-1 index: results must match an exact full scan, and stay fresh."""
import random

import pytest

np = pytest.importorskip("numpy")

from agentmem import AgentMemory, Layer, SQLiteStorage

TOPICS = ["billing invoice refund", "kubernetes cluster deploy", "hiring interview candidate",
          "marketing campaign launch", "security audit password", "travel flight hotel",
          "design mockup figma", "legal contract review"]


def corpus(mem, n=1500, seed=3):
    rng = random.Random(seed)
    for i in range(n):
        t = rng.choice(TOPICS).split()
        words = rng.sample(t, 2) + [f"item{rng.randint(0, 300)}", rng.choice(["alpha", "beta", "gamma"])]
        mem.remember(" ".join(words), importance=rng.random())
    mem.learn("The user's top priority is the security audit.", importance=0.9)


@pytest.mark.parametrize("backend", ["numpy", "faiss"])
def test_indexed_recall_matches_full_scan(backend, tmp_path):
    if backend == "faiss":
        pytest.importorskip("faiss")
    db = tmp_path / "m.db"
    exact = AgentMemory(db, index=None, auto_consolidate=False)
    corpus(exact)
    fast = AgentMemory(db, index=backend, auto_consolidate=False)
    overlaps = []
    for q in ["billing refund", "deploy the cluster", "password audit", "item42 alpha", "top priority"]:
        a = [h.item.id for h in exact.recall(q, k=10, reinforce=False)]
        b = [h.item.id for h in fast.recall(q, k=10, reinforce=False)]
        overlaps.append(len(set(a) & set(b)) / 10)
        if backend == "numpy":
            assert a[:3] == b[:3], q
    assert sum(overlaps) / len(overlaps) >= (0.99 if backend == "numpy" else 0.9), overlaps


def test_index_sees_own_writes_deletes_and_other_processes(tmp_path):
    db = tmp_path / "m.db"
    m = AgentMemory(db, auto_consolidate=False)
    a = m.remember("The quarterly zebra report is due")
    assert m.recall("zebra report", k=1)[0].item.id == a.id
    m.forget(a.id)
    assert all(h.item.id != a.id for h in m.recall("zebra report", k=5))
    other = AgentMemory(SQLiteStorage(db), auto_consolidate=False)  # another connection
    b = other.remember("A giraffe joined the zebra team")
    assert m.recall("giraffe", k=1)[0].item.id == b.id               # detected via data_version


def test_superseded_and_layer_filter_respected(tmp_path):
    m = AgentMemory(tmp_path / "m.db")
    with m.session():
        m.add("I really like dark roast coffee.", role="user")
    with m.session():
        m.add("I changed my mind, I prefer light roast coffee now.", role="user")
    facts = [h.item.content for h in m.recall("roast coffee", k=5, layers=[Layer.SEMANTIC])]
    assert "The user likes dark roast coffee." not in facts
    assert all(h.item.layer == Layer.SEMANTIC for h in m.recall("roast", layers=[Layer.SEMANTIC]))
