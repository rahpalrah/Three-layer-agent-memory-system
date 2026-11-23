"""The self-contained embedder: learned from text, no downloads, deterministic."""
import random

import pytest

from agentmem import AgentMemory, cosine
from agentmem.selfembed import SelfEmbedder


def corpus(n=400, seed=0):
    rng = random.Random(seed)
    drive = ["drive", "road", "engine", "fuel", "traffic", "garage", "wheels", "tyres"]
    food = ["bread", "pastry", "bakery", "oven", "flour", "butter", "fresh", "dough"]
    out = []
    for _ in range(n):
        for topic, words in [("car", drive), ("vehicle", drive), ("bread", food)]:
            out.append(" ".join([topic] + rng.sample(words, 4) + ["today"]))
    return out


def test_words_with_shared_contexts_are_similar_and_unrelated_ones_are_not():
    e = SelfEmbedder(dim=128).fit(corpus())
    assert cosine(e.embed("car"), e.embed("vehicle")) > 0.9
    assert cosine(e.embed("car"), e.embed("bread")) < 0.2


def test_sentence_similarity_without_shared_keywords():
    e = SelfEmbedder(dim=128).fit(corpus())
    assert cosine(e.embed("my automobile needs fuel"), e.embed("the engine and the road")) > 0.5


def test_deterministic_and_unit_length():
    a = SelfEmbedder(dim=64).fit(corpus(50)).embed("car engine")
    b = SelfEmbedder(dim=64).fit(corpus(50)).embed("car engine")
    assert a == b
    assert abs(sum(x * x for x in a) - 1.0) < 1e-9


def test_must_be_fitted_first():
    with pytest.raises(RuntimeError):
        SelfEmbedder().embed("anything")


def test_works_as_memory_embedder_and_persists(tmp_path):
    emb = SelfEmbedder(dim=128).fit(corpus())
    m = AgentMemory(tmp_path / "m.db", embedder=emb)
    m.remember("I need fuel for my car on the road trip")
    m.remember("Fresh bread from the bakery oven")
    top = m.recall("vehicle petrol", k=1, reinforce=False)[0].item.content
    assert "car" in top
