"""LongMemEval harness on a tiny synthetic dataset in the real file format."""
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("numpy")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
import longmemeval as LME  # noqa: E402
import longmemeval_qa as QA  # noqa: E402
from agentmem import HashingEmbedder, HeuristicImportanceScorer  # noqa: E402


def tiny_dataset(path):
    def sess(texts, answer_idx=None):
        out = []
        for i, t in enumerate(texts):
            turn = {"role": "user" if i % 2 == 0 else "assistant", "content": t}
            if i == answer_idx:
                turn["has_answer"] = True
            out.append(turn)
        return out
    q = {
        "question_id": "q1", "question_type": "knowledge-update", "question": "Which city do I live in now?",
        "question_date": "2023/06/01 (Thu) 10:00", "answer": "Berlin", "answer_session_ids": ["answer_1"],
        "haystack_session_ids": ["s_late", "answer_1", "s_early"],      # deliberately out of date order
        "haystack_dates": ["2023/05/20 (Sat) 09:00", "2023/05/10 (Wed) 09:00", "2023/05/01 (Mon) 09:00"],
        "haystack_sessions": [sess(["How do I bake sourdough bread?", "Use a starter and long fermentation."]),
                              sess(["I moved and now live in Berlin.", "Nice, Berlin is a great city."], answer_idx=0),
                              sess(["Explain kubernetes pods.", "A pod is the smallest deployable unit."])]}
    abstain = dict(q, question_id="q2_abs", answer_session_ids=["answer_x_abs"])
    path.write_text(json.dumps([q, abstain]))
    return q


def test_dates_cache_and_replay(tmp_path):
    q = tiny_dataset(tmp_path / "d.json")
    assert LME.parse_date("2023/05/20 (Sat) 09:00") < LME.parse_date("2023/06/01 (Thu) 10:00")
    emb = LME.CachedEmbedder(HashingEmbedder(), tmp_path / "cache")
    # CachedEmbedder needs embed_many; HashingEmbedder lacks it -> wrapper must still work via fallback
    mem, clock, turns = LME.build_memory(q, emb, HeuristicImportanceScorer())
    assert len(turns) == 6 and clock["t"] == LME.parse_date(q["question_date"])
    stamps = [e.created_at for e in sorted(mem.storage.list(mem.namespace, LME and __import__("agentmem").Layer.EPISODIC),
                                           key=lambda e: e.created_at)]
    assert stamps == sorted(stamps)
    fn = LME.agentmem_system(mem, __import__("agentmem").RetrievalWeights(lexical=0.8))
    got = fn(q["question"], 3)
    assert got[0] == ("answer_1", True)
    m = LME.score(got, q["answer_session_ids"], 3)
    assert m == {"session_any": True, "session_all": True, "turn_any": True}
    emb.save()
    assert LME.CachedEmbedder(HashingEmbedder(), tmp_path / "cache").cache       # persisted and reloadable


def test_reader_formatting_and_answer_parsing():
    ctx = QA.fmt_context([(2, "d2", "user", "later", "s2"), (1, "d1", "user", "earlier", "s1"),
                          (1, "d1", "assistant", "x" * 50, "s1")], max_chars=10)
    assert ctx.index("d1") < ctx.index("d2") and "x" * 11 not in ctx
    assert QA.final_answer("thinking...\nFinal answer: Berlin") == "Berlin"
    assert QA.final_answer("just text") == "just text"
