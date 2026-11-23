"""Customer Support on Twitter harness, on a tiny synthetic file in the same format
(the real dataset is CC BY-NC-SA and is not redistributed with the repo)."""
import csv
import sys
from pathlib import Path

import pytest

pytest.importorskip("numpy")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
import twcs as T  # noqa: E402
from agentmem import HashingEmbedder  # noqa: E402

FIELDS = ["tweet_id", "author_id", "inbound", "created_at", "text", "response_tweet_id", "in_response_to_tweet_id"]


def write_csv(path):
    rows = [
        # customer A: returns on day 3 about the same cracked screen
        ("1", "custA", "True", "Fri Apr 01 10:00:00 +0000 2016", "My order 4417 arrived with a cracked screen", "2", ""),
        ("2", "brand", "False", "Fri Apr 01 10:05:00 +0000 2016", "Sorry about the cracked screen, we will replace it", "", "1"),
        ("3", "custA", "True", "Sat Apr 02 09:00:00 +0000 2016", "thanks that is great", "", ""),
        ("4", "custA", "True", "Sun Apr 03 18:00:00 +0000 2016", "Any update on the cracked screen replacement?", "", ""),
        # customer B: one day only, never a probe
        ("6", "custB", "True", "Fri Apr 01 11:00:00 +0000 2016", "Where is my refund for the cracked phone case", "", ""),
        ("7", "custB", "True", "Fri Apr 01 12:00:00 +0000 2016", "still waiting on it", "", ""),
        ("8", "custB", "True", "Fri Apr 01 13:00:00 +0000 2016", "hello again", "", ""),
        # customer C: returns, but shares no distinctive words with its history
        ("9", "custC", "True", "Fri Apr 01 11:30:00 +0000 2016", "good morning team", "", ""),
        ("10", "custC", "True", "Sat Apr 02 11:30:00 +0000 2016", "one more question for you", "", ""),
        ("11", "custC", "True", "Sun Apr 03 11:30:00 +0000 2016", "happy sunday everyone", "", ""),
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(FIELDS)
        for r in rows:
            w.writerow((r[0], r[1], r[2], r[3], r[4], r[5], r[6]))
    return path


@pytest.fixture
def data(tmp_path):
    inbound, reply_of = T.load(write_csv(tmp_path / "twcs.csv"))
    df = T.token_df(inbound)
    return inbound, reply_of, df


def test_loading_links_replies_and_separates_customers(data):
    inbound, reply_of, _ = data
    assert [h["tid"] for h in inbound["custA"]] == ["1", "3", "4"]
    assert reply_of["1"]["text"].startswith("Sorry") and reply_of["1"]["author"] == "brand"


def test_probes_only_for_returning_customers_with_distinctive_overlap(data):
    inbound, _, df = data
    probes = {p["cid"]: p for p in T.build_probes(inbound, df)}
    assert set(probes) == {"custA"}                      # B is one day; C shares no distinctive token
    p = probes["custA"]
    assert p["query_tid"] == "4" and p["evidence"] == ["1"]   # the earlier 'cracked screen' tweet
    assert "4417" not in T.rare_tokens("order 4417", df)      # numbers are never distinctive


def test_history_excludes_query_and_pairs_replies(data):
    inbound, reply_of, df = data
    p = T.build_probes(inbound, df)[0]
    eps = T.history_episodes(p, inbound, reply_of)
    assert [e["tid"] for e in eps] == ["1", "2", "3"]    # reply 2 follows tweet 1
    assert eps[1]["role"] == "support" and eps[1]["parent"] == "1"


def test_masked_query_drops_distinctive_words(data):
    _, _, df = data
    assert "cracked" not in T.mask_query("Any update on the cracked screen replacement?", df)
    assert "the" in T.mask_query("Any update on the cracked screen replacement?", df)  # stopwords are kept


def test_quality_run_and_reopen_identical(data, tmp_path):
    inbound, reply_of, df = data
    probes = T.build_probes(inbound, df)
    emb = HashingEmbedder()
    per = T.run_quality(probes, inbound, reply_of, emb, None)
    summary, vs, n = T.score_quality(per, probes, __import__("random").Random(0))
    assert n == 1 and summary["BM25"][10]["any"] == 100.0
    e2e = T.run_end_to_end(probes, inbound, reply_of, emb, None, tmp_path / "t.db")
    assert e2e["customers"] == 1 and e2e["reopen_identical_results"] == 100.0
    assert e2e["evidence_recall_at_10"] == 100.0


def test_brand_mode_isolates_customers(data):
    inbound, reply_of, df = data
    r = T.run_brand("brand", T.build_probes(inbound, df), inbound, reply_of, HashingEmbedder(), None,
                    n_customers=10, seed=0, masked_mode=False, df=df)
    assert r["customers"] == 1
    assert r["systems"]["tag-isolated (this customer only)"]["evidence_recall@10"] == 100.0
