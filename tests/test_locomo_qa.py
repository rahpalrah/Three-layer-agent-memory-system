"""The QA harness scoring is correct (official-style F1, coverage) and the LLM path runs."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
import locomo_qa as Q  # noqa: E402


def test_f1_and_normalisation():
    assert Q.f1("7 May 2023", "7 May 2023") == 1.0
    assert Q.f1("The 7th of May, 2023", "7 May 2023") == Q.f1("7th of may 2023", "7 may 2023")
    assert 0 < Q.f1("psychology", "Psychology, counseling certification") < 1
    assert Q.f1("unknown", "Paris") == 0.0
    assert Q.coverage("[8 May] Caroline: I went to the LGBTQ support group", "LGBTQ support group") == 1.0


def test_llm_backend_parsing():
    assert Q.make_llm("none") is None
    import pytest
    with pytest.raises(ValueError):
        Q.make_llm("bogus:x")
