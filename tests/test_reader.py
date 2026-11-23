"""Extractive reader (no model) and the CLI's doctor command."""
import subprocess
import sys

from agentmem.reader import ExtractiveReader


def test_answers_by_question_type():
    r = ExtractiveReader()
    assert r.answer("When did Caroline go to the LGBTQ support group?",
                    ["Caroline: I went to the LGBTQ support group yesterday, it was 7 May 2023 and very useful."]) == "7 May 2023"
    assert r.answer("How many kids does Ada have?", ["Ada: I have 3 kids and a cat."]) == "3"
    assert r.answer("Where does Ada live?", ["Ada: I live in Berlin with my cat."]) == "Berlin"


def test_never_invents_and_says_unknown_without_evidence():
    r = ExtractiveReader()
    assert r.answer("What is the capital of Mars?", ["Ada: I have 3 kids."]) == "unknown"
    assert r.answer("anything", []) == "unknown"


def test_doctor_runs_without_optional_packages():
    out = subprocess.run([sys.executable, "-c", "import sys; from agentmem.cli import main; sys.exit(main(['doctor']))"],
                         capture_output=True, text=True)
    assert out.returncode == 0
    assert "core memory layers" in out.stdout and "extractive answer reader" in out.stdout
