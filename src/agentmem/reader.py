"""Extractive reader: answers a question from retrieved memory without any language model.

Given the question and the retrieved memory texts, it

  1. ranks candidate sentences by IDF-weighted overlap with the question;
  2. inside the best sentence, proposes answer candidates of the type the question asks for
     (dates, numbers, capitalised names), excluding words from the question;
  3. returns the candidate closest to the question terms, or a trimmed sentence as a fallback.

It copies spans from memory, so it cannot invent facts, but it is much weaker than an LLM. Its
purpose is an answer-accuracy baseline that runs anywhere with no model.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import List, Sequence

from .embeddings import tokenize

_SENT = re.compile(r"(?<=[.!?])\s+|\n+")
_SPEAKER = re.compile(r"^\s*[\[\(]?[^:\n]{1,40}[\]\)]?:\s*")
_DATE = re.compile(r"\b\d{1,2}\s+(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?,?\s*\d{4}\b"
                   r"|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{1,2}(?:,?\s*\d{4})?\b"
                   r"|\b\d{4}\b", re.I)
_CAP = re.compile(r"\b[A-Z][\w'&-]*(?:\s+[A-Z][\w'&-]*)*")
_NUM = re.compile(r"\b\d+(?:[.,]\d+)?\b")
_PRONOUNS = {"i", "me", "my", "we", "our", "you", "your", "he", "she", "it", "they", "the", "a", "an", "this", "that"}
_WH = [("how many", "number"), ("how much", "number"), ("how long", "number"),
       ("when", "date"), ("where", "name"), ("who", "name"), ("which", "name"), ("what", "any")]


class ExtractiveReader:
    def __init__(self, max_answer_words: int = 8) -> None:
        self.max_answer_words = max_answer_words

    @staticmethod
    def _sentences(texts: Sequence[str]) -> List[str]:
        out = []
        for t in texts:
            t = _SPEAKER.sub("", t or "")
            out += [s.strip() for s in _SENT.split(t) if s.strip()]
        return out

    @staticmethod
    def answer_kind(question: str) -> str:
        q = question.lower()
        for key, kind in _WH:
            if key in q:
                return kind
        return "any"

    def answer(self, question: str, texts: Sequence[str]) -> str:
        sents = self._sentences(texts)
        if not sents:
            return "unknown"
        q_tokens = set(tokenize(question))
        df = Counter(w for s in sents for w in set(tokenize(s)))
        n = len(sents)

        def score(s: str) -> float:
            return sum(math.log(1 + n / (1 + df[w])) for w in q_tokens & set(tokenize(s)))

        best = max(sents, key=score)
        if score(best) == 0:
            return "unknown"
        kind = self.answer_kind(question)
        q_words = {w.lower() for w in re.findall(r"[\w']+", question)}
        cands: List[str] = []
        if kind in ("date", "any"):
            cands += [m.group(0) for m in _DATE.finditer(best)]
        if kind in ("number", "any"):
            cands += [m.group(0) for m in _NUM.finditer(best)]
        if kind in ("name", "any"):
            cands += [m.group(0).strip() for m in _CAP.finditer(best)]
        cands = [c for c in cands if c and c.lower() not in q_words and c.lower() not in _PRONOUNS
                 and (c[0].isdigit() or len(c) > 1)]
        if not cands and kind == "date":  # a date question with no date in the best sentence
            cands = [m.group(0) for m in _NUM.finditer(best)]
        if cands:
            lower = best.lower()
            qpos = [lower.find(w) for w in q_tokens if lower.find(w) >= 0] or [0]
            return min(cands, key=lambda c: min(abs(best.find(c) - p) for p in qpos))
        words = [w for w in best.split() if re.sub(r"\W", "", w).lower() not in q_words]
        return " ".join(words[: self.max_answer_words]) or best
