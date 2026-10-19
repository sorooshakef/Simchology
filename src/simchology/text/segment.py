"""Sentence segmentation: regex (no dependencies), spaCy, or SaT for unreliable punctuation."""

from __future__ import annotations

import re
from collections.abc import Sequence

from .._optional import install_hint, require
from .._utils.logging import note
from ..config import DEFAULT_MODELS
from ..exceptions import ConfigurationError
from .spacy_models import default_spacy_model, load_spacy, spacy_model_installed

_ABBREVIATIONS = frozenset(
    "z.b bzw d.h u.a usw ca dr prof nr vgl etc e.g i.e mr mrs ms vs jh jhd chr bspw ggf inkl "
    "evtl st mio mrd abb bd hr fr str tel u.ä o.ä s.o s.u v.a z.t no fig approx".split()
)
_ABBREVIATIONS_NODOT = frozenset(a.replace(".", "") for a in _ABBREVIATIONS)
_CANDIDATE = re.compile(r"[.!?…]+[\"'”»)\]]*\s+")
_TERMINAL = re.compile(r"[.!?…][\"'”“»)\]]*\s*$")


def regex_sentences(text: str) -> list[str]:
    """Split ``text`` into sentences with a conservative rule-based splitter.

    A boundary is a sentence-final punctuation mark followed by whitespace and an
    upper-case letter, digit or quote — unless the period ends an abbreviation, an
    ordinal number ("7. Jahrhundert") or a single letter.
    """
    text = text.strip()
    if not text:
        return []
    out, start = [], 0
    for m in _CANDIDATE.finditer(text):
        end = m.end()
        if end >= len(text):
            break
        nxt = text[end]
        if not (nxt.isupper() or nxt.isdigit() or nxt in "\"'„“«("):
            continue
        punct = m.group(0).strip()
        if punct.startswith(".") and len(punct.rstrip("\"'”»)]")) == 1:
            before = text[start:m.start()].split()
            word = before[-1].lower().rstrip(".") if before else ""
            if (word.isdigit() or len(word) == 1 or word in _ABBREVIATIONS
                    or word.replace(".", "") in _ABBREVIATIONS_NODOT):
                continue
        sentence = text[start:end].strip()
        if sentence:
            out.append(sentence)
        start = end
    tail = text[start:].strip()
    if tail:
        out.append(tail)
    return out


def punctuation_reliability(texts: Sequence[str]) -> float:
    """Share of non-empty texts that end with sentence-final punctuation."""
    texts = [t for t in texts if isinstance(t, str) and t.strip()]
    if not texts:
        return 1.0
    return sum(bool(_TERMINAL.search(t)) for t in texts) / len(texts)


class Segmenter:
    """Split texts into sentences.

    ``kind`` is one of ``"none"`` (whole text), ``"regex"``, ``"spacy"``, ``"sat"`` (the
    neural SaT model of wtpsplit, robust to missing punctuation) or ``"auto"`` (spaCy if a
    pipeline for ``language`` is installed, otherwise regex).
    """

    KINDS = ("auto", "none", "regex", "spacy", "sat")

    def __init__(
        self,
        kind: str = "auto",
        language: str = "de",
        *,
        spacy_model: str | None = None,
        sat_model: str | None = None,
        device: str = "cpu",
        verbose: bool = True,
    ) -> None:
        if kind not in self.KINDS:
            raise ConfigurationError(f"segment must be one of {', '.join(self.KINDS)}; got {kind!r}")
        self.requested = kind
        self.language = language
        self.spacy_model = spacy_model
        self.sat_model = sat_model or DEFAULT_MODELS["sat"]
        self.device = device
        self.verbose = verbose
        self._kind: str | None = None
        self._sat = None
        self._cache: dict[str, list[str]] = {}

    @property
    def kind(self) -> str:
        if self._kind is None:
            if self.requested == "auto":
                name = self.spacy_model or default_spacy_model(self.language)
                self._kind = "spacy" if spacy_model_installed(name) else "regex"
            else:
                self._kind = self.requested
        return self._kind

    def describe(self) -> str:
        if self.kind == "spacy":
            return f"spaCy ({self.spacy_model or default_spacy_model(self.language)})"
        if self.kind == "sat":
            return f"SaT ({self.sat_model})"
        return {"none": "no segmentation", "regex": "rule-based sentence splitting"}[self.kind]

    def hint_if_unreliable(self, texts: Sequence[str]) -> None:
        if self.kind in ("regex", "spacy"):
            share = punctuation_reliability(texts)
            if share < 0.9:
                note(
                    f"Only {share:.0%} of the responses end with punctuation. Sentence "
                    "splitting relies on punctuation; consider segment='sat' "
                    f"({install_hint('wtpsplit')}), which does not.",
                    self.verbose,
                )

    def split(self, texts: Sequence[str]) -> list[list[str]]:
        todo = [t for t in dict.fromkeys(texts) if t not in self._cache]
        if todo:
            for t, sents in zip(todo, self._split_uncached(todo)):
                sents = [s.strip() for s in sents if s and s.strip()]
                self._cache[t] = sents or ([t.strip()] if t.strip() else [""])
        return [self._cache[t] for t in texts]

    def _split_uncached(self, texts: list[str]) -> list[list[str]]:
        kind = self.kind
        if kind == "none":
            return [[t] for t in texts]
        if kind == "regex":
            return [regex_sentences(t) for t in texts]
        if kind == "spacy":
            nlp = load_spacy(self.language, self.spacy_model)
            return [[s.text for s in doc.sents] for doc in nlp.pipe(texts, batch_size=64)]
        if kind == "sat":
            if self._sat is None:
                wtpsplit = require("wtpsplit", feature="sentence segmentation with SaT", extra="sat")
                sat = wtpsplit.SaT(self.sat_model)
                if self.device.startswith("cuda"):
                    sat.half().to(self.device)
                self._sat = sat
            return [list(s) for s in self._sat.split(list(texts))]
        raise AssertionError(kind)
