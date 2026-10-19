"""String-based inference: Jaccard similarity and edit distances (Levenshtein, OSA, Damerau-Levenshtein)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..exceptions import ConfigurationError
from ..explain.explanation import Explanation
from ..text.normalize import light_normalize
from ..text.tokenize import regex_tokens
from .base import MethodInfo, RepresentationMethod
from .registry import register_method


def _ngrams(seq: Sequence, n: int) -> list:
    if len(seq) < n:
        return [tuple(seq)] if seq else []
    return [tuple(seq[i:i + n]) for i in range(len(seq) - n + 1)]


@register_method
class JaccardMethod(RepresentationMethod):
    """Jaccard similarity: shared units divided by all units of both texts.

    Parameters
    ----------
    unit:
        ``"auto"`` (lemmas if a spaCy pipeline is installed, else surface tokens),
        ``"surface"``, ``"lemma"``, ``"word_ngram"`` or ``"char_ngram"``.
    n:
        n-gram length for the n-gram units.
    variant:
        ``"jaccard"`` (|A∩B| / |A∪B|, symmetric) or ``"containment"`` (|A∩B| / |B|: the
        share of the reference's units found in the response).
    """

    info = MethodInfo(
        name="jaccard", family="string",
        description="Jaccard similarity of preprocessed token sets",
        supports_explain=True, aliases=("jaccard_similarity",),
    )
    defaults = {"unit": "auto", "n": 3, "variant": "jaccard"}

    def __init__(self, **params):
        super().__init__(**params)
        if self.params["unit"] not in ("auto", "surface", "lemma", "word_ngram", "char_ngram"):
            raise ConfigurationError("unit must be auto, surface, lemma, word_ngram or char_ngram")
        if self.params["variant"] not in ("jaccard", "containment"):
            raise ConfigurationError("variant must be 'jaccard' or 'containment'")

    def _units(self, texts: list[str]) -> list[frozenset]:
        unit, n = self.params["unit"], int(self.params["n"])
        if unit in ("auto", "surface", "lemma"):
            return [frozenset(t) for t in self.preprocessor.tokens(texts, unit)]
        toks = self.preprocessor.tokens(texts)
        if unit == "word_ngram":
            return [frozenset(_ngrams(t, n)) for t in toks]
        return [frozenset("".join(g) for g in _ngrams(" ".join(t), n)) for t in toks]

    def encode(self, texts: list[str]) -> list[frozenset]:
        return self._units(texts)

    def compare(self, a: list[frozenset], b: list[frozenset]) -> np.ndarray:
        containment = self.params["variant"] == "containment"
        out = np.zeros(len(a))
        for i, (x, y) in enumerate(zip(a, b)):
            denom = len(y) if containment else len(x | y)
            out[i] = len(x & y) / denom if denom else 0.0
        return out

    def describe(self) -> str:
        kind = "containment" if self.params["variant"] == "containment" else "similarity"
        return f"Jaccard {kind} of {self.preprocessor.unit if self.params['unit'] == 'auto' else self.params['unit']} sets"

    def explain_pair(self, left: str, right: str, top_k: int = 10) -> Explanation:
        a, b = self._units([left, right])

        def fmt(u):
            return " ".join(u) if isinstance(u, tuple) else str(u)
        return Explanation(
            self.name, left, right, float(self.compare([a], [b])[0]),
            shared=sorted(fmt(u) for u in a & b), only_response=sorted(fmt(u) for u in a - b),
            only_reference=sorted(fmt(u) for u in b - a),
        )


class _EditSimilarity(RepresentationMethod):
    """Edit-distance similarity ``1 - distance / max(len)``.

    Parameters
    ----------
    level:
        ``"char"`` (characters) or ``"token"`` (whole words as units).
    normalize:
        True (default): similarity in [0, 1]. False: the negative raw distance, which
        reproduces analyses that thresholded raw edit distances.
    preprocess:
        True (default): compare preprocessed tokens (lower-cased, without punctuation
        and stop words; at character level the tokens are joined with "_", as in the
        original R scripts). False: compare the texts as written (whitespace-normalised).
    """

    _metric: str = "Levenshtein"
    defaults = {"level": "char", "normalize": True, "preprocess": True, "unit": "surface"}

    def __init__(self, **params):
        super().__init__(**params)
        if self.params["level"] not in ("char", "token"):
            raise ConfigurationError("level must be 'char' or 'token'")

    def score_range(self):
        return (0.0, 1.0) if self.params["normalize"] else (None, 0.0)

    def encode(self, texts: list[str]):
        if self.params["preprocess"]:
            toks = self.preprocessor.tokens(texts, self.params["unit"])
        else:
            toks = [tuple(regex_tokens(light_normalize(t).lower())) for t in texts]
        if self.params["level"] == "token":
            return [tuple(t) for t in toks]
        if self.params["preprocess"]:
            return ["_".join(t) for t in toks]
        return [light_normalize(t) for t in texts]

    def _scorer(self):
        from rapidfuzz import distance

        metric = getattr(distance, self._metric)
        return metric.normalized_similarity if self.params["normalize"] else metric.distance

    def compare(self, a, b) -> np.ndarray:
        from rapidfuzz import process

        values = process.cpdist(a, b, scorer=self._scorer(), workers=-1)
        values = np.asarray(values, dtype=float)
        return values if self.params["normalize"] else -values

    def describe(self) -> str:
        level = "character" if self.params["level"] == "char" else "token"
        norm = "normalised " if self.params["normalize"] else ""
        return f"{norm}{self.info.description} at {level} level"

    def explain_pair(self, left: str, right: str, top_k: int = 10) -> Explanation:
        from rapidfuzz import distance

        a, b = self.encode([left, right])
        metric = getattr(distance, self._metric)
        d = metric.distance(a, b)
        exp = Explanation(self.name, left, right, float(self.compare([a], [b])[0]),
                          components={"distance": float(d), "max_length": float(max(len(a), len(b)))})
        if hasattr(distance.Levenshtein, "editops"):
            ops = distance.Levenshtein.editops(a, b)
            exp.notes.append(
                f"{sum(o.tag == 'replace' for o in ops)} substitutions, "
                f"{sum(o.tag == 'insert' for o in ops)} insertions, "
                f"{sum(o.tag == 'delete' for o in ops)} deletions (Levenshtein alignment)"
            )
        exp.response_tokens = list(a) if isinstance(a, tuple) else None
        exp.reference_tokens = list(b) if isinstance(b, tuple) else None
        return exp


@register_method
class LevenshteinMethod(_EditSimilarity):
    info = MethodInfo(
        name="levenshtein", family="string", description="Levenshtein edit-distance similarity",
        supports_explain=True, aliases=("lv", "edit_distance"),
    )
    _metric = "Levenshtein"


@register_method
class OSAMethod(_EditSimilarity):
    info = MethodInfo(
        name="osa", family="string",
        description="optimal string alignment (restricted Damerau-Levenshtein) similarity",
        supports_explain=True, aliases=("optimal_string_alignment",),
    )
    _metric = "OSA"


@register_method
class DamerauLevenshteinMethod(_EditSimilarity):
    info = MethodInfo(
        name="damerau_levenshtein", family="string",
        description="Damerau-Levenshtein edit-distance similarity (adjacent transpositions count as one edit)",
        supports_explain=True, aliases=("dl", "damerau"),
    )
    _metric = "DamerauLevenshtein"
