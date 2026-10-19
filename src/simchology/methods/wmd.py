"""Word Mover's Distance (Kusner et al., 2015) over static word vectors.

The distance between two texts is the minimum total distance their words have to
"travel" in the embedding space to turn one text's (normalised bag-of-words)
distribution into the other's, solved exactly as an optimal-transport problem with
POT. Similarity is reported as ``1 / (1 + distance)`` (1 for identical texts).
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from .._optional import require
from ..exceptions import ConfigurationError
from ..explain.explanation import Explanation
from .base import MethodInfo, RepresentationMethod
from .registry import register_method
from .static_embeddings import _unit, _WordVectorMixin


@register_method
class WMDMethod(_WordVectorMixin, RepresentationMethod):
    """Word Mover's Distance similarity ``1 / (1 + WMD)``.

    Parameters
    ----------
    vectors:
        Word-vector source, as for ``word_vectors`` (default: the spaCy pipeline's vectors).
    cost:
        ``"euclidean"`` (default, as in the original WMD) or ``"cosine"`` (1 - cosine).
    normalize_vectors:
        Scale word vectors to unit length before computing distances (default True),
        so that distances do not depend on vector norms (which track word frequency).
    """

    info = MethodInfo(
        name="wmd", family="alignment", description="Word Mover's Distance similarity",
        score_range=(0.0, 1.0), supports_explain=True, cost_tier="moderate",
        extras=("wmd",), requires_modules=("ot",), aliases=("word_movers_distance",),
    )
    defaults = {"vectors": "spacy", "cost": "euclidean", "normalize_vectors": True}

    def __init__(self, **params):
        super().__init__(**params)
        if self.params["cost"] not in ("euclidean", "cosine"):
            raise ConfigurationError("cost must be 'euclidean' or 'cosine'")

    def encode(self, texts: list[str]) -> list[tuple[list[str], np.ndarray, np.ndarray]]:
        """Per text: unique words, their vectors and their nBOW weights."""
        out = []
        for t in texts:
            toks, vecs = self.token_vectors(t)
            counts = Counter(toks)
            words = list(counts)
            if words:
                first = {w: toks.index(w) for w in words}
                mat = vecs[[first[w] for w in words]].astype(np.float64)
                if self.params["normalize_vectors"]:
                    mat = _unit(mat)
                weights = np.array([counts[w] for w in words], dtype=np.float64)
                weights /= weights.sum()
            else:
                mat, weights = np.zeros((0, self.store().dim)), np.zeros(0)
            out.append((words, mat, weights))
        return out

    def _cost(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        if self.params["cost"] == "cosine":
            return np.clip(1 - _unit(a) @ _unit(b).T, 0.0, 2.0)
        diff = a[:, None, :] - b[None, :, :]
        return np.sqrt(np.maximum((diff**2).sum(-1), 0.0))

    def _distance(self, x, y) -> float:
        ot = require("ot", feature="the method 'wmd'", extra="wmd")
        (wa, va, pa), (wb, vb, pb) = x, y
        if not len(wa) or not len(wb):
            return float("inf")
        if wa == wb and np.allclose(pa, pb):
            return 0.0
        return float(ot.emd2(pa, pb, self._cost(va, vb)))

    def compare(self, a, b) -> np.ndarray:
        d = np.array([self._distance(x, y) for x, y in zip(a, b)], dtype=float)
        return np.where(np.isinf(d), 0.0, 1.0 / (1.0 + d))

    def describe(self) -> str:
        src = self._store.description if self._store is not None else "static word vectors"
        return f"Word Mover's Distance ({self.params['cost']} cost, {src}), reported as 1/(1+distance)"

    def explain_pair(self, left: str, right: str, top_k: int = 10) -> Explanation:
        ot = require("ot", feature="the method 'wmd'", extra="wmd")
        x, y = self.encode([left, right])
        (wa, va, pa), (wb, vb, pb) = x, y
        d = self._distance(x, y)
        score = 0.0 if np.isinf(d) else 1.0 / (1.0 + d)
        exp = Explanation(self.name, left, right, float(score), components={"distance": float(d)},
                          response_tokens=wa, reference_tokens=wb)
        if not wa or not wb:
            exp.notes.append("one of the texts has no words with a vector")
            return exp
        cost = self._cost(va, vb)
        plan = ot.emd(pa, pb, cost)
        flows = [(i, j) for i in range(len(wa)) for j in range(len(wb)) if plan[i, j] > 1e-12]
        flows.sort(key=lambda ij: -plan[ij])
        rows = [{"side": "response", "token": wa[i], "matched_token": wb[j],
                 "similarity": float(1.0 / (1.0 + cost[i, j])), "weight": float(plan[i, j])}
                for i, j in flows[: top_k or None]]
        exp.alignments = pd.DataFrame(rows, columns=["side", "token", "matched_token", "similarity", "weight"])
        exp.matrix = plan
        exp.notes.append("largest flows of the optimal transport plan: 'weight' is the share of "
                         "the text's mass moved, 'similarity' is 1/(1+distance) of the two words")
        return exp
